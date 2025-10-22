"""Full fine-tune script for RoBERTa-base (or other HF backbones).

This script attempts to use the HuggingFace Trainer when available. It provides
options to run a lightweight training loop if HF Trainer is not desired.

NOTE: On a 4GB GPU this may OOM; use very small batch sizes and gradient accumulation.
"""
import argparse
from pathlib import Path
import json
import os

import torch
from torch.utils.data import Dataset

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

from serxai.data.iemocap_dataset import read_manifest, train_val_split
from serxai.data.collators import TextCollator


class ManifestDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

        # canonical label map (keep in sync with TextCollator)
        self.label_map = {
            "neutral": 0,
            "angry": 1,
            "frustration": 2,
            "happy": 3,
            "sad": 4,
            "fear": 5,
            "disgust": 6,
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text = r.get('text','')
        lbl = r.get('label', -1)
        # canonicalize label to int if it is string-like
        if isinstance(lbl, str):
            key = lbl.lower().strip()
            if key in self.label_map:
                lbl = self.label_map[key]
            else:
                try:
                    lbl = int(lbl)
                except Exception:
                    lbl = -1
        toks = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze(0) for k,v in toks.items()}
        item['labels'] = torch.tensor(lbl if lbl is not None else -1, dtype=torch.long)
        return item


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='data/iemocap_manifest.jsonl')
    p.add_argument('--backbone', default='roberta-base')
    p.add_argument('--out_dir', default='artifacts')
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--accumulation_steps', type=int, default=2)
    p.add_argument('--num_class', type=int, default=7)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--fp16', action='store_true', help='Use mixed precision (torch.cuda.amp) in fallback training loop')
    p.add_argument('--evaluation_strategy', type=str, default='epoch', help='HF Trainer evaluation strategy ("no"/"steps"/"epoch")')
    p.add_argument('--save_total_limit', type=int, default=3, help='Max number of saved checkpoints')
    p.add_argument('--load_best_model_at_end', action='store_true', help='If set, load best model at end by metric')
    p.add_argument('--gradient_checkpointing', action='store_true', help='Enable model gradient checkpointing to save memory')
    p.add_argument('--early_stopping_patience', type=int, default=0, help='Enable early stopping with this patience (0 disables)')
    p.add_argument('--max_train_samples', type=int, default=0, help='Limit number of training samples (0 = use all)')
    p.add_argument('--max_val_samples', type=int, default=0, help='Limit number of validation samples (0 = use all)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    manifest = read_manifest(Path(args.manifest))
    train_records, val_records = train_val_split(manifest, val_ratio=0.1)

    if HF_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=args.num_class)
        # optionally enable gradient checkpointing (saves memory at the cost of compute)
        if args.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                # older HF models may not have this API, ignore gracefully
                pass
        # allow quick smoke runs by limiting samples; 0 means use all
        if args.max_train_samples and args.max_train_samples > 0:
            train_slice = train_records[: args.max_train_samples]
        else:
            train_slice = train_records

        if args.max_val_samples and args.max_val_samples > 0:
            val_slice = val_records[: args.max_val_samples]
        else:
            val_slice = val_records

        train_ds = ManifestDataset(train_slice, tokenizer, max_length=args.max_length)
        val_ds = ManifestDataset(val_slice, tokenizer, max_length=args.max_length)
        # Try using HF Trainer; if the installed transformers version's TrainingArguments
        # signature is incompatible, fall back to a small manual training loop.
        def simple_train(model, train_dataset, val_dataset, device, args):
            from torch.utils.data import DataLoader
            from torch.optim import AdamW
            import math
            use_fp16 = args.fp16 and device.type == 'cuda'
            scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

            model.to(device)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True)
            optimizer = AdamW(model.parameters(), lr=args.lr)

            model.train()
            total_steps = 0
            for epoch in range(args.epochs):
                running_loss = 0.0
                for step, batch in enumerate(train_loader):
                    # move tensors to device but keep labels handling separate
                    labels = batch.get('labels')
                    input_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'labels'}

                    if labels is not None:
                        labels = labels.to(device, non_blocking=True)
                        # filter out entries with label < 0 (unlabeled)
                        mask = labels >= 0
                        if mask.sum().item() == 0:
                            # nothing to train on in this batch
                            continue
                        # select only valid entries
                        for k in list(input_batch.keys()):
                            input_batch[k] = input_batch[k][mask]
                        labels = labels[mask]

                        # forward (mixed precision if requested)
                        if use_fp16:
                            with torch.cuda.amp.autocast():
                                outputs = model(**input_batch, labels=labels)
                                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                            scaler.scale(loss / max(1, args.accumulation_steps)).backward()
                        else:
                            outputs = model(**input_batch, labels=labels)
                            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                            (loss / max(1, args.accumulation_steps)).backward()
                    else:
                        # no labels provided; forward only (shouldn't happen for training)
                        if use_fp16:
                            with torch.cuda.amp.autocast():
                                outputs = model(**input_batch)
                                loss = None
                        else:
                            outputs = model(**input_batch)
                            loss = None

                    if loss is not None:
                        running_loss += float(loss.detach().cpu().item())

                    if (step + 1) % args.accumulation_steps == 0:
                        if use_fp16:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        total_steps += 1

                avg_loss = running_loss / max(1, len(train_loader))
                print(f"Epoch {epoch+1}/{args.epochs} finished - avg_loss={avg_loss:.4f} steps={total_steps}")

                # quick evaluation pass
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for vb in val_loader:
                        vb = {k: v.to(device) for k, v in vb.items()}
                        out = model(**{k: v for k, v in vb.items() if k!='labels'})
                        logits = out.logits
                        preds = torch.argmax(logits, dim=-1)
                        labels = vb.get('labels')
                        if labels is not None:
                            correct += (preds == labels).sum().item()
                            total += labels.size(0)

                acc = (correct / total) if total>0 else 0.0
                print(f"Validation accuracy after epoch {epoch+1}: {acc:.4f}")
                model.train()

            # save model and tokenizer
            os.makedirs(args.out_dir, exist_ok=True)
            try:
                model.save_pretrained(args.out_dir)
                tokenizer.save_pretrained(args.out_dir)
            except Exception:
                # fallback: save state_dict
                torch.save(model.state_dict(), os.path.join(args.out_dir, 'pytorch_model.bin'))
            print('Saved RoBERTa model to', args.out_dir)

        # Attempt to use HF Trainer; if incompatible, fall back to simple_train
        try:
            training_args = TrainingArguments(
                output_dir=args.out_dir,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                # some older HF versions don't accept evaluation_strategy kwarg; pass minimal set
                save_strategy='epoch',
                gradient_accumulation_steps=args.accumulation_steps,
                learning_rate=args.lr,
                fp16=args.fp16,
                logging_steps=10,
                load_best_model_at_end=args.load_best_model_at_end,
                save_total_limit=args.save_total_limit,
            )

            trainer_kwargs = dict(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
            if args.early_stopping_patience and args.early_stopping_patience > 0:
                try:
                    from transformers import EarlyStoppingCallback
                    trainer_kwargs['callbacks'] = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
                except Exception:
                    pass

            trainer = Trainer(**trainer_kwargs)
            trainer.train()
            trainer.save_model(args.out_dir)
            print('Saved RoBERTa model to', args.out_dir)
        except Exception as te:
            print('HF Trainer not available or incompatible, falling back to simple_train.\n', te)
            simple_train(model, train_ds, val_ds, device, args)
    else:
        print('transformers not available; aborting. Install transformers to run full finetune.')


if __name__ == '__main__':
    main()
