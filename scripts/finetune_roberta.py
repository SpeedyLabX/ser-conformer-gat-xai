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
import numpy as np

# labels for HF CrossEntropy ignore index
IGNORE_INDEX = -100

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

from serxai.data.iemocap_dataset import read_manifest, train_val_split, train_test_by_sessions
from serxai.data.collators import TextCollator
from serxai.data.labels import canonicalize_label


class ManifestDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text = r.get('text','')
        lbl = canonicalize_label(r.get('label', -1))
        toks = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze(0) for k,v in toks.items()}
        # HuggingFace CrossEntropyLoss uses ignore_index=-100 by default; map unknown labels to that
        if lbl is None or (isinstance(lbl, int) and lbl < 0):
            out_lbl = IGNORE_INDEX
        else:
            out_lbl = int(lbl)
        item['labels'] = torch.tensor(out_lbl, dtype=torch.long)
        return item


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='data/iemocap_manifest.jsonl')
    p.add_argument('--backbone', default='roberta-base')
    p.add_argument('--out_dir', default='artifacts')
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--eval_batch_size', type=int, default=0, help='Evaluation batch size (0 -> use train batch size)')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--accumulation_steps', type=int, default=2)
    p.add_argument('--num_class', type=int, default=6)
    p.add_argument('--split_by_session', action='store_true', help='If set, split train/test by session ids (train: 1-4, test: 5)')
    p.add_argument('--train_sessions', type=str, default='1,2,3,4', help='Comma-separated train session ids')
    p.add_argument('--test_sessions', type=str, default='5', help='Comma-separated test session ids')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--fp16', action='store_true', help='Use mixed precision (torch.cuda.amp) in fallback training loop')
    p.add_argument('--evaluation_strategy', type=str, default='epoch', help='HF Trainer evaluation strategy ("no"/"steps"/"epoch")')
    p.add_argument('--save_total_limit', type=int, default=3, help='Max number of saved checkpoints')
    p.add_argument('--load_best_model_at_end', action='store_true', help='If set, load best model at end by metric')
    p.add_argument('--gradient_checkpointing', action='store_true', help='Enable model gradient checkpointing to save memory')
    p.add_argument('--early_stopping_patience', type=int, default=0, help='Enable early stopping with this patience (0 disables)')
    p.add_argument('--max_train_samples', type=int, default=0, help='Limit number of training samples (0 = use all)')
    p.add_argument('--max_val_samples', type=int, default=0, help='Limit number of validation samples (0 = use all)')
    p.add_argument('--max_grad_norm', type=float, default=1.0, help='Max grad norm for clipping in fallback training')
    p.add_argument('--use_tqdm', action='store_true', help='Enable tqdm progress bars in fallback loop')
    p.add_argument('--resume_from_checkpoint', default=None, help='Path to checkpoint dir or file to resume fallback training')
    p.add_argument('--keep_unlabeled', action='store_true', help='If set, keep unlabeled records (default: drop unlabeled)')
    p.add_argument('--metric_for_best_model', type=str, default='eval_f1', help="Metric name to use for selecting best model (e.g. 'eval_f1' or 'eval_accuracy'). Empty to use eval_loss.")
    p.add_argument('--metric_greater_is_better', type=lambda x: str(x).lower() in ('1','true','yes'), default=True, help='Whether larger value of metric is better (default True for accuracy/F1)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    manifest = read_manifest(Path(args.manifest))
    if args.split_by_session:
        try:
            train_sessions = tuple(int(x) for x in args.train_sessions.split(','))
            test_sessions = tuple(int(x) for x in args.test_sessions.split(','))
        except Exception:
            train_sessions = (1, 2, 3, 4)
            test_sessions = (5,)
        train_records, val_records = train_test_by_sessions(manifest, train_sessions=train_sessions, test_sessions=test_sessions)
    else:
        train_records, val_records = train_val_split(manifest, val_ratio=0.1)

    # Quick diagnostics: count labeled vs unlabeled and optionally filter
    def _is_labeled(rec):
        return canonicalize_label(rec.get('label', None)) >= 0

    n_total = len(manifest)
    n_train = len(train_records)
    n_val = len(val_records)
    n_train_labeled = sum(1 for r in train_records if _is_labeled(r))
    n_val_labeled = sum(1 for r in val_records if _is_labeled(r))
    print(f"Manifest total={n_total} train={n_train} val={n_val} | labeled: train={n_train_labeled} val={n_val_labeled}")

    # By default, drop unlabeled records from training and validation because
    # they map to IGNORE_INDEX and can cause evaluation to produce NaN when
    # an eval split contains zero labeled samples. Provide a flag to keep them
    # if user explicitly wants to train with unlabeled examples.
    if not getattr(args, 'keep_unlabeled', False):
        train_records = [r for r in train_records if _is_labeled(r)]
        val_records = [r for r in val_records if _is_labeled(r)]
        print(f"Filtered unlabeled records -> train={len(train_records)} val={len(val_records)}")

    if len(val_records) == 0:
        raise RuntimeError('Validation set contains 0 labeled examples after filtering. Rebuild manifest or adjust split.')

    if HF_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=args.num_class)
        print('DEBUG: Loaded tokenizer and model from backbone', args.backbone, flush=True)
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
    # allow eval batch size override
    val_ds = ManifestDataset(val_slice, tokenizer, max_length=args.max_length)
    # Try using HF Trainer; if the installed transformers version's TrainingArguments
    # signature is incompatible, fall back to a small manual training loop.
    def save_checkpoint(model, optimizer, scaler, out_dir, step=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else {},
        }
        if scaler is not None:
            state['scaler_state_dict'] = scaler.state_dict()
        if step is not None:
            state['step'] = step
        # primary fallback checkpoint (contains optimizer/scaler)
        torch.save(state, out_dir / 'fallback_ckpt.pth')
        # also save HF-style model file for downstream evaluation convenience
        try:
            torch.save(state['model_state_dict'], out_dir / 'pytorch_model.bin')
        except Exception:
            try:
                torch.save(model.state_dict(), out_dir / 'pytorch_model.bin')
            except Exception:
                pass
        # if a tokenizer object is available in the outer scope, try to save it too
        try:
            # tokenizer may not be defined in this scope; guard with globals
            tok = globals().get('tokenizer', None)
            if tok is not None and hasattr(tok, 'save_pretrained'):
                tok.save_pretrained(str(out_dir))
        except Exception:
            pass


    def load_checkpoint(model, optimizer, scaler, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', {}))
        if 'optimizer_state_dict' in ckpt and optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt and scaler is not None:
            try:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            except Exception:
                pass
        return ckpt.get('step', 0)


    def simple_train(model, train_dataset, val_dataset, device, args):
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        import math
        # optional tqdm
        if getattr(args, 'use_tqdm', False):
            try:
                from tqdm.auto import tqdm
            except Exception:
                tqdm = None
        else:
            tqdm = None
        use_fp16 = args.fp16 and device.type == 'cuda'
        def _create_autocast_ctx():
            if use_fp16:
                if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                    try:
                        return torch.amp.autocast('cuda')
                    except TypeError:
                        # very recent torch expects keyword form
                        return torch.amp.autocast(device_type='cuda')
                return torch.cuda.amp.autocast()
            from contextlib import nullcontext
            return nullcontext()

        if use_fp16 and hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            try:
                scaler = torch.amp.GradScaler('cuda', enabled=True)
            except TypeError:
                try:
                    scaler = torch.amp.GradScaler(device_type='cuda', enabled=True)  # keyword-only form
                except TypeError:
                    scaler = torch.amp.GradScaler(enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

        # early stopping / resume bookkeeping
        best_val = -1.0
        best_epoch = -1
        epochs_no_improve = 0
        resume_step = 0

        model.to(device)
        eval_bs = args.eval_batch_size if getattr(args, 'eval_batch_size', 0) and args.eval_batch_size > 0 else args.batch_size
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=eval_bs, num_workers=2, pin_memory=True)
        optimizer = AdamW(model.parameters(), lr=args.lr)

        # optionally resume
        if getattr(args, 'resume_from_checkpoint', None):
            ckpt_p = Path(args.resume_from_checkpoint)
            if ckpt_p.is_dir():
                ckpt_file = ckpt_p / 'fallback_ckpt.pth'
            else:
                ckpt_file = ckpt_p
            if ckpt_file.exists():
                try:
                    resume_step = load_checkpoint(model, optimizer, scaler, str(ckpt_file), device)
                    print(f"Resumed fallback training from {ckpt_file} (step={resume_step})")
                except Exception as e:
                    print('Failed to load fallback checkpoint:', e)

        model.train()
        total_steps = 0
        for epoch in range(args.epochs):
            running_loss = 0.0
            iterator = enumerate(train_loader)
            if tqdm is not None:
                iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
            for step, batch in iterator:
                # skip batches if resuming mid-epoch
                if resume_step and step < resume_step:
                    continue
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
                        with _create_autocast_ctx():
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
                        with _create_autocast_ctx():
                            outputs = model(**input_batch)
                            loss = None
                    else:
                        outputs = model(**input_batch)
                        loss = None

                if loss is not None:
                    running_loss += float(loss.detach().cpu().item())

                if (step + 1) % args.accumulation_steps == 0:
                    # gradient clipping to avoid explosion
                    try:
                        # compute pre-clip grad norm (safe): sqrt(sum(norm(p.grad)^2)) over non-None grads
                        import math
                        total_norm_sq = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                try:
                                    gnorm = float(p.grad.data.norm(2).item())
                                except Exception:
                                    # numeric safety: skip unusual grads
                                    continue
                                total_norm_sq += gnorm * gnorm
                        pre_clip_norm = math.sqrt(total_norm_sq)
                        # actually clip and capture the returned norm (this is the norm AFTER clipping)
                        clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(args, 'max_grad_norm', 1.0)))
                    except Exception:
                        pre_clip_norm = None
                        clipped_norm = None

                    if use_fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    # print some grad diagnostic info occasionally (per accumulation step)
                    try:
                        if pre_clip_norm is not None and clipped_norm is not None:
                            print({'step': total_steps, 'pre_grad_norm': float(pre_clip_norm), 'clipped_grad_norm': float(clipped_norm), 'learning_rate': optimizer.param_groups[0].get('lr', None)})
                        else:
                            # fallback lightweight logging
                            print({'step': total_steps, 'clipped_grad_norm': float(getattr(args, 'max_grad_norm', 1.0)), 'learning_rate': optimizer.param_groups[0].get('lr', None)})
                    except Exception:
                        pass

                    total_steps += 1

            avg_loss = running_loss / max(1, len(train_loader))
            print(f"Epoch {epoch+1}/{args.epochs} finished - avg_loss={avg_loss:.4f} steps={total_steps}")

            # quick evaluation pass
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                v_iter = val_loader
                if tqdm is not None:
                    v_iter = tqdm(val_loader, total=len(val_loader), desc='Validation')
                    for vb in v_iter:
                        vb = {k: v.to(device) for k, v in vb.items()}
                        out = model(**{k: v for k, v in vb.items() if k!='labels'})
                        logits = out.logits
                        preds = torch.argmax(logits, dim=-1)
                        labels = vb.get('labels')
                        if labels is not None:
                            # respect ignore index when computing accuracy
                            mask = labels != IGNORE_INDEX
                            if mask.sum().item() > 0:
                                preds_masked = preds[mask]
                                labels_masked = labels[mask]
                                correct += (preds_masked == labels_masked).sum().item()
                                total += labels_masked.size(0)

            acc = (correct / total) if total>0 else 0.0
            print(f"Validation accuracy after epoch {epoch+1}: {acc:.4f}")
            # early stopping checks
            if acc > best_val:
                best_val = acc
                best_epoch = epoch
                epochs_no_improve = 0
                # save best checkpoint
                try:
                    save_checkpoint(model, optimizer, scaler, args.out_dir, step=total_steps)
                    print('Saved fallback best checkpoint to', args.out_dir)
                except Exception as e:
                    print('Failed to save fallback checkpoint:', e)
            else:
                epochs_no_improve += 1

            model.train()

            if args.early_stopping_patience and args.early_stopping_patience > 0:
                if epochs_no_improve >= args.early_stopping_patience:
                    print(f"Early stopping triggered (patience={args.early_stopping_patience}). Best epoch: {best_epoch+1}")
                    return

        # save model and tokenizer
        os.makedirs(args.out_dir, exist_ok=True)
        try:
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
        except Exception:
            # fallback: save state_dict
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'pytorch_model.bin'))
        print('Saved RoBERTa model to', args.out_dir)

    try:
        # build TrainingArguments; include metric_for_best_model/greatness if requested
        ta_kwargs = dict(
            output_dir=args.out_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=(args.eval_batch_size if getattr(args, 'eval_batch_size', 0) and args.eval_batch_size > 0 else args.batch_size),
            # some older HF versions don't accept evaluation_strategy kwarg; pass minimal set
            save_strategy=args.evaluation_strategy,
            evaluation_strategy=args.evaluation_strategy,
            gradient_accumulation_steps=args.accumulation_steps,
            learning_rate=args.lr,
            fp16=args.fp16,
            max_grad_norm=float(getattr(args, 'max_grad_norm', 1.0)),
            logging_steps=10,
            load_best_model_at_end=args.load_best_model_at_end,
            save_total_limit=args.save_total_limit,
        )
        if args.metric_for_best_model:
            ta_kwargs['metric_for_best_model'] = args.metric_for_best_model
            ta_kwargs['greater_is_better'] = bool(args.metric_greater_is_better)

        print('DEBUG: Building TrainingArguments...', flush=True)
        filtered_ta_kwargs = dict(ta_kwargs)
        dropped_kwargs = []
        while True:
            try:
                training_args = TrainingArguments(**filtered_ta_kwargs)
                break
            except TypeError as te:
                msg = str(te)
                if 'unexpected keyword argument' in msg:
                    bad_kw = msg.split("'")[1]
                    if bad_kw in filtered_ta_kwargs:
                        filtered_ta_kwargs.pop(bad_kw)
                        dropped_kwargs.append(bad_kw)
                        print(f"DEBUG: TrainingArguments dropped unsupported kwarg '{bad_kw}' and retrying", flush=True)
                        continue
                raise

        for dropped in dropped_kwargs:
            if dropped in ta_kwargs and hasattr(training_args, dropped):
                try:
                    setattr(training_args, dropped, ta_kwargs[dropped])
                    print(f"DEBUG: TrainingArguments attr '{dropped}' set post-init for compatibility", flush=True)
                except Exception:
                    pass

        trainer_kwargs = dict(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
        print('DEBUG: Trainer kwargs prepared (model, training_args, train_dataset, eval_dataset)', flush=True)
        if args.early_stopping_patience and args.early_stopping_patience > 0:
            try:
                from transformers import EarlyStoppingCallback
                trainer_kwargs['callbacks'] = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
            except Exception:
                pass

        # optional compute_metrics (accuracy + macro F1) to choose best model by desired metric
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            try:
                from sklearn.metrics import accuracy_score, f1_score
            except Exception:
                return {}
            preds = np.argmax(logits, axis=-1)
            # respect IGNORE_INDEX
            mask = labels != IGNORE_INDEX
            if mask.ndim > 1:
                # sometimes labels come as shape (n,1)
                labels = labels.squeeze()
                mask = labels != IGNORE_INDEX
            labels_f = labels[mask]
            preds_f = preds[mask]
            if labels_f.size == 0:
                return {}
            acc = float(accuracy_score(labels_f, preds_f))
            f1 = float(f1_score(labels_f, preds_f, average='macro', zero_division=0))
            return {'accuracy': acc, 'f1': f1, 'eval_accuracy': acc, 'eval_f1': f1}

        trainer_kwargs['compute_metrics'] = compute_metrics
        print('DEBUG: Creating Trainer instance...', flush=True)
        trainer = Trainer(**trainer_kwargs)
        print('DEBUG: Trainer created. Calling trainer.train()...', flush=True)
        trainer.train()
        print('DEBUG: trainer.train() finished. Saving model...', flush=True)
        trainer.save_model(args.out_dir)
        # ensure tokenizer is saved into output directory for offline evaluation
        try:
            tokenizer.save_pretrained(args.out_dir)
        except Exception:
            pass
        print('Saved RoBERTa model to', args.out_dir)
    except Exception as te:
        print('HF Trainer not available or incompatible, falling back to simple_train.\n', te)
        simple_train(model, train_ds, val_ds, device, args)

    # If HF is not available at all, inform the user
    if not HF_AVAILABLE:
        print('transformers not available; aborting. Install transformers to run full finetune.')


if __name__ == '__main__':
    main()
