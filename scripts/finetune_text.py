"""Lightweight finetune script: trains an MLP head on top of TextEncoder pooled outputs.

This script is intentionally dependency-light (no HF Trainer) and usable for quick smoke runs.
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from serxai.data.iemocap_dataset import IEMOCAPTextDataset, read_manifest, train_val_split
from serxai.data.collators import TextCollator
from serxai.models.tokenizer_encoder import TokenizerEncoder
from serxai.models.text_head import Head, instantiate_from_state_dict
from serxai.data.labels import canonicalize_label


def make_dataloader(records, tokenizer_encoder: TokenizerEncoder, batch_size=8, shuffle=False):
    coll = TextCollator(tokenizer_encoder.tokenizer, max_length=128)

    def ds_iter():
        for r in records:
            yield r

    class SimpleIt:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    dataset = SimpleIt(records)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda b: coll(b))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    manifest = read_manifest(Path(args.manifest))
    train_records, val_records = train_val_split(manifest, val_ratio=args.val_ratio)
    tokenc = TokenizerEncoder(backbone=args.backbone, proj_dim=args.proj_dim, force_fallback=args.force_fallback)
    train_loader = make_dataloader(train_records[:200], tokenc, batch_size=args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_records[:100], tokenc, batch_size=args.batch_size, shuffle=False)
    # determine num_class if not provided
    if args.num_class is None:
        labels = [r.get("label", -1) for r in train_records + val_records]
        # canonicalize provided labels (strings->int)
        labels = [canonicalize_label(l) for l in labels]
        labels = [l for l in labels if l is not None and l >= 0]
        if len(labels) == 0:
            num_class = 4
        else:
            num_class = int(max(labels)) + 1
    else:
        num_class = args.num_class

    # head
    head = Head(in_dim=tokenc.encoder.out_dim, n_class=num_class).to(device)
    opt = torch.optim.Adam(list(head.parameters()) + list(tokenc.encoder.parameters()), lr=args.lr) if not args.freeze_backbone else torch.optim.Adam(head.parameters(), lr=args.lr)
    if args.freeze_backbone:
        for p in tokenc.encoder.parameters():
            p.requires_grad = False

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        head.train()
        tot_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = tokenc.encoder(input_ids, attention_mask, return_tokens=False)
            pooled = out["pooled"].to(device)
            logits = head(pooled)
            # filter out -1 labels
            mask = labels >= 0
            if mask.sum() == 0:
                continue
            loss = loss_fn(logits[mask], labels[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()

        print(f"Epoch {epoch} train loss: {tot_loss:.4f}")

    # save head
    save_p = Path(args.out_dir) / "text_head.pth"
    save_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict()}, save_p)
    print("Saved checkpoint to", save_p)


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/iemocap_manifest.jsonl")
    p.add_argument("--backbone", default="fallback")
    p.add_argument("--force_fallback", action="store_true")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--num_class", type=int, default=None)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out_dir", default="artifacts")
    p.add_argument("--checkpoint", default=None, help="Path to checkpoint (for eval only)")
    p.add_argument("--eval_only", action="store_true", help="Only run evaluation using a saved checkpoint")
    args = p.parse_args()
    # if eval_only, run a lightweight evaluation using the checkpoint
    if args.eval_only:
        # determine checkpoint path
        ckpt_p = Path(args.checkpoint) if args.checkpoint else Path(args.out_dir) / "text_head.pth"
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
        # load manifest and val split
        manifest = read_manifest(Path(args.manifest))
        _, val_records = train_val_split(manifest, val_ratio=args.val_ratio)
        tokenc = TokenizerEncoder(backbone=args.backbone, proj_dim=args.proj_dim, force_fallback=args.force_fallback)
        coll = TextCollator(tokenc.tokenizer, max_length=128)
        ckpt = torch.load(str(ckpt_p), map_location=("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu"))
        head = instantiate_from_state_dict(ckpt.get("head", {}), in_dim=tokenc.encoder.out_dim)
        head.load_state_dict(ckpt.get("head", {}))
        head.eval()

        ys_true = []
        ys_pred = []
        for r in val_records:
            batch = coll([r])
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"].numpy()
            with torch.no_grad():
                out = tokenc.encoder(input_ids, attention_mask, return_tokens=False)
                pooled = out["pooled"]
                logits = head(pooled)
                pred = logits.argmax(dim=-1).cpu().numpy()
            ys_true.extend(labels.tolist())
            ys_pred.extend(pred.tolist())

        import numpy as np
        from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix

        ys_true = np.array(ys_true)
        ys_pred = np.array(ys_pred)
        mask = ys_true >= 0
        ys_true = ys_true[mask]
        ys_pred = ys_pred[mask]
        if len(ys_true) == 0:
            print("No labeled samples in validation split")
        else:
            wa = accuracy_score(ys_true, ys_pred)
            ua = recall_score(ys_true, ys_pred, average='macro', zero_division=0)
            mf1 = f1_score(ys_true, ys_pred, average='macro', zero_division=0)
            cm = confusion_matrix(ys_true, ys_pred).tolist()
            out = {
                "n_samples": int(len(ys_true)),
                "WA": float(wa),
                "UA": float(ua),
                "MacroF1": float(mf1),
                "confusion_matrix": cm,
            }
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            import json
            with open(out_dir / "eval_report.json", "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
            print("Wrote evaluation report to", out_dir / "eval_report.json")
            print(out)
        return
    train(args)


if __name__ == "__main__":
    cli()
