"""Evaluate a saved head on the validation split and write metrics to artifacts/eval_report.json.

This script intentionally avoids importing from scripts (uses package modules) and is lightweight.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix

from serxai.data.iemocap_dataset import read_manifest, train_val_split
from serxai.models.tokenizer_encoder import TokenizerEncoder
from serxai.data.collators import TextCollator
from serxai.models.text_head import instantiate_from_state_dict


def evaluate(manifest_p: Path, ckpt_p: Path, proj_dim: int = 256, backbone: str = "fallback", force_fallback: bool = True):
    manifest = read_manifest(manifest_p)
    train_records, val_records = train_val_split(manifest, val_ratio=0.1)
    tokenc = TokenizerEncoder(backbone=backbone, proj_dim=proj_dim, force_fallback=force_fallback)
    coll = TextCollator(tokenc.tokenizer, max_length=128)

    ckpt = torch.load(str(ckpt_p), map_location="cpu")
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

    ys_true = np.array(ys_true)
    ys_pred = np.array(ys_pred)
    mask = ys_true >= 0
    ys_true = ys_true[mask]
    ys_pred = ys_pred[mask]

    if len(ys_true) == 0:
        raise RuntimeError("No labeled samples found in validation split")

    wa = float(accuracy_score(ys_true, ys_pred))
    ua = float(recall_score(ys_true, ys_pred, average="macro", zero_division=0))
    mf1 = float(f1_score(ys_true, ys_pred, average="macro", zero_division=0))
    cm = confusion_matrix(ys_true, ys_pred).tolist()

    out = {
        "n_samples": int(len(ys_true)),
        "WA": wa,
        "UA": ua,
        "MacroF1": mf1,
        "confusion_matrix": cm,
    }

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_p = out_dir / "eval_report.json"
    with open(report_p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("Wrote evaluation report to", report_p)
    print(json.dumps(out, indent=2))


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/iemocap_manifest.jsonl")
    p.add_argument("--checkpoint", default="artifacts/text_head.pth")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--backbone", default="fallback")
    p.add_argument("--force_fallback", action="store_true", default=True)
    args = p.parse_args()
    evaluate(Path(args.manifest), Path(args.checkpoint), proj_dim=args.proj_dim, backbone=args.backbone, force_fallback=args.force_fallback)


if __name__ == "__main__":
    cli()
