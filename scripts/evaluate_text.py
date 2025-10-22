"""Evaluate a saved head on the validation split and write metrics to artifacts/eval_report.json.

This script intentionally avoids importing from scripts (uses package modules) and is lightweight.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from serxai.data.iemocap_dataset import read_manifest, train_val_split
from serxai.models.tokenizer_encoder import TokenizerEncoder
from serxai.data.collators import TextCollator
from serxai.models.text_head import instantiate_from_state_dict


def evaluate(
    manifest_p: Path,
    ckpt_p: Path,
    proj_dim: int = 256,
    backbone: str = "fallback",
    force_fallback: bool = True,
    max_val_samples: int = 0,
    out_dir_p: Path = Path("artifacts"),
    save_confusion_plot: bool = False,
):
    manifest = read_manifest(manifest_p)
    train_records, val_records = train_val_split(manifest, val_ratio=0.1)
    # allow limiting validation samples for quick checks
    if max_val_samples and max_val_samples > 0:
        val_records = val_records[:max_val_samples]
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
    cm = confusion_matrix(ys_true, ys_pred)

    # per-class metrics
    n_classes = cm.shape[0]
    per_class = {}
    for c in range(n_classes):
        cls_prec = float(precision_score(ys_true, ys_pred, labels=[c], average="macro", zero_division=0))
        cls_rec = float(recall_score(ys_true, ys_pred, labels=[c], average="macro", zero_division=0))
        cls_f1 = float(f1_score(ys_true, ys_pred, labels=[c], average="macro", zero_division=0))
        per_class[str(c)] = {"precision": cls_prec, "recall": cls_rec, "f1": cls_f1}

    out = {
        "n_samples": int(len(ys_true)),
        "WA": wa,
        "UA": ua,
        "MacroF1": mf1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    out_dir = Path(out_dir_p)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_p = out_dir / "eval_report.json"
    with open(report_p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("Wrote evaluation report to", report_p)
    print(json.dumps(out, indent=2))
    # optionally save a confusion matrix plot
    if save_confusion_plot:
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')
            ax.set_title('Confusion Matrix')
            # show counts
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
            fig.tight_layout()
            plot_p = out_dir / 'confusion_matrix.png'
            fig.savefig(plot_p)
            plt.close(fig)
            print('Saved confusion matrix plot to', plot_p)
        except Exception as e:
            print('Failed to save confusion matrix plot:', e)


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/iemocap_manifest.jsonl")
    p.add_argument("--checkpoint", default="artifacts/text_head.pth")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--backbone", default="fallback")
    p.add_argument("--force_fallback", action="store_true", default=True)
    p.add_argument("--max_val_samples", type=int, default=0, help='Limit validation samples (0 = use all)')
    p.add_argument("--out_dir", default="artifacts", help='Output directory for report and optional plot')
    p.add_argument("--save_confusion_plot", action="store_true", help='Save a confusion_matrix.png into out_dir')
    args = p.parse_args()
    evaluate(
        Path(args.manifest),
        Path(args.checkpoint),
        proj_dim=args.proj_dim,
        backbone=args.backbone,
        force_fallback=args.force_fallback,
        max_val_samples=args.max_val_samples,
        out_dir_p=Path(args.out_dir),
        save_confusion_plot=args.save_confusion_plot,
    )


if __name__ == "__main__":
    cli()
