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
from serxai.data.labels import LABELS, canonicalize_label, map_to_4class_idx


def evaluate(
    manifest_p: Path,
    ckpt_p: Path,
    proj_dim: int = 256,
    backbone: str = "fallback",
    force_fallback: bool = True,
    max_val_samples: int = 0,
    out_dir_p: Path = Path("artifacts"),
    save_confusion_plot: bool = False,
    map_to_4class: bool = False,
):
    manifest = read_manifest(manifest_p)
    train_records, val_records = train_val_split(manifest, val_ratio=0.1)
    # allow limiting validation samples for quick checks
    if max_val_samples and max_val_samples > 0:
        val_records = val_records[:max_val_samples]
    tokenc = TokenizerEncoder(backbone=backbone, proj_dim=proj_dim, force_fallback=force_fallback)
    coll = TextCollator(tokenc.tokenizer, max_length=128)
    # If ckpt_p is a directory containing a HuggingFace model (config + model.safetensors / pytorch_model.bin)
    # load the HF model + tokenizer and run evaluation directly. Otherwise, fall back to the old checkpoint dict format.
    ckpt_path = Path(ckpt_p)
    use_hf_model = ckpt_path.is_dir()

    if use_hf_model:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except Exception as e:
            raise RuntimeError("transformers required to evaluate HF model directory: " + str(e))

        # Try local tokenizer first; if missing, fall back to the provided backbone name (may download)
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), local_files_only=True)
        except Exception:
            hf_tokenizer = AutoTokenizer.from_pretrained(backbone)
        hf_model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path), local_files_only=True)
        hf_model.eval()
        # We'll not use TokenizerEncoder path below; use HF tokenizer+model to compute logits
        def predict_batch(records_batch):
            texts = [r.get('text', '') for r in records_batch]
            enc = hf_tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            with torch.no_grad():
                out = hf_model(**{k: v for k, v in enc.items()})
                logits = out.logits.cpu().numpy()
            return logits
        head = None
    else:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        head = instantiate_from_state_dict(ckpt.get("head", {}), in_dim=tokenc.encoder.out_dim)
        head.load_state_dict(ckpt.get("head", {}))
        head.eval()
        predict_batch = None

    ys_true = []
    ys_pred = []

    # Evaluate either using HF model directory or the head + TokenizerEncoder path
    batch_size = 32
    if use_hf_model:
        # chunk val_records and predict via HF model
        for i in range(0, len(val_records), batch_size):
            batch = val_records[i : i + batch_size]
            logits = predict_batch(batch)
            preds = logits.argmax(axis=-1)
            labs = [ (tokenc.tokenizer(r.get('text','')) and canonicalize_label(r.get('label'))) if False else None for r in batch ]
            # We need true labels as integer indices; use canonicalize_label from data.labels
            from serxai.data.labels import canonicalize_label as _canonicalize
            labs = [_canonicalize(r.get('label')) for r in batch]
            ys_true.extend(labs)
            ys_pred.extend(preds.tolist())
    else:
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
    uf1 = float(f1_score(ys_true, ys_pred, average="macro", zero_division=0))
    wf1 = float(f1_score(ys_true, ys_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(ys_true, ys_pred)

    # Optionally map to 4-class scheme for inference/analysis
    if map_to_4class:
        # map predictions and truths
        ys_true = np.array([map_to_4class_idx(int(x)) for x in ys_true])
        ys_pred = np.array([map_to_4class_idx(int(x)) for x in ys_pred])
        # filter unknowns again
        mask2 = (ys_true >= 0) & (ys_pred >= 0)
        ys_true = ys_true[mask2]
        ys_pred = ys_pred[mask2]

    n_classes = int(max(ys_true.max(), ys_pred.max()) + 1)
    if map_to_4class:
        label_names = ["neu", "hap", "ang", "sad"]
    else:
        label_names = LABELS[:n_classes]
    per_class = {}
    for c in range(n_classes):
        cls_prec = float(precision_score(ys_true, ys_pred, labels=[c], average="macro", zero_division=0))
        cls_rec = float(recall_score(ys_true, ys_pred, labels=[c], average="macro", zero_division=0))
        cls_f1 = float(f1_score(ys_true, ys_pred, labels=[c], average="macro", zero_division=0))
        name = label_names[c] if c < len(label_names) else str(c)
        per_class[name] = {"precision": cls_prec, "recall": cls_rec, "f1": cls_f1}

    out = {
        "n_samples": int(len(ys_true)),
        "WA": wa,
        "UA": ua,
        "UF1": uf1,
        "WF1": wf1,
        "per_class": per_class,
        "label_names": label_names,
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
            # Prepare figure folder
            figs_dir = out_dir / 'figures'
            figs_dir.mkdir(parents=True, exist_ok=True)
            # confusion matrix (paper-ready): use label names on ticks, no title/axis labels
            # Modern paper-ready confusion matrix with colorbar on the right
            import matplotlib
            matplotlib.rcParams.update({
                'font.size': 8,
                'axes.titlesize': 8,
                'axes.labelsize': 8,
            })
            labels = label_names if 'label_names' in locals() else [str(i) for i in range(cm.shape[0])]
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            # ticks with label names
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)
            # show counts
            fmt = 'd'
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    color = 'white' if cm[i, j] > thresh else 'black'
                    ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color=color, fontsize=7)
            # colorbar on the right showing range
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
            # remove axis titles for paper-ready
            ax.set_xlabel('')
            ax.set_ylabel('')
            fig.tight_layout()
            png_p = figs_dir / 'confusion_matrix.png'
            pdf_p = figs_dir / 'confusion_matrix.pdf'
            fig.savefig(png_p, dpi=300, bbox_inches='tight')
            fig.savefig(pdf_p, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print('Saved confusion matrix plots to', png_p, pdf_p)
            # Also save a per-class F1 bar chart (paper-ready)
            # Horizontal per-class F1 bar chart (clean, paper-ready)
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            classes = list(per_class.keys())
            f1s = [per_class[c]['f1'] for c in classes]
            y_pos = range(len(classes))
            ax2.barh(y_pos, f1s, color='C0')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(classes)
            ax2.set_xlim(0, 1)
            # remove axis titles
            ax2.set_xlabel('')
            ax2.set_ylabel('')
            # show f1 value at end of bar
            for i, v in enumerate(f1s):
                ax2.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=7)
            fig2.tight_layout()
            png2 = figs_dir / 'per_class_f1.png'
            pdf2 = figs_dir / 'per_class_f1.pdf'
            fig2.savefig(png2, dpi=300, bbox_inches='tight')
            fig2.savefig(pdf2, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print('Saved per-class F1 plots to', png2, pdf2)
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
    p.add_argument("--map_to_4class", action="store_true", help='Map 6-class predictions/labels to 4-class scheme for metrics')
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
        map_to_4class=args.map_to_4class,
    )


if __name__ == "__main__":
    cli()
