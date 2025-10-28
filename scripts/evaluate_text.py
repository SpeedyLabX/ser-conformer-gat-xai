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
from serxai.data.labels import LABELS, canonicalize_label, map_to_4class_idx, id_to_label


DISPLAY_LABELS_6 = {
    "neu": "Neutral",
    "hap": "Happy",
    "ang": "Anger",
    "sad": "Sad",
    "exc": "Excited",
    "fru": "Frustration",
}

FOUR_CLASS_LABELS = ["Neutral", "Happy/Excited", "Anger/Frustration", "Sad"]


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
        for i in range(0, len(val_records), batch_size):
            batch = val_records[i : i + batch_size]
            logits = predict_batch(batch)
            preds = logits.argmax(axis=-1)
            labs = [canonicalize_label(r.get("label")) for r in batch]
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

    ys_true = np.array(ys_true, dtype=int)
    ys_pred = np.array(ys_pred, dtype=int)
    mask = ys_true >= 0
    ys_true = ys_true[mask]
    ys_pred = ys_pred[mask]

    if len(ys_true) == 0:
        raise RuntimeError("No labeled samples found in validation split")

    def _label_name_from_idx(idx: int) -> str:
        canonical = id_to_label(idx)
        return DISPLAY_LABELS_6.get(canonical, canonical)

    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_indices, label_names):
        metrics = {
            "WA": float(accuracy_score(y_true, y_pred)),
            "UA": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "UF1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "WF1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        per_class = {}
        for idx, name in zip(label_indices, label_names):
            per_class[name] = {
                "precision": float(precision_score(y_true, y_pred, labels=[idx], average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, labels=[idx], average="macro", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, labels=[idx], average="macro", zero_division=0)),
            }
        return metrics, per_class

    label_indices_6 = list(range(len(LABELS)))
    label_names_6 = [_label_name_from_idx(i) for i in label_indices_6]
    cm_6 = confusion_matrix(ys_true, ys_pred, labels=label_indices_6)
    metrics_6, per_class_6 = _compute_metrics(ys_true, ys_pred, label_indices_6, label_names_6)

    if map_to_4class:
        ys_true_4 = np.array([map_to_4class_idx(int(x)) for x in ys_true], dtype=int)
        ys_pred_4 = np.array([map_to_4class_idx(int(x)) for x in ys_pred], dtype=int)
        mask2 = (ys_true_4 >= 0) & (ys_pred_4 >= 0)
        ys_true_4 = ys_true_4[mask2]
        ys_pred_4 = ys_pred_4[mask2]

        if len(ys_true_4) == 0:
            raise RuntimeError("No labeled samples remain after 4-class mapping")

        label_indices_4 = list(range(len(FOUR_CLASS_LABELS)))
        label_names_4 = FOUR_CLASS_LABELS
        cm_4 = confusion_matrix(ys_true_4, ys_pred_4, labels=label_indices_4)
        metrics_4, per_class_4 = _compute_metrics(ys_true_4, ys_pred_4, label_indices_4, label_names_4)

        out = {
            "n_samples": int(len(ys_true_4)),
            **metrics_4,
            "label_names": label_names_4,
            "metrics": metrics_4,
            "per_class": per_class_4,
            "confusion_matrix": cm_4.tolist(),
            "six_class": {
                "n_samples": int(len(ys_true)),
                **metrics_6,
                "label_names": label_names_6,
                "metrics": metrics_6,
                "per_class": per_class_6,
                "confusion_matrix": cm_6.tolist(),
            },
        }
    else:
        out = {
            "n_samples": int(len(ys_true)),
            **metrics_6,
            "label_names": label_names_6,
            "metrics": metrics_6,
            "per_class": per_class_6,
            "confusion_matrix": cm_6.tolist(),
        }

    out_dir = Path(out_dir_p)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_p = out_dir / "eval_report.json"
    with open(report_p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("Wrote evaluation report to", report_p)
    print(json.dumps(out, indent=2))
    # optionally save confusion matrix plots
    if save_confusion_plot:
        try:
            figs_dir = out_dir / 'figures'
            figs_dir.mkdir(parents=True, exist_ok=True)
            import matplotlib

            matplotlib.rcParams.update({
                'font.size': 8,
                'axes.titlesize': 8,
                'axes.labelsize': 8,
            })

            def _save_confusion_artifacts(report: dict, suffix: str):
                labels = report['label_names']
                cm_arr = np.array(report['confusion_matrix'])
                per_cls = report['per_class']
                suffix_str = f"_{suffix}" if suffix else ""

                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(cm_arr, interpolation='nearest', cmap='Blues')
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_yticklabels(labels)
                fmt = 'd'
                thresh = cm_arr.max() / 2.0 if cm_arr.size else 0
                for i in range(cm_arr.shape[0]):
                    for j in range(cm_arr.shape[1]):
                        color = 'white' if cm_arr[i, j] > thresh else 'black'
                        ax.text(j, i, format(cm_arr[i, j], fmt), ha='center', va='center', color=color, fontsize=7)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
                ax.set_xlabel('')
                ax.set_ylabel('')
                fig.tight_layout()
                png_p = figs_dir / f'confusion_matrix{suffix_str}.png'
                pdf_p = figs_dir / f'confusion_matrix{suffix_str}.pdf'
                fig.savefig(png_p, dpi=300, bbox_inches='tight')
                fig.savefig(pdf_p, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print('Saved confusion matrix plots to', png_p, pdf_p)

                fig2, ax2 = plt.subplots(figsize=(6, 3))
                classes = list(per_cls.keys())
                f1s = [per_cls[c]['f1'] for c in classes]
                y_pos = range(len(classes))
                ax2.barh(y_pos, f1s, color='C0')
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(classes)
                ax2.set_xlim(0, 1)
                ax2.set_xlabel('')
                ax2.set_ylabel('')
                for i, v in enumerate(f1s):
                    ax2.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=7)
                fig2.tight_layout()
                png2 = figs_dir / f'per_class_f1{suffix_str}.png'
                pdf2 = figs_dir / f'per_class_f1{suffix_str}.pdf'
                fig2.savefig(png2, dpi=300, bbox_inches='tight')
                fig2.savefig(pdf2, dpi=300, bbox_inches='tight')
                plt.close(fig2)
                print('Saved per-class F1 plots to', png2, pdf2)

            if map_to_4class:
                _save_confusion_artifacts(out, '4class')
                _save_confusion_artifacts(out['six_class'], '6class')
            else:
                _save_confusion_artifacts(out, '')
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
