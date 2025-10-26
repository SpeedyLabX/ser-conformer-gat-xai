# =========================
# Figure Utilities (Python)
# =========================
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, auc
)

# Global style (minimal, academic)
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,   # editable fonts in PDF
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _savefig(base_dir: str, name: str):
    _ensure_dir(base_dir)
    png_path = os.path.join(base_dir, f"{name}.png")
    pdf_path = os.path.join(base_dir, f"{name}.pdf")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

def plot_confusion(y_true, y_pred, class_names, out_dir, normalize="true", name="confusion_matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize=normalize)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Greys", colorbar=False, values_format=".2f")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # No title
    _savefig(out_dir, name)

def plot_pr_curves(y_true_bin, y_score, class_names, out_dir, name="pr_curves"):
    # y_true_bin: (N, C) one-hot; y_score: (N, C) probabilities
    fig, ax = plt.subplots(figsize=(5, 4))
    for c, cname in enumerate(class_names):
        p, r, _ = precision_recall_curve(y_true_bin[:, c], y_score[:, c])
        ap = auc(r, p)
        ax.plot(r, p, label=f"{cname} (AP={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False, ncol=2)
    _savefig(out_dir, name)

def plot_roc_curves(y_true_bin, y_score, class_names, out_dir, name="roc_curves"):
    fig, ax = plt.subplots(figsize=(5, 4))
    for c, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_score[:, c])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False, ncol=2)
    _savefig(out_dir, name)

def plot_learning_curves(train_losses, val_losses, out_dir, name="learning_curves"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    _savefig(out_dir, name)

# Usage example:
# plot_confusion(y_true, y_pred, CLASS_NAMES_6, out_dir, normalize="true", name="cm_6cls")
# plot_pr_curves(y_true_bin, y_prob, CLASS_NAMES_6, out_dir, name="pr_6cls")
# plot_roc_curves(y_true_bin, y_prob, CLASS_NAMES_6, out_dir, name="roc_6cls")
# plot_learning_curves(train_losses, val_losses, out_dir, name="lc_6cls")

# IMPORTANT
# - Use these exact utilities everywhere to keep figures consistent.
# - Export both PNG and PDF.
# - Do NOT add titles or emojis; keep English labels only.
# - Keep style minimal and academic.
