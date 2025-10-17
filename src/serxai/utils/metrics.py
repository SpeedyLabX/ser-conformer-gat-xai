"""Metrics: WA, UA, macro-F1, confusion matrix"""
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np


def wa(preds, targets):
    """Weighted accuracy (simple accuracy)."""
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return float((preds == targets).mean())


def ua(preds, targets):
    """Unweighted average recall (mean recall across classes)."""
    cm = confusion_matrix(targets, preds)
    recalls = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
    return float(np.nanmean(recalls))


def f1_macro(preds, targets):
    return float(f1_score(targets, preds, average="macro"))
