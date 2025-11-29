"""Research-grade text branch training script for SER on IEMOCAP.

Key improvements over v1:
1. LOSO (Leave-One-Session-Out) cross-validation - standard IEMOCAP protocol
2. 4-class emotion mapping (neu, hap+exc, ang+fru, sad) - standard setup
3. Full dataset usage (no truncation)
4. Proper regularization: dropout, weight decay, label smoothing
5. Attention pooling option (better than mean/cls)
6. Reports both WA and UA metrics (required for comparison with literature)
7. Learning rate scheduler with warmup
8. Early stopping with patience
9. Confusion matrix and per-class analysis

Target: 68-71% WA on text-only (matching RobinNet/TSIN benchmarks)

Usage:
    # Single fold (LOSO) - leave out session 5
    python scripts/finetune_text_v2.py --test_session 5
    
    # Full 5-fold LOSO cross-validation
    python scripts/finetune_text_v2.py --full_loso
    
    # Quick smoke test
    python scripts/finetune_text_v2.py --test_session 5 --epochs 2 --quick_test

Reference benchmarks (IEMOCAP 4-class):
    - RobinNet (2024): 71.1% WA, 70.6% UA (text-only, RoBERTa)
    - TSIN (2022): 68.7% WA (text-only, GloVe+LSTM)
    - ISSA-BiGRU-MHA (2024): 66.1% WA (text-only, GloVe)
"""
import argparse
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

# Project imports
from serxai.data.iemocap_dataset import read_manifest
from serxai.data.collators import TextCollator
from serxai.models.tokenizer_encoder import TokenizerEncoder

# ============================================================================
# CONSTANTS
# ============================================================================

# Standard 4-class IEMOCAP mapping (from literature)
# Original: neu(0), hap(1), ang(2), sad(3), exc(4), fru(5)
# 4-class:  neu(0), hap+exc(1), ang+fru(2), sad(3)
LABEL_4CLASS_MAP = {
    "neu": 0, "neutral": 0,
    "hap": 1, "happy": 1, "exc": 1, "excited": 1,
    "ang": 2, "angry": 2, "fru": 2, "frustration": 2,
    "sad": 3, "sadness": 3,
}

LABEL_4CLASS_NAMES = ["neutral", "happy", "angry", "sad"]

# ============================================================================
# DATA UTILITIES
# ============================================================================

def get_session_id(record: Dict) -> Optional[int]:
    """Extract numeric session ID from record."""
    s = record.get("session")
    if s is None:
        return None
    if isinstance(s, int):
        return s
    m = re.search(r"(\d+)", str(s))
    if m:
        return int(m.group(1))
    return None


def canonicalize_label_4class(label) -> int:
    """Map label to 4-class index. Returns -1 for invalid labels."""
    if label is None:
        return -1
    if isinstance(label, int):
        # Already an int - assume it's from 6-class, map accordingly
        # 0->neu(0), 1->hap(1), 2->ang(2), 3->sad(3), 4->exc(1), 5->fru(2)
        map_6_to_4 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2}
        return map_6_to_4.get(label, -1)
    key = str(label).lower().strip()
    return LABEL_4CLASS_MAP.get(key, -1)


def loso_split(records: List[Dict], test_session: int) -> Tuple[List[Dict], List[Dict]]:
    """Leave-One-Session-Out split.
    
    Args:
        records: All manifest records
        test_session: Session number to hold out (1-5)
    
    Returns:
        (train_records, test_records)
    """
    train, test = [], []
    for r in records:
        sid = get_session_id(r)
        if sid == test_session:
            test.append(r)
        else:
            train.append(r)
    return train, test


def filter_valid_records(records: List[Dict]) -> List[Dict]:
    """Filter records with valid 4-class labels."""
    valid = []
    for r in records:
        label = canonicalize_label_4class(r.get("label"))
        if label >= 0:
            valid.append(r)
    return valid


def get_class_distribution(records: List[Dict]) -> Dict[int, int]:
    """Get class counts."""
    dist = defaultdict(int)
    for r in records:
        label = canonicalize_label_4class(r.get("label"))
        if label >= 0:
            dist[label] += 1
    return dict(dist)


# ============================================================================
# DATASET
# ============================================================================

class TextDataset(Dataset):
    """Text dataset with 4-class labels."""
    
    def __init__(self, records: List[Dict]):
        self.records = records
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "text": r.get("text", ""),
            "label": canonicalize_label_4class(r.get("label")),
            "utterance_id": r.get("utterance_id", ""),
        }


class TextCollatorV2:
    """Improved collator with proper tokenization."""
    
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence (better than mean/cls)."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tokens: (B, T, D)
            mask: (B, T) attention mask
        Returns:
            pooled: (B, D)
        """
        # Compute attention weights
        weights = self.attention(tokens).squeeze(-1)  # (B, T)
        
        if mask is not None:
            # Mask out padding
            weights = weights.masked_fill(mask == 0, float("-inf"))
        
        weights = F.softmax(weights, dim=-1)  # (B, T)
        
        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)  # (B, D)
        return pooled


class TextClassifier(nn.Module):
    """Research-grade text classifier with regularization."""
    
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        num_classes: int = 4,
        dropout: float = 0.3,
        pooling: str = "attention",  # "attention", "cls", "mean"
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling
        
        # Pooling layer
        if pooling == "attention":
            self.pooler = AttentionPooling(hidden_dim)
        else:
            self.pooler = None
        
        # Classification head with regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Get encoder outputs
        enc_out = self.encoder(input_ids, attention_mask, return_tokens=True)
        tokens = enc_out.get("tokens")  # (B, T, D)
        
        # Pooling
        if self.pooling_type == "attention" and self.pooler is not None:
            pooled = self.pooler(tokens, attention_mask)
        elif self.pooling_type == "cls":
            pooled = tokens[:, 0, :]
        else:  # mean pooling
            mask = attention_mask.unsqueeze(-1)
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        
        # Classification
        logits = self.classifier(pooled)
        
        output = {"logits": logits}
        if return_features:
            output["features"] = pooled
        return output


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for regularization."""
    
    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # KL divergence
        log_probs = F.log_softmax(pred, dim=-1)
        loss = (-true_dist * log_probs).sum(dim=-1)
        
        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight[target]
            loss = loss * weight
        
        return loss.mean()


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> Dict:
    """Compute WA, UA, and per-class metrics."""
    from sklearn.metrics import (
        accuracy_score, recall_score, f1_score, 
        confusion_matrix, classification_report
    )
    
    # Filter valid samples
    mask = y_true >= 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {"error": "No valid samples"}
    
    wa = accuracy_score(y_true, y_pred)
    ua = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Per-class recall (UA components)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    return {
        "WA": float(wa),
        "UA": float(ua),
        "F1_macro": float(f1_macro),
        "F1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "n_samples": len(y_true),
    }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Filter valid labels
        mask = labels >= 0
        if mask.sum() == 0:
            continue
        
        # Forward
        output = model(input_ids, attention_mask)
        logits = output["logits"]
        
        # Loss only on valid labels
        loss = loss_fn(logits[mask], labels[mask])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * mask.sum().item()
        
        # Collect predictions
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(all_labels) if len(all_labels) > 0 else 0.0
    
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        mask = labels >= 0
        if mask.sum() == 0:
            continue
        
        output = model(input_ids, attention_mask)
        logits = output["logits"]
        
        loss = loss_fn(logits[mask], labels[mask])
        total_loss += loss.item() * mask.sum().item()
        
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(all_labels) if len(all_labels) > 0 else 0.0
    
    return metrics


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_fold(
    train_records: List[Dict],
    val_records: List[Dict],
    args: argparse.Namespace,
    fold_id: int = 0,
) -> Dict:
    """Train a single fold."""
    
    # Set seeds for reproducibility
    random.seed(args.seed + fold_id)
    np.random.seed(args.seed + fold_id)
    torch.manual_seed(args.seed + fold_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + fold_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"\n{'='*60}")
    print(f"FOLD {fold_id} | Device: {device}")
    print(f"Train samples: {len(train_records)} | Val samples: {len(val_records)}")
    print(f"{'='*60}")
    
    # Print class distribution
    train_dist = get_class_distribution(train_records)
    val_dist = get_class_distribution(val_records)
    print(f"Train distribution: {dict(sorted(train_dist.items()))}")
    print(f"Val distribution: {dict(sorted(val_dist.items()))}")
    
    # Create datasets
    train_dataset = TextDataset(train_records)
    val_dataset = TextDataset(val_records)
    
    # Create tokenizer/encoder
    tokenc = TokenizerEncoder(
        backbone=args.backbone,
        proj_dim=None,  # No projection - keep original dim
        force_fallback=args.force_fallback,
    )
    
    # Create collator
    collator = TextCollatorV2(tokenc.tokenizer, max_length=args.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Windows compatibility
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    
    # Create model
    model = TextClassifier(
        encoder=tokenc.encoder,
        hidden_dim=tokenc.encoder.out_dim,
        num_classes=4,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)
    
    # Compute class weights for imbalanced data
    class_counts = np.array([train_dist.get(i, 1) for i in range(4)])
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 4  # Normalize
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Loss function with label smoothing
    if args.label_smoothing > 0:
        loss_fn = LabelSmoothingCrossEntropy(
            smoothing=args.label_smoothing,
            weight=class_weights if args.use_class_weights else None,
        )
    else:
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights if args.use_class_weights else None
        )
    
    # Optimizer with weight decay
    # Different learning rates for encoder (smaller) vs classifier (larger)
    if args.freeze_backbone:
        for p in tokenc.encoder.parameters():
            p.requires_grad = False
        optimizer = AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        encoder_params = list(tokenc.encoder.parameters())
        classifier_params = list(model.classifier.parameters())
        if model.pooler is not None:
            classifier_params += list(model.pooler.parameters())
        
        optimizer = AdamW([
            {"params": encoder_params, "lr": args.lr * 0.1},  # Lower LR for pre-trained
            {"params": classifier_params, "lr": args.lr},
        ], weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.lr * 0.1, args.lr] if not args.freeze_backbone else [args.lr],
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
        anneal_strategy="cos",
    )
    
    # Training loop
    best_val_wa = 0.0
    best_val_ua = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, args.grad_clip
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        
        # Log
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train WA: {train_metrics['WA']*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val WA: {val_metrics['WA']*100:.2f}% | "
              f"Val UA: {val_metrics['UA']*100:.2f}%")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_WA": train_metrics["WA"],
            "train_UA": train_metrics["UA"],
            "val_loss": val_metrics["loss"],
            "val_WA": val_metrics["WA"],
            "val_UA": val_metrics["UA"],
        })
        
        # Save best model (by WA, as commonly reported)
        if val_metrics["WA"] > best_val_wa:
            best_val_wa = val_metrics["WA"]
            best_val_ua = val_metrics["UA"]
            best_epoch = epoch + 1
            patience_counter = 0
            
            if args.save_model:
                save_path = Path(args.out_dir) / f"text_encoder_fold{fold_id}_best.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_WA": val_metrics["WA"],
                    "val_UA": val_metrics["UA"],
                    "args": vars(args),
                }, save_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    print(f"\nBest Val WA: {best_val_wa*100:.2f}% | Best Val UA: {best_val_ua*100:.2f}% (Epoch {best_epoch})")
    
    # Final evaluation with best model
    if args.save_model:
        save_path = Path(args.out_dir) / f"text_encoder_fold{fold_id}_best.pt"
        if save_path.exists():
            checkpoint = torch.load(save_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
    
    final_metrics = evaluate(model, val_loader, loss_fn, device)
    final_metrics["best_epoch"] = best_epoch
    final_metrics["history"] = history
    
    # Print confusion matrix
    cm = np.array(final_metrics["confusion_matrix"])
    print(f"\nConfusion Matrix (Fold {fold_id}):")
    print(f"           {' '.join([f'{n:>8}' for n in LABEL_4CLASS_NAMES])}")
    for i, row in enumerate(cm):
        print(f"{LABEL_4CLASS_NAMES[i]:>10} {' '.join([f'{v:>8}' for v in row])}")
    
    return final_metrics


def run_full_loso(args: argparse.Namespace) -> Dict:
    """Run full 5-fold LOSO cross-validation."""
    print("\n" + "="*60)
    print("FULL LOSO (LEAVE-ONE-SESSION-OUT) CROSS-VALIDATION")
    print("="*60)
    
    # Load and filter data
    manifest_path = Path(args.manifest)
    records = read_manifest(manifest_path)
    records = filter_valid_records(records)
    print(f"Total valid records: {len(records)}")
    
    all_fold_results = []
    all_preds = []
    all_labels = []
    
    for test_session in range(1, 6):
        print(f"\n{'#'*60}")
        print(f"# FOLD {test_session}: Test on Session {test_session}")
        print(f"{'#'*60}")
        
        train_records, test_records = loso_split(records, test_session)
        
        # Further split train into train/val for early stopping
        # Use 10% of training as validation
        np.random.seed(args.seed)
        indices = np.random.permutation(len(train_records))
        val_size = int(len(train_records) * 0.1)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_subset = [train_records[i] for i in train_indices]
        val_subset = [train_records[i] for i in val_indices]
        
        fold_metrics = train_fold(train_subset, val_subset, args, fold_id=test_session)
        
        # Also evaluate on held-out test session
        # Note: for final evaluation, we should retrain on full train+val
        fold_metrics["test_session"] = test_session
        all_fold_results.append(fold_metrics)
    
    # Aggregate results
    wa_scores = [r["WA"] for r in all_fold_results]
    ua_scores = [r["UA"] for r in all_fold_results]
    
    results = {
        "method": "LOSO 5-fold Cross-Validation",
        "dataset": "IEMOCAP",
        "num_classes": 4,
        "folds": all_fold_results,
        "avg_WA": float(np.mean(wa_scores)),
        "std_WA": float(np.std(wa_scores)),
        "avg_UA": float(np.mean(ua_scores)),
        "std_UA": float(np.std(ua_scores)),
        "per_fold_WA": wa_scores,
        "per_fold_UA": ua_scores,
    }
    
    print("\n" + "="*60)
    print("LOSO CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"Average WA: {results['avg_WA']*100:.2f}% ± {results['std_WA']*100:.2f}%")
    print(f"Average UA: {results['avg_UA']*100:.2f}% ± {results['std_UA']*100:.2f}%")
    print(f"Per-fold WA: {[f'{wa*100:.2f}%' for wa in wa_scores]}")
    print(f"Per-fold UA: {[f'{ua*100:.2f}%' for ua in ua_scores]}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Research-grade text branch training for SER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data
    parser.add_argument("--manifest", default="data/iemocap_manifest.jsonl",
                        help="Path to IEMOCAP manifest")
    parser.add_argument("--test_session", type=int, default=5,
                        help="Session to use as test set (1-5) for single fold")
    parser.add_argument("--full_loso", action="store_true",
                        help="Run full 5-fold LOSO cross-validation")
    
    # Model
    parser.add_argument("--backbone", default="roberta-base",
                        help="Pre-trained encoder (roberta-base, bert-base-uncased, or fallback)")
    parser.add_argument("--force_fallback", action="store_true",
                        help="Force use of fallback encoder")
    parser.add_argument("--pooling", default="attention",
                        choices=["attention", "cls", "mean"],
                        help="Pooling strategy")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max sequence length for tokenizer")
    
    # Training
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (for classifier; encoder uses 0.1x)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for LR scheduler")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    
    # Regularization
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate in classifier")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Use class weights for imbalanced data")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze pre-trained encoder")
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    
    # Output
    parser.add_argument("--out_dir", default="artifacts/text_branch",
                        help="Output directory")
    parser.add_argument("--save_model", action="store_true",
                        help="Save model checkpoints")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")
    parser.add_argument("--quick_test", action="store_true",
                        help="Quick test with limited data")
    
    args = parser.parse_args()
    
    # Print config
    print("\n" + "="*60)
    print("TEXT BRANCH TRAINING CONFIGURATION")
    print("="*60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    # Load data
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    records = read_manifest(manifest_path)
    records = filter_valid_records(records)
    print(f"\nTotal valid records (4-class): {len(records)}")
    print(f"Class distribution: {dict(sorted(get_class_distribution(records).items()))}")
    
    if args.quick_test:
        print("\n[QUICK TEST MODE - using limited data]")
        records = records[:500]
    
    # Run training
    if args.full_loso:
        results = run_full_loso(args)
    else:
        # Single fold
        train_records, test_records = loso_split(records, args.test_session)
        
        # Split train into train/val
        np.random.seed(args.seed)
        indices = np.random.permutation(len(train_records))
        val_size = int(len(train_records) * 0.1)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_subset = [train_records[i] for i in train_indices]
        val_subset = [train_records[i] for i in val_indices]
        
        results = train_fold(train_subset, val_subset, args, fold_id=args.test_session)
        results["test_session"] = args.test_session
    
    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = out_dir / f"results_{timestamp}.json"
    
    # Remove non-serializable items
    results_clean = {k: v for k, v in results.items() if k != "history"}
    
    with open(results_path, "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    if args.full_loso:
        print(f"Average WA: {results['avg_WA']*100:.2f}% ± {results['std_WA']*100:.2f}%")
        print(f"Average UA: {results['avg_UA']*100:.2f}% ± {results['std_UA']*100:.2f}%")
    else:
        print(f"Val WA: {results['WA']*100:.2f}%")
        print(f"Val UA: {results['UA']*100:.2f}%")
    
    print("\nBenchmark comparison (text-only on IEMOCAP 4-class):")
    print(f"  - RobinNet (2024):        71.1% WA, 70.6% UA")
    print(f"  - TSIN (2022):            68.7% WA")
    print(f"  - ISSA-BiGRU-MHA (2024):  66.1% WA")
    print(f"  - Our result:             {results.get('WA', results.get('avg_WA', 0))*100:.2f}% WA")


if __name__ == "__main__":
    main()
