"""Train CLI that wires the multimodal Conformer–GAT SER model end-to-end."""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from serxai.data.datamodule import DataModule
from serxai.data.labels import LABELS
from serxai.models.multimodal import MultimodalSERModel
from serxai.utils import metrics as metrics_mod
from serxai.utils.seed import set_seed


def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if "extends" in cfg:
        base_path = (path.parent / cfg["extends"]).resolve()
        base_cfg = load_config(base_path)
        cfg = _deep_update(base_cfg, {k: v for k, v in cfg.items() if k != "extends"})
    return cfg


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def build_loss(loss_cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    loss_type = (loss_cfg or {}).get("type", "cross_entropy").lower()
    class_weights = loss_cfg.get("class_weights")
    weight_tensor = None
    if class_weights:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if loss_type == "focal":
        gamma = float(loss_cfg.get("gamma", 2.0))
        return FocalLoss(gamma=gamma, weight=weight_tensor)
    return torch.nn.CrossEntropyLoss(weight=weight_tensor)


def evaluate(
    model: MultimodalSERModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    metric_names: list[str],
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            labels = batch["labels"]
            outputs = model(batch)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            preds_all.append(logits.argmax(dim=-1).cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    metrics = {"loss": total_loss / max(1, total_samples)}
    if preds_all:
        preds_np = np.concatenate(preds_all)
        labels_np = np.concatenate(labels_all)
        for name in metric_names:
            if name == "loss":
                continue
            if hasattr(metrics_mod, name):
                metrics[name] = getattr(metrics_mod, name)(preds_np, labels_np)
            elif name == "cm":
                metrics[name] = confusion_matrix(labels_np, preds_np).tolist()
    return metrics


def serialize_attention(attn_batch):
    serialised = []
    for sample_attn in attn_batch:
        sample_layers = []
        for layer_attn in sample_attn:
            layer_serial = {}
            for dest, info in layer_attn.items():
                weights = info["weights"]
                if torch.is_tensor(weights):
                    weights = weights.cpu().tolist()
                layer_serial[str(dest)] = {
                    "sources": info["sources"],
                    "weights": weights,
                }
            sample_layers.append(layer_serial)
        serialised.append(sample_layers)
    return serialised


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Conformer–GAT multimodal SER model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Run a single forward pass and exit.")
    parser.add_argument("--run-dir", type=str, help="Override output directory.")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 42)))

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    trainer_cfg = cfg.get("trainer", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts")).resolve()

    text_backbone = Path(model_cfg.get("text", {}).get("checkpoint", artifacts_dir / "roberta-text-encoder"))
    tokenizer_kwargs = {}
    if text_backbone.exists():
        tokenizer_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(str(text_backbone), **tokenizer_kwargs)

    manifest_path = data_cfg.get("manifest")
    if manifest_path is None:
        manifest_candidates = [
            Path(data_cfg.get("root", "data")) / "iemocap_manifest.jsonl",
            Path("data") / "iemocap_manifest.jsonl",
        ]
        for cand in manifest_candidates:
            if Path(cand).exists():
                manifest_path = cand
                break
    if manifest_path is None:
        raise FileNotFoundError("Could not locate manifest. Please set data.manifest in the config.")

    datamodule = DataModule(
        manifest_path=str(manifest_path),
        tokenizer=tokenizer,
        dataset_root=data_cfg.get("root"),
        batch_size=int(trainer_cfg.get("batch_size", 16)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        split=data_cfg.get("split"),
        seed=int(cfg.get("seed", 42)),
        session_split=data_cfg.get("session_split"),
        max_text_length=int(data_cfg.get("max_text_len", 128)),
        max_audio_frames=data_cfg.get("max_audio_frames"),
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    audio_ckpt = model_cfg.get("audio", {}).get(
        "checkpoint", artifacts_dir / "audio-encoder" / "conformer_encoder.pkl"
    )
    fusion_cfg = model_cfg.get("fusion", {})
    text_cfg = model_cfg.get("text", {})

    model = MultimodalSERModel(
        audio_checkpoint=audio_ckpt,
        text_backbone=text_backbone,
        text_proj_dim=int(text_cfg.get("proj_dim", 128)),
        fusion_hidden=int(fusion_cfg.get("gat_hidden", 256)),
        fusion_heads=int(fusion_cfg.get("gat_heads", 4)),
        fusion_layers=int(fusion_cfg.get("gat_layers", 2)),
        num_classes=len(LABELS),
        freeze_audio=bool(model_cfg.get("audio", {}).get("freeze", True)),
        freeze_text=bool(text_cfg.get("freeze", True)),
    ).to(device)

    if args.dry_run:
        batch = next(iter(train_loader))
        batch = batch_to_device(batch, device)
        outputs = model(batch, return_attention=True)
        print("Logits:", outputs["logits"].shape)
        if outputs["fusion_attention"] is not None:
            print("Captured fusion attention for", len(outputs["fusion_attention"]), "samples.")
        return

    criterion = build_loss(model_cfg.get("loss", {}), device)
    lr = float(trainer_cfg.get("lr", 3e-4))
    weight_decay = float(trainer_cfg.get("weight_decay", 1e-4))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    grad_clip = trainer_cfg.get("grad_clip")

    metrics_list = [m.lower() for m in cfg.get("metrics", ["wa", "ua", "f1_macro"])]
    monitor_metric = next((m for m in ("wa", "f1_macro", "ua") if m in metrics_list), "loss")
    best_score = float("inf") if monitor_metric == "loss" else float("-inf")
    patience = int(trainer_cfg.get("patience", 7))
    patience_ctr = 0
    epochs = int(trainer_cfg.get("epochs", 50))

    run_root = Path(
        args.run_dir
        or cfg.get("log_dir", "experiments/runs")
        or "experiments/runs"
    )
    run_dir = (run_root / time.strftime("%Y%m%d-%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        preds_all = []
        labels_all = []
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            labels = batch["labels"]
            outputs = model(batch)
            logits = outputs["logits"]
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            preds_all.append(logits.argmax(dim=-1).detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())

        train_metrics = {
            "loss": total_loss / max(1, total_samples),
        }
        if preds_all:
            preds_np = np.concatenate(preds_all)
            labels_np = np.concatenate(labels_all)
            for name in metrics_list:
                if name == "loss":
                    continue
                if hasattr(metrics_mod, name):
                    train_metrics[name] = getattr(metrics_mod, name)(preds_np, labels_np)
                elif name == "cm":
                    train_metrics[name] = confusion_matrix(labels_np, preds_np).tolist()

        val_metrics = evaluate(model, val_loader, criterion, device, metrics_list)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if monitor_metric == "loss":
            current_value = val_metrics["loss"]
            improved = current_value < best_score
        else:
            current_value = val_metrics.get(monitor_metric, float("-inf"))
            improved = current_value > best_score

        if improved:
            best_score = current_value
            patience_ctr = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            with (run_dir / "best_metrics.json").open("w") as fh:
                json.dump(val_metrics, fh, indent=2)
        else:
            patience_ctr += 1

        print(
            f"[Epoch {epoch}] train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} monitor={monitor_metric}:{current_value:.4f} "
            f"patience={patience_ctr}/{patience}"
        )
        if patience_ctr >= patience:
            print("Early stopping triggered.")
            break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Export a sample of fusion attention weights for XAI inspection
    try:
        first_batch = next(iter(val_loader))
    except StopIteration:
        first_batch = None
    if first_batch:
        best_ckpt = run_dir / "best_model.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
        model.eval()
        batch_dev = batch_to_device(first_batch, device)
        with torch.no_grad():
            outputs = model(batch_dev, return_attention=True)
        if outputs["fusion_attention"]:
            serial = serialize_attention(outputs["fusion_attention"])
            with (run_dir / "fusion_attention.json").open("w") as fh:
                json.dump(serial, fh, indent=2)


if __name__ == "__main__":
    main()
