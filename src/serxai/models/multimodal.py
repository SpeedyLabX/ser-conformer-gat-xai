"""Multimodal SER model wiring pretrained audio/text encoders with GAT fusion."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from serxai.models.classifier import Classifier
from serxai.models.conformer import ConformerConfig, ConformerEncoder
from serxai.models.gat_fusion import GATFusion
from serxai.models.text_encoder import TextEncoder


def _slice_sequence(seq: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return seq
    valid = ~mask
    if valid.dim() == 1:
        valid = valid.unsqueeze(-1)
    return seq[valid.squeeze(-1)] if valid.any() else seq[:0]


class MultimodalSERModel(nn.Module):
    """Combine pretrained encoders, GAT fusion, and classifier head."""

    def __init__(
        self,
        *,
        audio_checkpoint: Path | str,
        text_backbone: Path | str,
        text_proj_dim: int = 128,
        fusion_hidden: int = 256,
        fusion_heads: int = 4,
        fusion_layers: int = 2,
        num_classes: int = 6,
        freeze_audio: bool = True,
        freeze_text: bool = True,
        classifier_hidden: Optional[int] = None,
    ):
        super().__init__()
        audio_path = Path(audio_checkpoint)
        self.audio_encoder = ConformerEncoder.from_pretrained(audio_path)
        if freeze_audio:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False

        backbone_path = Path(text_backbone)
        self.text_encoder = TextEncoder(
            backbone=str(text_backbone),
            proj_dim=text_proj_dim,
            freeze_backbone=freeze_text,
            local_files_only=backbone_path.exists(),
        )
        fusion_in_dim = max(self.audio_encoder.output_dim, text_proj_dim)
        if self.audio_encoder.output_dim != fusion_in_dim:
            self.audio_adapter = nn.Linear(self.audio_encoder.output_dim, fusion_in_dim)
        else:
            self.audio_adapter = nn.Identity()
        if text_proj_dim != fusion_in_dim:
            self.text_adapter = nn.Linear(text_proj_dim, fusion_in_dim)
        else:
            self.text_adapter = nn.Identity()

        self.fusion = GATFusion(
            in_dim=fusion_in_dim,
            hidden_dim=fusion_hidden,
            heads=fusion_heads,
            layers=fusion_layers,
        )
        cls_hidden = classifier_hidden or fusion_hidden
        self.classifier = Classifier(input_dim=fusion_hidden, hidden_dim=cls_hidden, n_classes=num_classes)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        audio_out = self.audio_encoder(
            batch["audio_features"], lengths=batch.get("audio_lengths")
        )
        text_out = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            return_tokens=True,
        )

        audio_sequences = []
        audio_masks = audio_out.get("mask")
        for idx in range(batch["audio_features"].size(0)):
            mask = audio_masks[idx] if audio_masks is not None else None
            seq = audio_out["sequence"][idx]
            if mask is not None:
                seq = seq[~mask]
            audio_sequences.append(self.audio_adapter(seq))

        text_sequences = []
        attention_mask = batch.get("attention_mask")
        tokens = text_out["tokens"]
        for idx in range(tokens.size(0)):
            if attention_mask is None:
                text_seq = tokens[idx]
            else:
                valid = attention_mask[idx].bool()
                text_seq = tokens[idx][valid]
            text_sequences.append(self.text_adapter(text_seq))

        fusion_out = self.fusion(audio_sequences, text_sequences, return_attention=return_attention)
        if return_attention:
            fusion_vec, attn = fusion_out
        else:
            fusion_vec = fusion_out
            attn = None

        logits = self.classifier(fusion_vec)
        return {
            "logits": logits,
            "audio": audio_out,
            "text": text_out,
            "fusion_attention": attn,
        }


__all__ = ["MultimodalSERModel"]
