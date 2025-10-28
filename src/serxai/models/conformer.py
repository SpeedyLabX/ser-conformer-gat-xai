"""Conformer encoder implementation compatible with pretrained checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def _make_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Return BoolTensor mask where True denotes padded positions."""
    if lengths is None:
        return None
    max_len = int(max_len or lengths.max().item())
    range_row = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return range_row >= lengths.unsqueeze(1)


class FeedForwardModule(nn.Module):
    """Position-wise feed-forward module used inside Conformer blocks."""

    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion_factor
        self.layers = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvModule(nn.Module):
    """Depthwise convolutional module used inside Conformer blocks."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x


class ConformerBlock(nn.Module):
    """Single Conformer block with macaron-style FFNs."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn_scale = 0.5
        self.ffn1 = FeedForwardModule(d_model, expansion_factor=ff_expansion, dropout=dropout).layers
        self.norm_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConvModule(d_model, kernel_size=conv_kernel, dropout=dropout).layers
        self.ffn2 = FeedForwardModule(d_model, expansion_factor=ff_expansion, dropout=dropout).layers
        self.norm_final = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # macaron FFN 1
        x = x + self.ffn_scale * self.ffn1(x)

        # MHSA
        residual = x
        attn_in = self.norm_attn(x)
        attn_out, _ = self.self_attn(
            attn_in, attn_in, attn_in, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = residual + self.dropout_attn(attn_out)

        # Convolution module
        residual = x
        conv_in = self.norm_conv(x)
        conv_feat = conv_in.transpose(1, 2)
        conv_feat = self.conv(conv_feat)
        conv_out = conv_feat.transpose(1, 2)
        x = residual + conv_out

        # macaron FFN 2
        x = x + self.ffn_scale * self.ffn2(x)
        return self.norm_final(x)


@dataclass
class ConformerConfig:
    input_dim: int = 80
    d_model: int = 256
    num_layers: int = 4
    n_heads: int = 4
    ff_expansion: int = 4
    conv_kernel: int = 31
    dropout: float = 0.1
    output_dim: int = 128


class ConformerEncoder(nn.Module):
    """Conformer encoder compatible with pretrained weights stored under `artifacts/`."""

    def __init__(self, cfg: ConformerConfig):
        super().__init__()
        self.config = cfg
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ff_expansion=cfg.ff_expansion,
                    conv_kernel=cfg.conv_kernel,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.output_dim),
            nn.LayerNorm(cfg.output_dim),
        )

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    def forward(
        self, features: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> dict:
        """Encode log-mel features.

        Args:
            features: Tensor shaped (B, T, F) with F = cfg.input_dim.
            lengths: Optional Tensor of shape (B,) containing valid frame counts.

        Returns:
            dict with keys:
              - "sequence": contextualised embeddings (B, T, output_dim)
              - "mask": BoolTensor padding mask (B, T) where True == padded
              - "pooled": length-aware mean pooling (B, output_dim)
        """
        mask = _make_padding_mask(lengths, max_len=features.size(1)) if lengths is not None else None
        x = self.input_proj(features)
        for block in self.conformer_blocks:
            x = block(x, key_padding_mask=mask)
        seq = self.output_proj(x)

        if mask is None:
            pooled = seq.mean(dim=1)
        else:
            inv_mask = (~mask).float()
            pooled = (seq * inv_mask.unsqueeze(-1)).sum(dim=1) / inv_mask.sum(dim=1, keepdim=True).clamp(
                min=1.0
            )

        return {"sequence": seq, "mask": mask, "pooled": pooled}

    @classmethod
    def from_pretrained(cls, checkpoint_path: Path | str) -> "ConformerEncoder":
        """Instantiate encoder from a serialized checkpoint (see artifacts/audio-encoder)."""
        path = Path(checkpoint_path)
        with path.open("rb") as fh:
            payload = torch.load(fh, map_location="cpu", weights_only=False) if path.suffix in {
                ".pt",
                ".pth",
                ".bin",
            } else None
        if payload is None:
            # fallback: pickle containing {state_dict, config}
            import pickle

            with path.open("rb") as fh:
                payload = pickle.load(fh)

        if "config" in payload:
            cfg_dict = payload["config"]
            cfg = ConformerConfig(
                input_dim=cfg_dict.get("input_dim", 80),
                d_model=cfg_dict.get("d_model", 256),
                num_layers=cfg_dict.get("num_layers", 4),
                n_heads=cfg_dict.get("n_heads", 4),
                ff_expansion=cfg_dict.get("ff_expansion", 4),
                conv_kernel=cfg_dict.get("conv_kernel", 31),
                dropout=cfg_dict.get("dropout", 0.1),
                output_dim=cfg_dict.get("output_dim", 128),
            )
            state_dict = payload["state_dict"]
        else:
            raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

        model = cls(cfg)
        model.load_state_dict(state_dict, strict=True)
        return model


__all__ = ["ConformerConfig", "ConformerEncoder"]
