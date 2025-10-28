"""Text encoder wrapper with optional HuggingFace backbone.

Provides:
 - token embeddings (B, T, D)
 - pooled vector (B, D_out) using CLS or mean pooling

Contract (inputs/outputs):
 - forward(input_ids, attention_mask) -> dict{'tokens': Tensor, 'pooled': Tensor}

This implementation will try to import HuggingFace `transformers` and load
an AutoModel; if not available it falls back to a lightweight nn.Embedding + TransformerEncoder placeholder.
"""

from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self,
                 backbone: str = "roberta-base",
                 proj_dim: Optional[int] = 512,
                 pooling: str = "mean",  # 'mean' or 'cls'
                 freeze_backbone: bool = False,
                 force_fallback: bool = False,
                 local_files_only: Optional[bool] = None):
        super().__init__()
        self.backbone_name = backbone
        self.pooling = pooling
        self.proj_dim = proj_dim
        self.freeze_backbone = freeze_backbone

        # Attempt to load HF AutoModel unless force_fallback is set or backbone explicitly
        # requests the fallback.
        self.force_fallback = force_fallback or (backbone == "fallback")
        if not self.force_fallback:
            try:
                from transformers import AutoModel

                self.use_hf = True
                kwargs = {}
                if local_files_only is None:
                    if Path(backbone).exists():
                        kwargs["local_files_only"] = True
                else:
                    kwargs["local_files_only"] = local_files_only
                self.hf = AutoModel.from_pretrained(backbone, **kwargs)
                hidden_size = getattr(self.hf.config, "hidden_size", 768)
                if freeze_backbone:
                    for p in self.hf.parameters():
                        p.requires_grad = False
            except Exception:
                # if HF failed for any reason, fall back
                self.use_hf = False
                hidden_size = 768
                self.embed = nn.Embedding(30522, hidden_size)
                encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            # explicit fallback requested
            self.use_hf = False
            hidden_size = 768
            self.embed = nn.Embedding(30522, hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # projection layer (optional)
        if proj_dim is not None:
            self.proj = nn.Linear(hidden_size, proj_dim)
            out_dim = proj_dim
        else:
            self.proj = nn.Identity()
            out_dim = hidden_size

        self.out_dim = out_dim

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.Tensor = None, return_tokens: bool = False):
        """Forward pass.

        Args:
            input_ids: (B, T) int tensor
            attention_mask: (B, T) int tensor
            return_tokens: whether to include token-level embeddings in the output

        Returns:
            dict with keys: 'pooled' (B, out_dim) and optional 'tokens' (B, T, out_dim)
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        if self.use_hf:
            outputs = self.hf(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            # token embeddings
            tokens = outputs.last_hidden_state  # (B, T, H)
        else:
            x = self.embed(input_ids)  # (B, T, H)
            # Transformer expects (T, B, H)
            x = x.transpose(0, 1)
            x = self.encoder(x)
            tokens = x.transpose(0, 1)

        # pooling
        if self.pooling == "cls":
            pooled = tokens[:, 0, :]
        else:  # mean pooling with mask support
            if attention_mask is None:
                pooled = tokens.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1)
                pooled = (tokens * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-8))

        tokens = self.proj(tokens)
        pooled = self.proj(pooled)

        out = {"pooled": pooled}
        if return_tokens:
            out["tokens"] = tokens
        return out


if __name__ == "__main__":
    # quick smoke test
    enc = TextEncoder(backbone="roberta-base", proj_dim=256)
    import torch
    ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones_like(ids)
    o = enc(ids, mask, return_tokens=True)
    print('pooled', o['pooled'].shape, 'tokens', o['tokens'].shape)
