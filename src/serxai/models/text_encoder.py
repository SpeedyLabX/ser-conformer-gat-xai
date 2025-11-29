"""Text encoder wrapper with optional HuggingFace backbone.

Provides:
 - token embeddings (B, T, D)
 - pooled vector (B, D_out) using CLS, mean, or attention pooling

Contract (inputs/outputs):
 - forward(input_ids, attention_mask) -> dict{'tokens': Tensor, 'pooled': Tensor}

This implementation will try to import HuggingFace `transformers` and load
an AutoModel; if not available it falls back to a lightweight nn.Embedding + TransformerEncoder placeholder.

Improvements (v2):
 - Added attention pooling option (better than mean/cls for SER)
 - Added dropout for regularization
 - Support for layer-wise learning rate decay
 - Better handling of fallback encoder
"""

from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence.
    
    Better than simple mean/cls pooling for emotion recognition
    as it learns to attend to emotionally salient tokens.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )
    
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tokens: (B, T, D) token embeddings
            mask: (B, T) attention mask (1 for valid, 0 for padding)
        Returns:
            pooled: (B, D) pooled representation
        """
        weights = self.attention(tokens).squeeze(-1)  # (B, T)
        
        if mask is not None:
            weights = weights.masked_fill(mask == 0, float("-inf"))
        
        weights = F.softmax(weights, dim=-1)  # (B, T)
        pooled = torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)  # (B, D)
        return pooled


class TextEncoder(nn.Module):
    def __init__(self,
                 backbone: str = "roberta-base",
                 proj_dim: Optional[int] = 512,
                 pooling: str = "mean",  # 'mean', 'cls', or 'attention'
                 freeze_backbone: bool = False,
                 force_fallback: bool = False,
                 local_files_only: Optional[bool] = None,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone_name = backbone
        self.pooling = pooling
        self.proj_dim = proj_dim
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout

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
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size, nhead=8, dropout=dropout, batch_first=False
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
                self.embed_dropout = nn.Dropout(dropout)
        else:
            # explicit fallback requested
            self.use_hf = False
            hidden_size = 768
            self.embed = nn.Embedding(30522, hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=8, dropout=dropout, batch_first=False
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.embed_dropout = nn.Dropout(dropout)
        
        # Attention pooling layer (if using attention pooling)
        self._hidden_size = hidden_size
        if pooling == "attention":
            self.attn_pool = AttentionPooling(hidden_size, dropout=dropout)
        else:
            self.attn_pool = None

        # projection layer (optional)
        if proj_dim is not None:
            self.proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, proj_dim),
            )
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
            x = self.embed_dropout(x)
            # Transformer expects (T, B, H)
            x = x.transpose(0, 1)
            x = self.encoder(x)
            tokens = x.transpose(0, 1)

        # pooling
        if self.pooling == "attention" and self.attn_pool is not None:
            pooled = self.attn_pool(tokens, attention_mask)
        elif self.pooling == "cls":
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
    
    def get_encoder_params(self):
        """Get encoder parameters for separate learning rate."""
        if self.use_hf:
            return list(self.hf.parameters())
        else:
            return list(self.embed.parameters()) + list(self.encoder.parameters())
    
    def get_head_params(self):
        """Get head (projection + pooling) parameters."""
        params = list(self.proj.parameters())
        if self.attn_pool is not None:
            params += list(self.attn_pool.parameters())
        return params


if __name__ == "__main__":
    # quick smoke test
    enc = TextEncoder(backbone="roberta-base", proj_dim=256)
    import torch
    ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones_like(ids)
    o = enc(ids, mask, return_tokens=True)
    print('pooled', o['pooled'].shape, 'tokens', o['tokens'].shape)
