"""Utility that binds a tokenizer and the TextEncoder for training/inference.

Uses HF tokenizer when available, otherwise uses SimpleTokenizer. Provides
`encode_batch` and `forward_batch` helpers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
from pathlib import Path

from serxai.models.text_encoder import TextEncoder
from serxai.data.collators import SimpleTokenizer


class TokenizerEncoder:
    def __init__(self, backbone: str = "roberta-base", proj_dim: Optional[int] = 256, force_fallback: bool = False, max_length: int = 128):
        self.backbone = backbone
        self.max_length = max_length
        self.force_fallback = force_fallback or (backbone == "fallback")
        # tokenizer
        try:
            if not self.force_fallback:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(backbone)
            else:
                raise Exception("force fallback")
        except Exception:
            self.tokenizer = SimpleTokenizer()

        # encoder
        self.encoder = TextEncoder(backbone=backbone, proj_dim=proj_dim, force_fallback=force_fallback)

    def encode_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Return input_ids and attention_mask tensors for a list of texts."""
        if hasattr(self.tokenizer, "__call__"):
            toked = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            return {"input_ids": toked["input_ids"], "attention_mask": toked["attention_mask"]}
        else:
            ids = [self.tokenizer.encode(t, max_length=self.max_length) for t in texts]
            maxlen = max(len(x) for x in ids) if ids else 0
            maxlen = min(maxlen, self.max_length)
            import torch

            input_ids = torch.zeros((len(ids), maxlen), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for i, seq in enumerate(ids):
                L = min(len(seq), maxlen)
                if L > 0:
                    input_ids[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
                    attention_mask[i, :L] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward_batch(self, texts: List[str], return_tokens: bool = False):
        batch = self.encode_batch(texts)
        return self.encoder(batch["input_ids"], batch["attention_mask"], return_tokens=return_tokens)

