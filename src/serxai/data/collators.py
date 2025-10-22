"""Tokenization and collate utilities for Text Branch.

Uses HuggingFace tokenizer when available, otherwise falls back to a simple
whitespace tokenizer and maps tokens to incremental ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from pathlib import Path


class SimpleTokenizer:
    """A naive whitespace tokenizer with built vocab built from dataset on demand."""

    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1}

    def encode(self, text: str, max_length: int = 128):
        toks = text.strip().split()
        ids = []
        for t in toks[: max_length - 1]:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
            ids.append(self.vocab[t])
        return ids


@dataclass
class TextCollator:
    tokenizer_any: Any  # either HF tokenizer or SimpleTokenizer
    max_length: int = 128

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        # try HF style
        if hasattr(self.tokenizer_any, "__call__"):
            toked = self.tokenizer_any(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            input_ids = toked["input_ids"]
            attention_mask = toked["attention_mask"]
        else:
            # fallback
            ids = [self.tokenizer_any.encode(t, max_length=self.max_length) for t in texts]
            maxlen = max(len(x) for x in ids)
            maxlen = min(maxlen, self.max_length)
            input_ids = torch.zeros((len(ids), maxlen), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for i, seq in enumerate(ids):
                L = min(len(seq), maxlen)
                if L > 0:
                    input_ids[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
                    attention_mask[i, :L] = 1

        labels = []
        for b in batch:
                lab = b.get("label")
                if lab is None:
                    labels.append(-1)
                else:
                    # if label is already integer-like, keep it
                    if isinstance(lab, int):
                        labels.append(lab)
                    else:
                        # map common canonical labels to ints, fallback: attempt int conversion
                        label_map = {
                            "neutral": 0,
                            "angry": 1,
                            "frustration": 2,
                            "happy": 3,
                            "sad": 4,
                            "fear": 5,
                            "disgust": 6,
                        }
                        if isinstance(lab, str):
                            key = lab.lower().strip()
                            labels.append(label_map.get(key, -1))
                        else:
                            try:
                                labels.append(int(lab))
                            except Exception:
                                labels.append(-1)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor(labels, dtype=torch.long)}

