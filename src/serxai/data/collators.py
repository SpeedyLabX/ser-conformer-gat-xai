"""Tokenization and collate utilities for Text Branch.

Uses HuggingFace tokenizer when available, otherwise falls back to a simple
whitespace tokenizer and maps tokens to incremental ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from serxai.data.labels import canonicalize_label
from torch.nn.utils.rnn import pad_sequence


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

        labels = [canonicalize_label(b.get("label")) for b in batch]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor(labels, dtype=torch.long)}


@dataclass
class MultimodalCollator:
    """Collate function that pads audio frames and tokenises text."""

    tokenizer_any: Any
    max_length: int = 128
    audio_padding_value: float = 0.0

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        audio_seqs = [
            torch.as_tensor(item["audio_features"], dtype=torch.float32) for item in batch
        ]
        audio_lengths = torch.tensor([seq.shape[0] for seq in audio_seqs], dtype=torch.long)
        audio_padded = pad_sequence(
            audio_seqs, batch_first=True, padding_value=self.audio_padding_value
        )
        max_frames = audio_padded.size(1)
        audio_mask = (
            torch.arange(max_frames, device=audio_lengths.device)
            .unsqueeze(0)
            .expand(len(batch), max_frames)
            >= audio_lengths.unsqueeze(1)
        )

        texts = [item["text"] for item in batch]
        if hasattr(self.tokenizer_any, "__call__"):
            tokens = self.tokenizer_any(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
        else:
            # fallback to SimpleTokenizer-style encode
            ids = [self.tokenizer_any.encode(t, max_length=self.max_length) for t in texts]
            maxlen = min(max(len(x) for x in ids) if ids else 0, self.max_length)
            input_ids = torch.zeros((len(ids), maxlen), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for i, seq in enumerate(ids):
                L = min(len(seq), maxlen)
                if L > 0:
                    input_ids[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
                    attention_mask[i, :L] = 1

        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {
            "audio_features": audio_padded,
            "audio_lengths": audio_lengths,
            "audio_mask": audio_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "metadata": [item["raw"] for item in batch],
        }

