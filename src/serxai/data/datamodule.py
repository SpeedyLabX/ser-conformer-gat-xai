"""PyTorch-style data module producing loaders for multimodal SER training."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Sequence

from torch.utils.data import DataLoader

from serxai.data.collators import MultimodalCollator
from serxai.data.iemocap_dataset import (
    IEMOCAPMultimodalDataset,
    read_manifest,
    train_test_by_sessions,
    train_val_split,
)
from serxai.data.labels import canonicalize_label
from serxai.data.collators import SimpleTokenizer


class DataModule:
    """Lightweight data module handling dataset splits and collators."""

    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        batch_size: int = 16,
        num_workers: int = 4,
        split: Optional[Dict[str, float]] = None,
        seed: int = 42,
        session_split: Optional[Dict[str, Sequence[int]]] = None,
        max_text_length: int = 128,
        max_audio_frames: Optional[int] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.split = split or {"train": 0.8, "val": 0.1, "test": 0.1}
        self.session_split = session_split
        self.max_text_length = max_text_length
        self.max_audio_frames = max_audio_frames
        self.tokenizer = tokenizer if tokenizer is not None else SimpleTokenizer()

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None
        self._collator = MultimodalCollator(
            tokenizer_any=self.tokenizer,
            max_length=self.max_text_length,
        )

    def setup(self, stage: Optional[str] = None):
        records = read_manifest(self.manifest_path)
        records = [r for r in records if canonicalize_label(r.get("label")) != -1]
        if self.session_split:
            train_records, test_records = train_test_by_sessions(
                records,
                train_sessions=self.session_split.get("train", (1, 2, 3, 4)),
                test_sessions=self.session_split.get("test", (5,)),
            )
            train_records, val_records = train_val_split(
                train_records,
                val_ratio=self.split.get("val", 0.1),
                seed=self.seed,
            )
        else:
            rng = random.Random(self.seed)
            rng.shuffle(records)
            n = len(records)
            n_train = int(n * self.split.get("train", 0.8))
            n_val = int(n * self.split.get("val", 0.1))
            train_records = records[:n_train]
            val_records = records[n_train : n_train + n_val]
            test_records = records[n_train + n_val :]

        self._train_ds = IEMOCAPMultimodalDataset(train_records, max_frames=self.max_audio_frames)
        self._val_ds = IEMOCAPMultimodalDataset(val_records, max_frames=self.max_audio_frames)
        self._test_ds = IEMOCAPMultimodalDataset(test_records, max_frames=self.max_audio_frames)

    def _dataloader(self, dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collator,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._dataloader(self._train_ds, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self._val_ds, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self._test_ds, shuffle=False)
