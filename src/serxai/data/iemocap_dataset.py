"""Simple dataset loader for the IEMOCAP manifest (JSONL)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def read_manifest(jsonl_path: Path) -> List[Dict]:
    js = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            js.append(json.loads(line))
    return js


class IEMOCAPTextDataset:
    """A minimal dataset wrapper around the JSONL manifest.

    Each item is a dict containing keys: 'utterance_id','text','label','wav','start','end', 'session'
    """
    def __init__(self, manifest_path: Path, max_samples: Optional[int] = None):
        self.manifest_path = Path(manifest_path)
        self.records = read_manifest(self.manifest_path)
        if max_samples:
            self.records = self.records[:max_samples]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    def __iter__(self) -> Iterable[Dict]:
        for r in self.records:
            yield r


def train_val_split(records: List[Dict], val_ratio: float = 0.1, seed: int = 42):
    import random

    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    cut = int(len(records) * (1 - val_ratio))
    train_idx = idx[:cut]
    val_idx = idx[cut:]
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    return train, val


def collate_from_manifest(batch: List[Dict], collator_any) -> Dict:
    """Utility to collate a batch of manifest records using provided collator."""
    return collator_any(batch)


