"""Canonical label list and mapping for emotion classes used across the repo.

Keep this file as the single source of truth for label ↔ index mapping.
"""
from typing import List, Dict

LABELS: List[str] = [
    "neutral",
    "angry",
    "frustration",
    "happy",
    "sad",
    "fear",
    "disgust",
]

# string lower -> int mapping
LABEL_MAP: Dict[str, int] = {k: i for i, k in enumerate(LABELS)}

def canonicalize_label(lab) -> int:
    """Return canonical integer label for a label which may be int or string.

    Returns -1 for unknown/missing labels.
    """
    if lab is None:
        return -1
    if isinstance(lab, int):
        return lab
    if isinstance(lab, str):
        key = lab.lower().strip()
        return LABEL_MAP.get(key, -1)
    try:
        return int(lab)
    except Exception:
        return -1
