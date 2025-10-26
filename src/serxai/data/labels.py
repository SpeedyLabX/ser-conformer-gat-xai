"""Canonical label list and mapping for emotion classes used across the repo.

Keep this file as the single source of truth for label ↔ index mapping.
"""
from typing import List, Dict

LABELS: List[str] = [
    "neu",
    "hap",
    "ang",
    "sad",
    "exc",
    "fru",
]

# string lower -> int mapping
LABEL_MAP: Dict[str, int] = {k: i for i, k in enumerate(LABELS)}


def canonicalize_label(lab) -> int:
    """Return canonical integer label for a label which may be int or string.

    Accepts short tokens (e.g., 'neu','hap','ang','sad','exc','fru') or ints.
    Returns -1 for unknown/missing labels.
    """
    if lab is None:
        return -1
    if isinstance(lab, int):
        return lab if 0 <= lab < len(LABELS) else -1
    if isinstance(lab, str):
        key = lab.lower().strip()
        # allow some common synonyms
        if key in ("neutral", "neu"):
            return LABEL_MAP.get("neu")
        if key in ("happy", "hap"):
            return LABEL_MAP.get("hap")
        if key in ("excited", "exc"):
            return LABEL_MAP.get("exc")
        if key in ("angry", "ang"):
            return LABEL_MAP.get("ang")
        if key in ("sad", "sadness"):
            return LABEL_MAP.get("sad")
        if key in ("frustration", "fru", "frustr"):
            return LABEL_MAP.get("fru")
        return LABEL_MAP.get(key, -1)
    try:
        v = int(lab)
        return v if 0 <= v < len(LABELS) else -1
    except Exception:
        return -1


def id_to_label(idx: int) -> str:
    if 0 <= idx < len(LABELS):
        return LABELS[idx]
    return "UNK"


def map_to_4class_idx(idx: int) -> int:
    """Map a 6-class index to a 4-class index.

    Mapping used: neu->neu(0), hap+exc->hap(1), ang+fru->ang(2), sad->sad(3)
    Returns -1 for unknown indices.
    """
    if idx < 0 or idx >= len(LABELS):
        return -1
    label = LABELS[idx]
    if label == "neu":
        return 0
    if label in ("hap", "exc"):
        return 1
    if label in ("ang", "fru"):
        return 2
    if label == "sad":
        return 3
    return -1
