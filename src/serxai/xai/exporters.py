"""XAI helpers to map attention weights back to tokens and save results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def save_attention_mapping(out_dir: Path, tokens: List[str], attn_export: Dict[int, Dict]):
    out = {"tokens": tokens, "attn": {}}
    for dst, info in attn_export.items():
        out["attn"][str(dst)] = {"sources": info["sources"], "weights": info["weights"].tolist()}
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "attn.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
