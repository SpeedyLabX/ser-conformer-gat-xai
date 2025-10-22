"""Example runner for the Text Branch: load one batch and run TextEncoder."""
from pathlib import Path
import torch

from serxai.models.text_encoder import TextEncoder
from serxai.data.iemocap_dataset import IEMOCAPTextDataset
from serxai.data.collators import TextCollator


def main():
    manifest = Path(__file__).resolve().parents[1] / "data" / "iemocap_manifest.jsonl"
    ds = IEMOCAPTextDataset(manifest, max_samples=8)

    # prefer HF tokenizer if installed
    # If we are going to run the encoder in fallback mode, prefer the SimpleTokenizer
    # to avoid token id ranges that exceed the fallback embedding size.
    try:
        from transformers import AutoTokenizer
        hf_available = True
    except Exception:
        hf_available = False

    from serxai.data.collators import SimpleTokenizer
    if hf_available:
        # use HF tokenizer only if we plan to use HF backbone; here we force fallback,
        # so prefer the simple tokenizer to keep ids small and compatible with fallback embed.
        tok = SimpleTokenizer()
    else:
        tok = SimpleTokenizer()

    collator = TextCollator(tok, max_length=64)
    batch = collator([ds[i] for i in range(min(4, len(ds)))])

    enc = TextEncoder(backbone="fallback", proj_dim=128, pooling="mean", force_fallback=True)
    out = enc(batch["input_ids"], batch["attention_mask"], return_tokens=True)
    print("pooled", out["pooled"].shape)
    print("tokens", out["tokens"].shape)
    # print first token vector snippet
    print("first token vector sample:", out["tokens"][0, 0, :5].tolist())


if __name__ == "__main__":
    main()
