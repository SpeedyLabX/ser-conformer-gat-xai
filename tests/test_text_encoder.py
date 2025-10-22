import torch

from serxai.models.text_encoder import TextEncoder


def test_text_encoder_shapes():
    # Use a small vocab id range to ensure embedding fallback works
    enc = TextEncoder(backbone="nonexistent-model", proj_dim=128, force_fallback=True)
    input_ids = torch.randint(0, 1000, (4, 20))
    attention_mask = torch.ones_like(input_ids)
    out = enc(input_ids, attention_mask, return_tokens=True)
    assert "pooled" in out
    assert "tokens" in out
    assert out["pooled"].shape == (4, 128)
    assert out["tokens"].shape == (4, 20, 128)
