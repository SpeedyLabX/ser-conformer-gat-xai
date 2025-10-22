from pathlib import Path
import numpy as np
from serxai.models.tokenizer_encoder import TokenizerEncoder
from serxai.data.graph_builder import build_text_graph
import torch
from serxai.models.gat_text import SimpleGAT
from serxai.xai.exporters import save_attention_mapping


def main():
    texts = ["I am not happy with this service"]
    te = TokenizerEncoder(backbone="fallback", proj_dim=64, force_fallback=True, max_length=32)
    enc_out = te.forward_batch(texts, return_tokens=True)
    tokens = enc_out["tokens"][0].detach().numpy()  # (T, D)
    # produce token strings via simple split for demo
    token_strs = texts[0].split()
    graph = build_text_graph(tokens, window_k=1, knn_k=1)
    gat = SimpleGAT(in_dim=tokens.shape[1], hidden=32, heads=2, n_layers=2)
    feats, attns = gat(torch.tensor(graph["node_feats"], dtype=torch.float32), graph["edges"], return_attention=True)
    # attns is a list (per layer) of dict(dst -> {sources, weights})
    out = save_attention_mapping(Path("artifacts/xai_demo"), token_strs, attns[0])
    print("Saved attention mapping to", out)


if __name__ == "__main__":
    main()
