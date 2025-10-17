"""GAT fusion placeholder: builds graph between audio/text nodes and fuses features."""
import torch.nn as nn

class GATFusion(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, heads=4, layers=2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        # placeholder: simple MLP fusion
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, audio_feats, text_feats):
        # audio_feats: (B, Ta, D), text_feats: (B, Tt, D)
        # naive pooling + concat
        a = audio_feats.mean(dim=1)
        t = text_feats.mean(dim=1)
        f = (a + t) / 2
        return self.mlp(f)
