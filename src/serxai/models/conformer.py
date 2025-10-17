"""Compact Conformer encoder placeholder"""
import torch.nn as nn

class CompactConformer(nn.Module):
    def __init__(self, input_dim=80, d_model=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # A minimal stack placeholder
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead=8, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x):
        # x: (B, T, F)
        h = self.input_proj(x)
        h = h.transpose(0,1)  # T,B,D for transformer
        for l in self.layers:
            h = l(h)
        h = h.transpose(0,1)
        return h
