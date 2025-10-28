"""Simple GAT module for text nodes with attention export.

Lightweight, single-file implementation to support development and XAI export.
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.attn_r = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, x: torch.Tensor, edge_index: List[Tuple[int, int]], return_attention: bool = False):
        # x: (N, in_dim)
        N = x.shape[0]
        h = self.linear(x)  # (N, heads*out_dim)
        h = h.view(N, self.heads, self.out_dim)  # (N, H, D)

        # compute attention for each edge
        attn_scores = []
        out = torch.zeros((N, self.heads, self.out_dim), device=x.device)
        # naive edge loop (ok for small N during development)
        for i, j in edge_index:
            hi = h[i].unsqueeze(0)  # (1,H,D)
            hj = h[j].unsqueeze(0)
            score = (hi * self.attn_l).sum(-1) + (hj * self.attn_r).sum(-1)  # (1,H)
            score = F.leaky_relu(score)
            attn_scores.append(((i, j), score.squeeze(0)))

        # build normalized attention per destination node for aggregation
        # collect per dest
        dest_map = {}
        for (i, j), score in attn_scores:
            dest_map.setdefault(j, []).append((i, score))

        attn_export = {}
        for j, incoming in dest_map.items():
            idxs = [i for i, s in incoming]
            scores = torch.stack([s for i, s in incoming], dim=0)  # (M, H)
            weights = F.softmax(scores, dim=0)  # softmax over sources
            # aggregate
            h_src = h[idxs]  # (M, H, D)
            # expand weights
            w = weights.unsqueeze(-1)  # (M, H, 1)
            agg = (w * h_src).sum(0)  # (H, D)
            out[j] = agg
            # save attention
            attn_export[j] = {"sources": idxs, "weights": weights.detach().cpu()}

        out = out.view(N, self.heads * self.out_dim)
        out = F.elu(out)
        if return_attention:
            return out, attn_export
        return out


class SimpleGAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, heads: int = 4, n_layers: int = 2):
        super().__init__()
        layers = []
        layers.append(SimpleGATLayer(in_dim, hidden, heads=heads))
        for _ in range(n_layers - 1):
            layers.append(SimpleGATLayer(hidden * heads, hidden, heads=heads))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: List[Tuple[int, int]], return_attention: bool = False):
        attns = []
        for l in self.layers:
            if return_attention:
                x, attn = l(x, edge_index, return_attention=True)
                attns.append(attn)
            else:
                x = l(x, edge_index, return_attention=False)
        if return_attention:
            return x, attns
        return x

