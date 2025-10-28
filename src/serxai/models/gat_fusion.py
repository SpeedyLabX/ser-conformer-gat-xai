"""GAT-based fusion module that operates on audio + text token graphs."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from serxai.models.gat_text import SimpleGAT


class GATFusion(nn.Module):
    """Fuse audio and text token embeddings through a lightweight GAT graph.

    The module builds a bipartite graph between audio frames and text tokens for each sample.
    Incoming edges include self-loops, audio→text, and text→audio connections so that the
    SimpleGAT encoder can propagate information across modalities. The resulting embeddings
    are pooled separately (audio/text) and combined via an MLP.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for SimpleGAT output size.")
        gat_hidden = hidden_dim // heads
        self.gat = SimpleGAT(in_dim=in_dim, hidden=gat_hidden, heads=heads, n_layers=layers)
        fusion_in_dim = hidden_dim * 2  # audio summary + text summary
        self.mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    @staticmethod
    @lru_cache(maxsize=32)
    def _build_edge_index(n_audio: int, n_text: int) -> List[Tuple[int, int]]:
        """Create a cached list of directed edges for the bipartite graph."""
        edges: List[Tuple[int, int]] = []
        total = n_audio + n_text
        # self-loops ensure every node has at least one incoming edge
        for idx in range(total):
            edges.append((idx, idx))

        # cross-modal connections
        text_offset = n_audio
        for a in range(n_audio):
            for t in range(n_text):
                b = text_offset + t
                edges.append((a, b))  # audio influences text
                edges.append((b, a))  # text influences audio
        return edges

    def forward(
        self,
        audio_feats: Union[torch.Tensor, Sequence[torch.Tensor]],
        text_feats: Union[torch.Tensor, Sequence[torch.Tensor]],
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """Fuse audio/text sequences.

        Args:
            audio_feats: Tensor (B, Ta, D)
            text_feats: Tensor (B, Tt, D)
            return_attention: if True, returns per-layer attention snapshots for XAI.

        Returns:
            fused: (B, hidden_dim)
            attentions (optional): list of length B containing layer-wise attention dicts.
        """
        if isinstance(audio_feats, torch.Tensor):
            audio_list = [audio_feats[b] for b in range(audio_feats.size(0))]
        else:
            audio_list = list(audio_feats)

        if isinstance(text_feats, torch.Tensor):
            text_list = [text_feats[b] for b in range(text_feats.size(0))]
        else:
            text_list = list(text_feats)

        if len(audio_list) != len(text_list):
            raise ValueError("Audio and text batches must be aligned.")

        batch_outputs: List[torch.Tensor] = []
        batch_attn: List = []

        for a_seq, t_seq in zip(audio_list, text_list):
            n_audio = a_seq.size(0)
            n_text = t_seq.size(0)
            if n_audio == 0 or n_text == 0:
                raise ValueError("Fusion received empty audio or text sequence.")

            nodes = torch.cat([a_seq, t_seq], dim=0)
            edges = self._build_edge_index(n_audio, n_text)
            gat_out, attn = self.gat(nodes, edges, return_attention=True)

            audio_repr = gat_out[:n_audio].mean(dim=0)
            text_repr = gat_out[n_audio:].mean(dim=0)
            combined = torch.cat([audio_repr, text_repr], dim=0)
            batch_outputs.append(combined)
            if return_attention:
                batch_attn.append(attn)

        fused = self.mlp(self.dropout(torch.stack(batch_outputs, dim=0)))
        if return_attention:
            return fused, batch_attn
        return fused


__all__ = ["GATFusion"]
