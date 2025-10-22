"""Graph builder for text: sliding-window edges and semantic kNN on token embeddings.

This module produces node features and edge index lists suitable for a GAT.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors


def sliding_window_edges(n_tokens: int, k: int = 2) -> List[Tuple[int, int]]:
    edges = []
    for i in range(n_tokens):
        for j in range(max(0, i - k), min(n_tokens, i + k + 1)):
            if i == j:
                continue
            edges.append((i, j))
    return edges


def semantic_knn_edges(embeddings: np.ndarray, k: int = 2) -> List[Tuple[int, int]]:
    if embeddings.shape[0] <= k:
        return []
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, embeddings.shape[0]), algorithm="auto").fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    edges = []
    for i, neigh in enumerate(indices):
        for j in neigh:
            if i == j:
                continue
            edges.append((i, j))
    return edges


def build_text_graph(token_embeddings: np.ndarray, window_k: int = 2, knn_k: int = 2):
    n = token_embeddings.shape[0]
    edges = sliding_window_edges(n, k=window_k)
    try:
        knn = semantic_knn_edges(token_embeddings, k=knn_k)
        edges.extend(knn)
    except Exception:
        pass
    # deduplicate
    edges = list(set(edges))
    return {
        "n_nodes": n,
        "edges": edges,
        "node_feats": token_embeddings,
    }

