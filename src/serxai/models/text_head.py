import torch
import torch.nn as nn
from typing import Optional


class Head(nn.Module):
    """Classification head for text encoder.
    
    Improvements (v2):
    - Added dropout for regularization
    - Added layer normalization for training stability
    - Configurable hidden dimensions
    """

    def __init__(
        self,
        in_dim: int,
        n_class: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(64, in_dim // 2)
        
        layers = []
        
        # First layer
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        
        # Second layer (output)
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, n_class))
        
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.n_class = n_class
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.net(x)


class HeadLegacy(nn.Module):
    """Original simple MLP head (kept for backward compatibility)."""

    def __init__(self, in_dim: int, n_class: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, max(1, in_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, in_dim // 2), n_class)
        )

    def forward(self, x):
        return self.net(x)


def instantiate_from_state_dict(state_dict: dict, in_dim: int) -> "Head":
    """Helper to instantiate a Head whose final dim is inferred from state_dict.

    Looks for the final linear layer weight key ending with 'net.2.weight' and uses its shape[0]
    as the number of classes. Falls back to 4 if not found.
    """
    n_class = 4
    for k, v in state_dict.items():
        if k.endswith("net.2.weight"):
            n_class = v.shape[0]
            break
    head = Head(in_dim=in_dim, n_class=n_class)
    return head
