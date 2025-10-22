import torch.nn as nn
from typing import Optional


class Head(nn.Module):
    """Small MLP head used by finetune_text.py.

    Kept intentionally simple so scripts can import and reuse it.
    """

    def __init__(self, in_dim: int, n_class: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, max(1, in_dim // 2)), nn.ReLU(), nn.Linear(max(1, in_dim // 2), n_class))

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
