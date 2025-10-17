"""Training utilities: checkpointing and misc helpers"""
import torch


def save_checkpoint(state, path: str):
    """Save training checkpoint state dict to path."""
    torch.save(state, path)


def load_checkpoint(path: str, map_location=None):
    return torch.load(path, map_location=map_location)
