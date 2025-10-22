"""Audio preprocessing helpers: z-score, framing, augmentation (specaugment placeholder)"""
import numpy as np

def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)

def frame_audio(x: np.ndarray, frame_len: int = 512, hop: int = 256):
    # placeholder framing (no overlap handling)
    n_frames = max(1, (len(x) - frame_len) // hop + 1)
    frames = [x[i*hop:i*hop+frame_len] for i in range(n_frames)]
    return np.stack(frames)
