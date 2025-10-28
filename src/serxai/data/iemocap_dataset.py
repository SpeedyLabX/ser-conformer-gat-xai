"""Dataset utilities for IEMOCAP manifests."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import torchaudio
from torchaudio.functional import resample as ta_resample
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from serxai.data.labels import canonicalize_label

try:
    import soundfile as sf  # type: ignore[import]

    _HAS_SF = True
except Exception:  # pragma: no cover - optional dependency
    sf = None
    _HAS_SF = False

import wave


def _load_waveform_fallback(path: Path) -> Tuple[torch.Tensor, int]:
    """Best-effort waveform loader when torchaudio backends are unavailable."""
    if _HAS_SF:
        data, sr = sf.read(str(path), always_2d=False)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            tensor = torch.from_numpy(data).unsqueeze(0)
        else:
            tensor = torch.from_numpy(data.T)
        return tensor, sr

    with wave.open(str(path), "rb") as wav_reader:
        sr = wav_reader.getframerate()
        n_frames = wav_reader.getnframes()
        n_channels = wav_reader.getnchannels()
        sampwidth = wav_reader.getsampwidth()
        frames = wav_reader.readframes(n_frames)

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sampwidth not in dtype_map:
        raise RuntimeError(f"Unsupported sample width {sampwidth} bytes in {path}")

    data = np.frombuffer(frames, dtype=dtype_map[sampwidth]).astype(np.float32)
    if sampwidth == 1:
        data = (data - 128.0) / 128.0
    else:
        scale = float(2 ** (8 * sampwidth - 1))
        data = data / scale

    if n_channels > 1:
        data = data.reshape(-1, n_channels).T
    else:
        data = data.reshape(1, -1)

    tensor = torch.from_numpy(data)
    return tensor, sr


def _load_audio_segment(
    wav_path: Path,
    start: float,
    end: float,
    target_sr: int = 16000,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, int]:
    """Load an audio segment and resample if required."""
    try:
        waveform, sr = torchaudio.load(str(wav_path))
    except Exception:
        waveform, sr = _load_waveform_fallback(wav_path)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = ta_resample(waveform, sr, target_sr)
        sr = target_sr

    if end is None or math.isclose(end, 0.0):
        end = waveform.size(-1) / sr
    start_sample = max(0, int(start * sr))
    end_sample = max(start_sample + 1, int(end * sr))
    segment = waveform[..., start_sample:end_sample]
    if segment.numel() == 0:
        segment = waveform
    return segment.squeeze(0), sr


def _mel_transform(
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int],
) -> Tuple[MelSpectrogram, AmplitudeToDB]:
    mel = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length or n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        mel_scale="htk",
    )
    amp = AmplitudeToDB(stype="power")
    return mel, amp


def read_manifest(jsonl_path: Path) -> List[Dict]:
    js = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            js.append(json.loads(line))
    return js


class IEMOCAPTextDataset:
    """A minimal dataset wrapper around the JSONL manifest.

    Each item is a dict containing keys: 'utterance_id','text','label','wav','start','end', 'session'
    """
    def __init__(self, manifest_path: Path, max_samples: Optional[int] = None):
        self.manifest_path = Path(manifest_path)
        self.records = read_manifest(self.manifest_path)
        if max_samples:
            self.records = self.records[:max_samples]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    def __iter__(self) -> Iterable[Dict]:
        for r in self.records:
            yield r


def train_val_split(records: List[Dict], val_ratio: float = 0.1, seed: int = 42):
    import random

    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    cut = int(len(records) * (1 - val_ratio))
    train_idx = idx[:cut]
    val_idx = idx[cut:]
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    return train, val


def train_test_by_sessions(records: List[Dict], train_sessions=(1, 2, 3, 4), test_sessions=(5,)):
    """Split manifest records into train/test lists based on the numeric session id.

    Records should include a 'session' field which is either an int or a string
    such as 'session1' or 'S1'. The function extracts the first integer found
    and assigns the record to train or test depending on the provided tuples.
    Records with unknown session are placed into the training set by default.
    """
    import re

    train_s = set(int(x) for x in train_sessions)
    test_s = set(int(x) for x in test_sessions)

    train = []
    test = []
    for r in records:
        s = r.get('session', None)
        sess_id = None
        if s is None:
            sess_id = None
        else:
            if isinstance(s, int):
                sess_id = s
            else:
                # attempt to extract a digit
                m = re.search(r"(\d+)", str(s))
                if m:
                    try:
                        sess_id = int(m.group(1))
                    except Exception:
                        sess_id = None
                else:
                    sess_id = None

        if sess_id in test_s:
            test.append(r)
        else:
            # default to train for unknown or train sessions
            train.append(r)

    return train, test


def collate_from_manifest(batch: List[Dict], collator_any) -> Dict:
    """Utility to collate a batch of manifest records using provided collator."""
    return collator_any(batch)


class IEMOCAPMultimodalDataset(Dataset):
    """Multimodal dataset yielding audio log-mels and raw text for each utterance."""

    def __init__(
        self,
        records: Sequence[Dict],
        target_sr: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        max_frames: Optional[int] = None,
    ):
        super().__init__()
        self.records = [r for r in records if canonicalize_label(r.get("label")) != -1]
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_frames = max_frames
        self.mel_transform, self.amp_to_db = _mel_transform(
            target_sr, n_mels, n_fft, hop_length, None
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        wav_path = Path(rec["wav"])
        audio, sr = _load_audio_segment(
            wav_path,
            float(rec.get("start", 0.0)),
            float(rec.get("end", 0.0)),
            target_sr=self.target_sr,
        )
        mel = self.mel_transform(audio.unsqueeze(0))
        mel = self.amp_to_db(mel + 1e-6)
        log_mel = mel.squeeze(0).transpose(0, 1)
        if self.max_frames is not None and log_mel.size(0) > self.max_frames:
            log_mel = log_mel[: self.max_frames]

        label = canonicalize_label(rec.get("label"))
        return {
            "utt": rec.get("utterance_id"),
            "text": rec.get("text", ""),
            "audio_features": log_mel,  # (T, n_mels)
            "audio_length": log_mel.size(0),
            "label": label,
            "raw": rec,
        }


