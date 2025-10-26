"""IEMOCAP dataset text preprocessing utilities.

Provides helpers to parse IEMOCAP transcription, lab, and categorical emotion files
and produce a manifest (JSONL / CSV) of utterances with text, timestamps, labels
and wav paths.

The parser is intentionally defensive to work with partial folders and multiple
categorical annotation files (it will aggregate and take majority vote when
multiple annotators exist).
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


UTT_RE = re.compile(r"(?P<utt>\S+) \[(?P<start>[\d.]+)-(?P<end>[\d.]+)\]:\s*(?P<text>.*)")
CAT_RE = re.compile(r"(?P<utt>\S+)\s*:(?P<label>[^;]+);")
EMO_HEADER_RE = re.compile(r"^\[(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\]\s+(?P<utt>\S+)\s+(?P<code>\S+)\b")


def parse_transcription_file(path: Path) -> List[Dict]:
    """Parse a transcription text file into utterance dicts.

    Each line of the file normally looks like:
        Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.

    Returns a list of dicts with keys: utterance_id, start, end, text
    """
    out = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = UTT_RE.match(line)
        if m:
            out.append({
                "utterance_id": m.group("utt"),
                "start": float(m.group("start")),
                "end": float(m.group("end")),
                "text": m.group("text").strip(),
            })
        # else: ignore header lines or other session metadata
    return out


def parse_categorical_file(path: Path) -> Dict[str, str]:
    """Parse a categorical annotation file into a mapping utt -> label.

    The files use lines such as:
        Ses01F_impro01_F000 :Neutral state; ()
    """
    mapping: Dict[str, str] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = CAT_RE.match(line)
        if m:
            utt = m.group("utt")
            label = m.group("label").strip()
            mapping[utt] = label
    return mapping


def parse_emo_eval_file(path: Path) -> Dict[str, str]:
    """Parse EmoEvaluation header-style files for short codes (e.g., 'neu', 'fru').

    Lines look like:
      [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
    Returns mapping utt -> short_code (lowercased)
    """
    mapping: Dict[str, str] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        m = EMO_HEADER_RE.match(line)
        if m:
            utt = m.group("utt")
            code = m.group("code").strip()
            mapping[utt] = code.lower()
    return mapping


def collect_categorical_for_session(cat_dir: Path) -> Dict[str, str]:
    """Collect labels from all categorical files in a session's categorical dir.

    If multiple annotators exist for the same utterance, take the majority vote.
    """
    votes: Dict[str, List[str]] = defaultdict(list)
    if not cat_dir.exists():
        return {}
    for p in sorted(cat_dir.glob("*_cat.txt")):
        m = parse_categorical_file(p)
        for utt, lab in m.items():
            votes[utt].append(lab)
    out: Dict[str, str] = {}
    for utt, lablist in votes.items():
        most = Counter(lablist).most_common(1)[0][0]
        out[utt] = most
    return out


def canonicalize_label(label: Optional[str]) -> Optional[str]:
    # We canonicalize to the 6-token set used for training: ["neu","hap","ang","sad","exc","fru"]
    if label is None:
        return None
    lab = str(label).lower().strip()
    if lab in ("neu", "neutral", "neutral state"):
        return "neu"
    if "happ" in lab or lab == "hap":
        return "hap"
    if "exc" in lab and not lab.startswith("excuse"):
        return "exc"
    if "ang" in lab or "anger" in lab:
        return "ang"
    if "sad" in lab or "sadness" in lab:
        return "sad"
    if "frustr" in lab or lab == "fru":
        return "fru"
    # otherwise unknown / out-of-scope
    return None


def build_manifest(iemocap_root: Path) -> List[Dict]:
    """Walk the provided IEMOCAP dataset folder and produce a list of utterance records.

    Each record contains:
      - utterance_id
      - session
      - wav (relative path)
      - start, end (seconds)
      - speaker
      - text
      - label (canonicalized, optional)

    The function is defensive and will skip utterances missing required pieces.
    """
    root = iemocap_root
    records: List[Dict] = []
    for session in sorted(root.iterdir()):
        if not session.is_dir():
            continue
        dialog = session / "dialog"
        if not dialog.exists():
            continue
        trans_dir = dialog / "transcriptions"
        lab_dir = dialog / "lab"
        wav_dir = dialog / "wav"
        emo_eval_dir = dialog / "EmoEvaluation"

        # collect labels from EmoEvaluation header files (preferred)
        session_labels = {}
        if emo_eval_dir.exists():
            for p in sorted(emo_eval_dir.glob("*.txt")):
                # skip per-annotator categorical files with *_cat.txt under a Categorical subfolder
                if p.name.endswith("_cat.txt"):
                    continue
                m = parse_emo_eval_file(p)
                for utt, code in m.items():
                    if utt not in session_labels:
                        session_labels[utt] = code

        # parse all transcription files
        if not trans_dir.exists():
            continue
        for txt in sorted(trans_dir.glob("*.txt")):
            utts = parse_transcription_file(txt)
            # determine the corresponding wav filename for this conversation
            # transcription filenames are like Ses01F_impro01.txt -> wav Ses01F_impro01.wav
            wav_name = txt.name.rsplit(".", 1)[0] + ".wav"
            wav_path = (wav_dir / wav_name).as_posix() if (wav_dir / wav_name).exists() else ""
            for u in utts:
                utt_id = u["utterance_id"]
                speaker = None
                if "_F" in utt_id:
                    speaker = "F"
                elif "_M" in utt_id:
                    speaker = "M"
                label_raw = session_labels.get(utt_id)
                label = canonicalize_label(label_raw)
                rec = {
                    "utterance_id": utt_id,
                    "session": session.name,
                    "wav": wav_path,
                    "start": float(u["start"]),
                    "end": float(u["end"]),
                    "speaker": speaker,
                    "text": u["text"],
                    "label": label,
                }
                records.append(rec)
    return records


def write_jsonl(records: Iterable[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(records: Iterable[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if not records:
        return
    keys = list(records[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build IEMOCAP text manifest")
    p.add_argument("data_root", help="Path to data/IEMOCAP_dataset")
    p.add_argument("out_jsonl", help="Output JSONL manifest path")
    p.add_argument("--out-csv", help="Optional output CSV path")
    args = p.parse_args()

    recs = build_manifest(Path(args.data_root))
    write_jsonl(recs, Path(args.out_jsonl))
    if args.out_csv:
        write_csv(recs, Path(args.out_csv))
"""Text preprocessing helpers: tokenization wrappers and alignment helpers"""
from typing import List

def tokenize(texts: List[str], tokenizer, max_length: int = 128):
    # tokenizer should be a HuggingFace tokenizer
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
