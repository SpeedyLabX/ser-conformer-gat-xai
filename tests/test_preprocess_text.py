import json
from pathlib import Path

from serxai.data.preprocess_text import build_manifest, parse_transcription_file, parse_categorical_file


def write_text(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_parse_transcription_and_categorical(tmp_path):
    # create minimal session structure
    root = tmp_path / "IEMOCAP_dataset"
    ses = root / "SessionX"
    trans = ses / "dialog" / "transcriptions"
    emo_cat = ses / "dialog" / "EmoEvaluation" / "Categorical"
    wav = ses / "dialog" / "wav"
    trans.mkdir(parents=True)
    emo_cat.mkdir(parents=True)
    wav.mkdir(parents=True)

    tfile = trans / "SessX_impro01.txt"
    tcontent = """
SessX_impro01_F000 [001.000-002.000]: Hello there.
SessX_impro01_M000 [003.000-004.500]: Hi.
"""
    write_text(tfile, tcontent)

    catfile = emo_cat / "SessX_impro01_e1_cat.txt"
    ccontent = """
SessX_impro01_F000 :Neutral state; ()
SessX_impro01_M000 :Anger; ()
"""
    write_text(catfile, ccontent)

    # create matching wav file path (empty)
    (wav / "SessX_impro01.wav").write_text("", encoding="utf-8")

    # run build_manifest
    recs = build_manifest(root)
    assert isinstance(recs, list)
    assert any(r["utterance_id"] == "SessX_impro01_F000" for r in recs)
    f = [r for r in recs if r["utterance_id"] == "SessX_impro01_F000"][0]
    assert f["text"] == "Hello there."
    assert f["label"] == "neutral"
