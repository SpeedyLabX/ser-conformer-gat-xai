"""Example: build a manifest from local IEMOCAP dataset and write JSONL."""
from pathlib import Path
from serxai.data.preprocess_text import build_manifest, write_jsonl


def main():
    root = Path(__file__).resolve().parents[1] / "data" / "IEMOCAP_dataset"
    out = Path(__file__).resolve().parents[1] / "data" / "iemocap_manifest.jsonl"
    recs = build_manifest(root)
    write_jsonl(recs, out)
    print(f"Wrote {len(recs)} records to {out}")


if __name__ == "__main__":
    main()
