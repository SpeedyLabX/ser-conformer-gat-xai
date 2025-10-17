# SER: Conformer–GAT Fusion with XAI

Multimodal Speech Emotion Recognition: Compact Conformer (audio) + BERT/RoBERTa (text) + GAT fusion, with explainability (attention maps, SHAP/IG).

## Highlights
- Audio: WavLM-Base → Compact Conformer (2–4 layers)
- Text: BERT-base/RoBERTa-base
- Fusion: GAT with intra-/inter-modal edges; BiGRU + Pooling + FC
- Metrics: WA, UA, Macro-F1, Confusion Matrix
- XAI: attention visualization, optional SHAP

## Setup
```bash
# Python 3.11 recommended
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install -U pip
pip install -e .[dev]  # if pyproject has extras
# or: pip install -r requirements.txt
pre-commit install
```

## Data

Place datasets under `data/` or configure DVC remotes. Paths are set in `configs/*.yaml`.

## Train

```bash
python -m src.cli.train --config configs/iemocap.yaml
```

## Evaluate

```bash
python -m src.cli.evaluate --ckpt artifacts/best.ckpt --config configs/iemocap.yaml
```

## Export XAI

```bash
python -m src.cli.evaluate --ckpt artifacts/best.ckpt --export-attn
```

See `docs/` for architecture and model card.
