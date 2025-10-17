# Model Card — SER Conformer–GAT (Multimodal)

## Intended Use
Multimodal speech emotion recognition on acted/conversational datasets (IEMOCAP, RAVDESS, MELD).

## Architecture
- Audio: WavLM-Base -> Compact Conformer (2–4 layers, 512-d)
- Text: BERT/RoBERTa (256–512-d)
- Fusion: GAT (intra/inter-modal), BiGRU + Pooling + FC
- Loss: Cross-Entropy or Focal (γ=2)

## Training & Evaluation
- Splits: train/val/test per config
- Metrics: WA, UA, macro-F1, confusion matrix

## Explainability
- Attention maps exported per batch/utterance
- Optional SHAP/IG for ablation

## Limitations
- Domain shift from acted to in-the-wild
- Class imbalance affects rare emotions

## Safety & Ethics
- Emotional inference risk; avoid sensitive/biometric misuse
