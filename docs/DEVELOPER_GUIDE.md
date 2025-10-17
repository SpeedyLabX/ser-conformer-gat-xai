# 📘 **SLX02 Developer Guide**

*(For internal research use — SpeedyLabX Research Group)*

## 1️⃣ Overview

This document serves as the **comprehensive guide** for researchers and contributors working on the project:

> **Enhancing Multimodal Speech Emotion Recognition via a Conformer–GAT Fusion Architecture with XAI**

It covers:

* Environment setup
* Dataset organization
* Training and evaluation pipeline
* Logging and artifact management
* Packaging and exporting models

---

## 2️⃣ Environment Setup

### 🧩 Prerequisites

* Python ≥ 3.10
* Git ≥ 2.34
* (Optional) CUDA ≥ 12.0 for GPU acceleration
* Recommended environment manager: **venv** or **conda**

### 🪄 Installation

```bash
# 1. Clone the repository
git clone https://github.com/SpeedyLabX/ser-conformer-gat-xai.git
cd ser-conformer-gat-xai

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate       # (Windows: .venv\Scripts\activate)

# 3. Install dependencies
pip install -U pip
pip install -e .[dev]           # use pyproject.toml extras
# or:
# pip install -r requirements.txt

# 4. Initialize pre-commit hooks
pre-commit install
```

### 🧠 Verifying installation

```bash
python -m src.cli.train --help
python -m src.cli.evaluate --help
```

---

## 3️⃣ Dataset Preparation

### Default Datasets

| Dataset | Source       | Location        | Modality     |
| :------ | :----------- | :-------------- | :----------- |
| IEMOCAP | USC          | `data/iemocap/` | Audio + Text |
| RAVDESS | Ryerson      | `data/ravdess/` | Audio + Text |
| MELD    | EmotionLines | `data/meld/`    | Audio + Text |

### Folder Structure

```
data/
├─ iemocap/
│  ├─ audio/
│  ├─ transcripts/
│  ├─ labels.csv
│  └─ manifest.json
├─ ravdess/
│  ├─ audio/
│  └─ labels.csv
└─ meld/
   ├─ dialogues/
   └─ labels.csv
```

### Data Preprocessing

```bash
# Run standard preprocessing
python -m src.serxai.data.preprocess_audio --dataset iemocap
python -m src.serxai.data.preprocess_text --dataset iemocap
```

You can modify paths and dataset splits in:

```
configs/base.yaml
configs/iemocap.yaml
```

---

## 4️⃣ Training the Model

### Baseline Training

```bash
python -m src.cli.train --config configs/iemocap.yaml
```

### Custom Configuration

You can override parameters directly:

```bash
python -m src.cli.train --config configs/iemocap.yaml \
  --override trainer.epochs=10 \
  --override model.audio.conformer_layers=4 \
  --override model.loss.type="focal"
```

### Logging & Artifacts

All runs are stored in:

```
experiments/runs/<timestamp>/
│
├─ metrics.json
├─ best.ckpt
└─ logs/
```

You can visualize metrics using TensorBoard:

```bash
tensorboard --logdir experiments/runs
```

---

## 5️⃣ Evaluation & Explainability

### Run Evaluation

```bash
python -m src.cli.evaluate --ckpt artifacts/best.ckpt \
  --config configs/iemocap.yaml
```

### Export Attention Maps

```bash
python -m src.cli.evaluate --ckpt artifacts/best.ckpt \
  --export-attn
```

### XAI Tools

| Method               | Module                 | Output                     |
| :------------------- | :--------------------- | :------------------------- |
| Attention Weights    | `xai/attention_viz.py` | Heatmaps                   |
| SHAP                 | `xai/shap_tools.py`    | Local feature importance   |
| Integrated Gradients | `xai/shap_tools.py`    | Gradient-based attribution |

Results are exported to:

```
docs/xai_gallery/
```

---

## 6️⃣ Reproducibility and Random Seeds

Default seed = 2025
To ensure deterministic behavior:

```yaml
seed: 2025
deterministic: true
```

in `configs/base.yaml`.

---

## 7️⃣ Packaging and Deployment

### Export Model

```bash
python -m src.cli.export_artifacts --ckpt artifacts/best.ckpt --format torchscript
```

Supported formats:

* `torchscript` → For PyTorch runtime
* `onnx` → For lightweight inference/export

### Package as Python Module

```bash
# Build wheel
python -m build

# Install locally
pip install dist/serxai-0.1.0-py3-none-any.whl
```

### Inference Example

```python
from serxai.models import ConformerGATModel
from serxai.utils import load_checkpoint

model = ConformerGATModel.from_checkpoint("artifacts/best.ckpt")
pred = model.predict(audio_path="sample.wav", text="I'm so happy to see you!")
print(pred)
```

---

## 8️⃣ DVC (Optional – for data versioning)

```bash
# Initialize
dvc init
dvc remote add -d storage gdrive://<folder-id>

# Track dataset
dvc add data/iemocap data/ravdess
git add data/.gitignore *.dvc .dvc/config
git commit -m "chore: add datasets to DVC"
```

---

## 9️⃣ Troubleshooting

| Issue                 | Cause                 | Solution                                                    |
| :-------------------- | :-------------------- | :---------------------------------------------------------- |
| CUDA error            | Mismatch CUDA version | Update torch to match CUDA toolkit                          |
| Tokenizer error       | Missing model         | Run `transformers-cli download`                             |
| DVC permission denied | GDrive token expired  | `dvc remote modify storage gdrive_use_service_account true` |

---

## 🔚 Summary

✅ You now have:

* Configurable multimodal SER pipeline
* Modular Conformer–GAT fusion model
* XAI-ready evaluation workflow
* Packaging and export utilities

**Next step:** Read [`CONTRIBUTING.md`](../CONTRIBUTING.md) (Doc #2) for push, PR, and review workflow.

---

### 📄 Document Info

* **File:** `docs/DEVELOPER_GUIDE.md`
* **Maintainer:** Le Nguyen Gia Hung (`@hei1sme`)
* **Last updated:** October 2025
* **License:** CC BY-NC 4.0
* **Project:** SLX02 – SpeedyLabX Research Group