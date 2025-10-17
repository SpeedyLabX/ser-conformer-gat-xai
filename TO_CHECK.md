# 1) Tên repo & mô tả

* **Repo**: `ser-conformer-gat-xai`
* **Description**: Multimodal SER with Compact Conformer (audio) + BERT/RoBERTa (text) + GAT fusion, with XAI visualizations (attention/shap). Target datasets: IEMOCAP (train/eval), RAVDESS/MELD (generalization).

# 2) Nhánh & bảo vệ

* `main`: stable/release (cấm push trực tiếp, yêu cầu PR + 2 reviewers + status checks pass).
* `dev`: integration (yêu cầu PR + 1 reviewer + checks pass).
* Conventional Commits + Semantic Versioning.

# 3) Cấu trúc thư mục (khởi tạo)

```
ser-conformer-gat-xai/
├─ src/
│  ├─ serxai/
│  │  ├─ __init__.py
│  │  ├─ data/
│  │  │  ├─ datamodule.py          # Lightning/vanilla PyTorch loader (IEMOCAP, RAVDESS, MELD)
│  │  │  ├─ preprocess_audio.py    # Z-score, framing, augment (specaug optional)
│  │  │  └─ preprocess_text.py     # tokenization, alignment helpers
│  │  ├─ models/
│  │  │  ├─ conformer.py           # Compact Conformer encoder (2–4 layers)
│  │  │  ├─ text_encoder.py        # BERT/RoBERTa wrapper
│  │  │  ├─ gat_fusion.py          # GAT intra/inter-modal + fusion head
│  │  │  └─ classifier.py          # BiGRU/Pooling + FC + Softmax, Focal/CrossEntropy
│  │  ├─ xai/
│  │  │  ├─ attention_viz.py       # export attention maps
│  │  │  └─ shap_tools.py          # optional SHAP/IG hooks
│  │  └─ utils/
│  │     ├─ metrics.py             # WA/UA, macro-F1, cm
│  │     ├─ train_utils.py
│  │     └─ seed.py
│  └─ cli/
│     ├─ train.py                  # train loop (argparse + config)
│     ├─ evaluate.py               # test/val, export metrics + confusion matrix
│     └─ export_artifacts.py       # save best.ckpt, onnx/torchscript (optional)
├─ configs/
│  ├─ base.yaml
│  ├─ iemocap.yaml
│  └─ meld.yaml
├─ notebooks/
│  ├─ 00_eda.ipynb
│  └─ 10_ablation.ipynb
├─ experiments/
│  ├─ runs/                        # logs, metrics (ignored)
│  └─ ablations/
├─ data/                           # tracked by DVC/LFS (ignored by git)
├─ docs/
│  ├─ model_card.md
│  ├─ architecture.md
│  └─ xai_gallery.md
├─ .github/
│  ├─ workflows/
│  │  ├─ ci.yml
│  │  ├─ train.yml
│  │  └─ docs.yml
│  ├─ ISSUE_TEMPLATE/
│  │  ├─ bug_report.yaml
│  │  └─ feature_request.yaml
│  └─ PULL_REQUEST_TEMPLATE.md
├─ .pre-commit-config.yaml
├─ .gitignore
├─ .gitattributes
├─ dvc.yaml                        # nếu dùng DVC
├─ pyproject.toml                  # hoặc requirements.txt
├─ CODEOWNERS
├─ LICENSE
└─ README.md
```

# 4) Nội dung file khởi tạo

## 4.1 `README.md`

````md
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
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U pip
pip install -e .[dev]  # if pyproject has extras
# or: pip install -r requirements.txt
pre-commit install
````

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

## Results

* Baselines: WavLM-Base + Conformer (no GAT)
* Proposed: + GAT fusion (multi-head), focal loss
  See `docs/model_card.md`, `docs/xai_gallery.md`.

````

## 4.2 `pyproject.toml` (tối giản)
```toml
[project]
name = "serxai"
version = "0.1.0"
description = "Multimodal SER with Conformer–GAT fusion + XAI"
requires-python = ">=3.10"
dependencies = [
  "torch>=2.2",
  "torchaudio>=2.2",
  "transformers>=4.44",
  "huggingface-hub>=0.24",
  "numpy",
  "scikit-learn",
  "pyyaml",
  "tqdm",
  "matplotlib",
  "networkx",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "ruff", "black", "pre-commit", "dvc[gdrive,s3]>=3.50"]

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
````

*(Nếu không dùng `pyproject.toml`, tạo `requirements.txt` tương đương.)*

## 4.3 `configs/base.yaml`

```yaml
seed: 2025
device: "cuda"
trainer:
  epochs: 50
  batch_size: 16
  lr: 3e-4
  weight_decay: 1e-4
  grad_clip: 1.0
  patience: 7
model:
  audio:
    backbone: "wavlm-base"         # 512-dim
    conformer_layers: 3
    conformer_dim: 512
    dropout: 0.1
  text:
    backbone: "roberta-base"
    proj_dim: 512
    dropout: 0.1
  fusion:
    gat_heads: 4
    gat_layers: 2
    gat_hidden: 256
    bigru_hidden: 256
    pooling: "max"
  loss:
    type: "focal"                   # focal | cross_entropy
    gamma: 2.0
    class_weights: null
data:
  dataset: "iemocap"
  root: "data/iemocap"
  num_workers: 4
  split: {train: 0.7, val: 0.15, test: 0.15}
  alignment: "cross_modal_attention"
metrics: ["wa", "ua", "f1_macro", "cm"]
artifacts_dir: "artifacts"
log_dir: "experiments/runs"
```

*(Tạo `configs/iemocap.yaml`, `configs/meld.yaml` kế thừa `base.yaml`, chỉ khác `data.root`, số lớp/epochs, v.v.)*

## 4.4 `.gitignore`

```
# python
.venv/
__pycache__/
*.pyc

# experiments / artifacts
experiments/runs/
artifacts/
outputs/
logs/

# data
data/
*.zip
*.tar
*.tar.gz
*.pt
*.ckpt

# dvc
.dvc/
*.dvc
```

## 4.5 `.gitattributes` (LFS cho dữ liệu lớn)

```
/data/** filter=lfs diff=lfs merge=lfs -text
/artifacts/** filter=lfs diff=lfs merge=lfs -text
```

## 4.6 `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.8
    hooks: [{ id: ruff }]
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks: [{ id: black }]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
```

## 4.7 CI/CD – `.github/workflows/ci.yml`

```yaml
name: CI
on:
  push: { branches: [dev] }
  pull_request: { branches: [dev, main] }
jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: python -m pip install -U pip
      - run: pip install -e .[dev] || pip install -r requirements.txt
      - name: Lint
        run: |
          ruff check .
          black --check .
      - name: Tests
        run: |
          pytest -q --maxfail=1 --disable-warnings --cov=src --cov-report=xml
```

## 4.8 Runner train nhẹ (tùy chọn) – `.github/workflows/train.yml`

```yaml
name: Train-Sanity
on:
  workflow_dispatch:
jobs:
  train-sanity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -e .[dev]
      - name: Dry-run (tiny epoch on toy split)
        run: |
          python -m src.cli.train --config configs/iemocap.yaml --dry-run --override trainer.epochs=1
```

## 4.9 Docs – `.github/workflows/docs.yml` (tối giản, nếu muốn publish Github Pages)

```yaml
name: Docs
on:
  push: { branches: [main] }
jobs:
  build-deploy:
    runs-on: ubuntu-latest
    permissions: { contents: write, pages: write, id-token: write }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install mkdocs-material
      - run: mkdocs build
      - uses: actions/upload-pages-artifact@v3
        with: { path: site }
      - uses: actions/deploy-pages@v4
```

## 4.10 Templates

`.github/PULL_REQUEST_TEMPLATE.md`

```md
## Summary
-

## Checks
- [ ] Lint & tests pass
- [ ] No large binaries (use LFS/DVC)
- [ ] Configs updated (if needed)
- [ ] Docs/Model Card updated (if needed)

## Related
Closes #
```

`.github/ISSUE_TEMPLATE/bug_report.yaml`

```yaml
name: Bug Report
description: Report a bug
body:
  - type: textarea
    id: desc
    attributes: { label: Describe the bug }
    validations: { required: true }
  - type: input
    id: repro
    attributes: { label: Minimal Repro (commit/branch) }
```

`.github/ISSUE_TEMPLATE/feature_request.yaml`

```yaml
name: Feature Request
description: Propose a feature
body:
  - type: textarea
    id: value
    attributes: { label: Value / Rationale }
    validations: { required: true }
```

`CODEOWNERS`

```
/src/              @SpeedyLabX/research-leads @SpeedyLabX/core-admin
/configs/          @SpeedyLabX/research-leads
/docs/             @SpeedyLabX/research-leads
```

`LICENSE` → chọn MIT/Apache-2.0 theo ý bạn.

## 4.11 `docs/model_card.md` (khung)

```md
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
```

---

# 5) Script khởi tạo nhanh (gh CLI)

```bash
# login
gh auth login

# create private repo
gh repo create SpeedyLabX/ser-conformer-gat-xai --private --description "Multimodal SER: Conformer–GAT Fusion with XAI"

# push scaffold
git init
git add .
git commit -m "chore: bootstrap SER Conformer–GAT XAI scaffold"
git branch -M main
git remote add origin git@github.com:SpeedyLabX/ser-conformer-gat-xai.git
git push -u origin main

# create dev branch
git checkout -b dev
git push -u origin dev
```

# 6) Gợi ý DVC (nếu dùng)

```bash
dvc init
dvc remote add -d storage gdrive://<folder-id>   # hoặc s3://bucket/path
dvc add data/iemocap data/ravdess data/meld
git add data/.gitignore *.dvc .dvc/config
git commit -m "chore(dvc): track datasets"
```

# 7) Gợi ý thực thi (mapping đề cương → mã)

* **Audio pipeline**: `preprocess_audio.py` (Z-score, framing) → `conformer.py`.
* **Text pipeline**: `preprocess_text.py` (tokenization) → `text_encoder.py`.
* **Alignment**: cơ chế cross-modal attention (trong `gat_fusion.py`) để đồng bộ A↔T theo thời gian.
* **GAT**: đồ thị động với **intra-modal** (A–A, T–T) và **inter-modal** (A–T) + multi-head attention; export attention maps để đáp ứng yêu cầu XAI.
* **Loss/metrics**: Focal/CrossEntropy + WA/UA/F1-macro, confusion matrix; kèm `xai/attention_viz.py` để sinh hình.
  Những điểm này bám đúng proposal của bạn: dùng **WavLM-Base + Compact Conformer**, **BERT/RoBERTa** cho text, **GAT** cho fusion, đánh giá bằng **WA/UA/F1**, và trực quan hóa attention để giải thích mô hình (XAI). 