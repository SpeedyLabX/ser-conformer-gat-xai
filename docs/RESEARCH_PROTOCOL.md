# 🧪 **Research Protocol and Experimentation Framework**

**Project:** *SLX02 – Enhancing Multimodal Speech Emotion Recognition via a Conformer–GAT Fusion Architecture with XAI*
**Organization:** *SpeedyLabX Research Group, FPT University*
**License:** *CC BY-NC 4.0*

---

## 1️⃣ Objective

This document defines the **standardized research protocol** for conducting, documenting, and reporting experiments within the SLX02 project.
It ensures **reproducibility**, **traceability**, and **academic integrity** in all experimental results submitted for publication.

The protocol is designed for:

* Research reproducibility across team members
* Consistent experiment documentation
* Ethical and transparent data/model management

---

## 2️⃣ Research Lifecycle Overview

```
Design → Experimentation → Analysis → Reporting → Publication
```

### Workflow summary:

| Stage              | Output                                       | Responsible          |
| :----------------- | :------------------------------------------- | :------------------- |
| 1. Design          | Research proposal, hypothesis, baseline plan | Core team            |
| 2. Implementation  | Code modules, configs, model prototypes      | Assigned developer   |
| 3. Experimentation | Results, logs, ablation records              | Researcher-in-charge |
| 4. Analysis        | Metrics tables, charts, interpretability     | Group lead           |
| 5. Reporting       | Draft for paper/technical appendix           | Writing team         |
| 6. Publication     | Submission package, supplementary repo       | Core-admin + mentor  |

---

## 3️⃣ Experiment Management

### Directory structure

All experiments are stored under:

```
experiments/
├─ runs/
│  ├─ 2025-10-17_01_conformer_baseline/
│  │  ├─ config_used.yaml
│  │  ├─ metrics.json
│  │  ├─ best.ckpt
│  │  ├─ train.log
│  │  └─ notes.md
│  ├─ 2025-10-20_02_conformer_gat/
│  └─ ...
└─ ablations/
   ├─ table_summary.csv
   └─ shap_visuals/
```

### File naming convention

```
YYYY-MM-DD_<index>_<experiment_name>
```

Examples:

```
2025-10-21_03_gat_layers_ablation
2025-10-23_05_audio_only_baseline
```

---

## 4️⃣ Configuration and Metadata

Each run must include a snapshot of its configuration:

**`config_used.yaml`**

```yaml
dataset: iemocap
audio_model: wavlm-base
text_model: roberta-base
fusion: GAT
gat_layers: 2
gat_heads: 4
batch_size: 16
epochs: 50
loss: focal
optimizer: adamw
seed: 2025
timestamp: "2025-10-17_20:34:00"
run_id: SLX02_EXP03
```

**`metrics.json`**

```json
{
  "wa": 0.871,
  "ua": 0.845,
  "macro_f1": 0.854,
  "loss": 0.423,
  "epoch_best": 37,
  "params": 18.2e6
}
```

**`notes.md`**

* Purpose:

  > Compare Conformer-only vs. Conformer–GAT with shared node features.
* Observation:

  > Fusion improves recall for “sad” and “fear” classes by ~5%.
* Decision:

  > Proceed with GAT(2-layer, 4-head) for main experiment.

---

## 5️⃣ Logging Standards

Use unified formatting for logs:

```
[2025-10-17 20:35:12] Epoch 10 | Loss=0.621 | WA=0.753 | UA=0.740 | F1=0.745
[2025-10-17 20:38:21] Epoch 20 | Loss=0.513 | WA=0.812 | UA=0.798 | F1=0.801
[2025-10-17 20:44:59] Early stopping triggered (patience=7)
```

Each log file should:

* Record seed, device, dataset, config
* Include both **train** and **validation** metrics
* End with summary block:

  ```
  Best epoch: 38 | WA=0.862 | UA=0.840 | F1=0.847
  Model saved to: artifacts/best.ckpt
  ```

---

## 6️⃣ Ablation Study Protocol

### Purpose

To evaluate the **effect of architectural and training choices** on model performance.

### Structure

| Variable      | Values                            | Notes                                           |
| :------------ | :-------------------------------- | :---------------------------------------------- |
| GAT layers    | [1, 2, 3]                         | trade-off between interpretability and accuracy |
| Fusion method | [concat, gated, GAT]              | tested under same hidden dim                    |
| Loss function | [CE, Focal(γ=2), Label Smoothing] | robustness comparison                           |
| Text encoder  | [BERT-base, RoBERTa-base]         | cross-lingual effect                            |

### Template (`ablations/table_summary.csv`)

| Experiment ID | Config             | WA    | UA    | F1    | ΔF1(%) | Notes                      |
| :------------ | :----------------- | :---- | :---- | :---- | :----- | :------------------------- |
| EXP01         | Conformer baseline | 0.821 | 0.798 | 0.803 | –      | baseline                   |
| EXP02         | + GAT (2L,4H)      | 0.871 | 0.845 | 0.854 | +6.3   | improved contextual fusion |
| EXP03         | + Focal Loss       | 0.878 | 0.851 | 0.860 | +0.7   | better rare class handling |

---

## 7️⃣ Explainability Protocol (XAI)

### Required deliverables per experiment

| Method                  | Output                     | Format          |
| :---------------------- | :------------------------- | :-------------- |
| Attention visualization | Class-level attention maps | `.png`, `.csv`  |
| SHAP                    | Top-10 features per sample | `.json`, `.png` |
| Integrated Gradients    | Attribution overlay        | `.png`          |
| Summary report          | Modal contribution stats   | `notes.md`      |

### Directory

```
docs/xai_gallery/
├─ attention/
├─ shap/
├─ integrated_gradients/
└─ summary/
```

### Reporting format (`xai_summary.md`)

```
Experiment: EXP02 (Conformer–GAT)
Dataset: IEMOCAP
---------------------------------
• Attention peaks align with emotional keywords ("angry", "laugh").
• SHAP shows dominant audio cues in low-frequency band (<300Hz).
• IG confirms consistent multimodal attribution patterns.
```

---

## 8️⃣ Statistical Reporting

Each experiment must include:

* **Mean ± SD** over ≥ 3 random seeds
* 95% confidence interval for major metrics
* Statistical significance (paired t-test) if comparing variants

Example:

```
WA = 0.871 ± 0.008
UA = 0.845 ± 0.010
Macro-F1 = 0.854 ± 0.007
p < 0.05 vs. baseline
```

All final reported results must be **averaged**, not single-run.

---

## 9️⃣ Visualization and Figures

### Standard plots:

* Confusion matrix (`plots/confusion_matrix.png`)
* ROC curves (per class)
* SHAP summary plots
* Attention heatmaps (word/time alignment)

### Matplotlib settings

```python
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})
```

### Color palette

* Use color-blind friendly palette: `#1b9e77`, `#d95f02`, `#7570b3`, `#e7298a`.

---

## 🔟 Reporting & Publication Preparation

Each milestone should produce:

| Output             | File                                | Description                          |
| :----------------- | :---------------------------------- | :----------------------------------- |
| Technical summary  | `docs/reports/<exp_name>_report.md` | Description + metrics + visuals      |
| Paper figures      | `figures/`                          | 600 DPI, vector format (.pdf/.eps)   |
| Statistical tables | `tables/`                           | CSV or LaTeX formatted               |
| Model card         | `docs/model_card.md`                | Final architecture + dataset ethics  |
| Appendix           | `appendix/`                         | Additional ablations and XAI visuals |

Before camera-ready submission:

* Freeze repository (`main` branch snapshot)
* Tag version: `v1.0-camera-ready`
* Archive supplementary materials (PDF + code zip)

---

## 11️⃣ Ethics and Data Compliance

* All datasets are public (IEMOCAP, RAVDESS, MELD) and used under academic license.
* No personal or private data shall be uploaded.
* Audio samples must remain anonymized if shown in demos.
* Attribution and dataset licenses included in `docs/dataset_license.md`.

---

## 12️⃣ Research Documentation Standards

| Type             | Format                         | Location                   |
| :--------------- | :----------------------------- | :------------------------- |
| Research Notes   | Markdown (`notes.md`)          | `/experiments/runs/`       |
| Meeting Minutes  | Google Docs / Notion           | Shared drive               |
| Weekly Summary   | Markdown (`weekly_summary.md`) | `/docs/meetings/`          |
| Figures / Tables | 300+ DPI / vector PDF          | `/figures/`                |
| Draft Paper      | Overleaf (`SLX02_paper/`)      | Linked to repo commit hash |

---

## 13️⃣ Version Control for Papers

Each submission stage corresponds to a **Git tag**:

| Tag                 | Description                           |
| :------------------ | :------------------------------------ |
| `v0.1-baseline`     | Conformer-only results                |
| `v0.3-fusion`       | Conformer–GAT ablations               |
| `v0.5-xai`          | Explainability experiments            |
| `v0.8-draft`        | Draft submitted to Overleaf           |
| `v1.0-camera-ready` | Final version submitted to IJCAI 2026 |

---

## 14️⃣ Quality Assurance Checklist

Before claiming a result as *final*:

* [x] Code reproduces results with ≤2% variance
* [x] Config + random seed logged
* [x] Metrics validated via sklearn
* [x] Figures exported at ≥300 DPI
* [x] Ethics compliance confirmed
* [x] Paper draft updated with latest numbers

---

## 15️⃣ Contacts

**Research Lead:**
Hoang Pham Gia Bao ([BAOHOANG2005](https://github.com/BAOHOANG2005))

**Modeling & Writing:**
Le Nguyen Gia Hung ([hei1sme](https://github.com/hei1sme))

**Evaluation & Experiments:**
Vo Tan Phat ([FappLord](https://github.com/FappLord))

**Evaluation & Experiments:**
Le Nguyen Thien Danh([TBA](TBA))

**Mentor:**
Ms. Thu Le – FPT University

---

### 📄 Document Info

* **File:** `docs/RESEARCH_PROTOCOL.md`
* **Maintainer:** SpeedyLabX Core Research Team
* **Last Updated:** October 2025
* **Applies To:** All SLX02 researchers
* **License:** CC BY-NC 4.0