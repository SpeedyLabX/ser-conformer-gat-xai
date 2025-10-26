# Enhancing Multimodal Speech Emotion Recognition via a Conformer–GAT Fusion Architecture with Explainable AI

<!-- Badges -->
<p align="left">
  <img src="https://img.shields.io/badge/Status-Under%20Research-orange" alt="Project Status">
  <img src="https://img.shields.io/badge/Conference-Target%3A%20IJCAI%202026-brightgreen" alt="Conference Target">
  <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue" alt="License">
  <a href="https://github.com/SpeedyLabX/ser-conformer-gat-xai/issues"><img src="https://img.shields.io/github/issues/SpeedyLabX/ser-conformer-gat-xai" alt="GitHub issues"></a>
</p>

---

### Abstract

This repository accompanies the research project *“Enhancing Multimodal Speech Emotion Recognition via a Conformer–GAT Fusion Architecture with Explainable AI (XAI)”*.  
The study proposes a **hybrid multimodal framework** integrating **acoustic** and **linguistic** representations through a **Compact Conformer** encoder for temporal modeling and a **Graph Attention Network (GAT)** for inter-modal reasoning.  
Explainability is achieved via **attention visualization**, **SHAP**, and **Integrated Gradients**, allowing interpretability of model decisions.

Our goal is to contribute an interpretable and reproducible multimodal SER architecture toward submission to **ACIIDS 2026**.

---

## 1 · Introduction

Emotion recognition from speech is inherently multimodal, involving both **prosodic** and **semantic** cues.  
Unimodal or early-fusion approaches often fail to model the contextual dependencies between modalities.  
This work introduces a **Conformer–GAT hybrid** that jointly captures temporal patterns and relational dependencies, thereby improving emotion discrimination and interpretability.

**Main contributions**
- A **Compact Conformer** audio encoder built upon **WavLM-Base** embeddings.  
- A **BERT/RoBERTa** text encoder for semantic representation.  
- A **dual-channel GAT fusion** mechanism modeling intra- and inter-modal relations.  
- An integrated **XAI suite** combining attention-based visualization, SHAP, and IG for transparent evaluation.

---

## 2 · System Overview

**Feature Encoding**  
- Audio → WavLM + Compact Conformer (2–4 layers, 512 dim)  
- Text → BERT/RoBERTa contextual embeddings (512 dim)

**Graph-Based Fusion**  
- Inter- and intra-modal GAT layers (multi-head, adaptive adjacency)  
- BiGRU + Pooling + Fully Connected Classifier

**Explainability Layer**  
- Attention heatmaps, SHAP, and Integrated Gradients  
- Quantitative attribution of modality contributions

---

## 3 · Datasets and Evaluation

| Dataset | Domain | Modality | Usage |
| :--- | :--- | :--- | :--- |
| **IEMOCAP** | Dyadic speech | Audio + Text | Primary benchmark |
| **RAVDESS** | Acted speech | Audio + Text | Cross-domain validation |
| **MELD** | Multi-party dialogue | Audio + Text | Generalization study |

**Metrics:** Weighted Accuracy (WA), Unweighted Accuracy (UA), Macro-F1, Confusion Matrix  
Configuration files for all experiments are provided under `/configs/`.

---

## 4 · Experimental Roadmap

| Phase | Description | Timeline | Status |
| :--- | :--- | :--- | :--- |
| Phase I | Dataset Preparation & EDA | Sep 2025 | ✅ Completed |
| Phase II | Model Architecture Design (Conformer–GAT) | Oct 2025 | 🟠 Ongoing |
| Phase III | Training, Ablation & XAI Evaluation | Nov–Dec 2025 | ⏳ Pending |
| Phase IV | Paper Writing & Submission (ACIIDS 2026) | Jan–Feb 2026 | ⏳ Planned |

---

## 5 · Repository Structure
```

ser-conformer-gat-xai/
├─ src/serxai/
│  ├─ data/          # Dataloaders & preprocessing
│  ├─ models/        # Conformer, GAT & fusion modules
│  ├─ xai/           # Explainability tools (SHAP, attention)
│  └─ utils/         # Metrics & reproducibility
├─ configs/          # YAML configuration files
├─ notebooks/        # EDA & ablation studies
├─ docs/             # Model cards & visualizations
└─ experiments/      # Logs and artifacts (ignored by Git)

```

---

## 6 · Citation

If you find this work useful, please cite:

```bibtex
@misc{SpeedyLabX2025SER,
  title        = {Enhancing Multimodal Speech Emotion Recognition via a Conformer–GAT Fusion Architecture with Explainable AI},
  author       = {Le Nguyen Gia Hung and Hoang Pham Gia Bao, Vo Tan Phat, Le Nguyen Thien Danh and Thu Le},
  organization = {SpeedyLabX Research Group, FPT University},
  year         = {2025},
  note         = {Work in progress, Target: ACIIDS 2026}
}
```

---

## 7 · Authors and Acknowledgement

**SpeedyLabX Research Group — FPT University**

| Member                    | Role                                 | GitHub                                           |
| :-------------------------| :----------------------------------- | :----------------------------------------------- |
| **Hoang Pham Gia Bao**    | Audio Modeling · Fusion Design · XAI | [@BAOHOANG2005](https://github.com/BAOHOANG2005) |
| **Le Nguyen Gia Hung**    | Text Modeling · Research Writing     | [@hei1sme](https://github.com/hei1sme)           |
| **Vo Tan Phat**           | Evaluation · Benchmarking            | [@FappLord](https://github.com/FappLord)         |
| **Le Nguyen Thien Danh**  | Evaluation · Benchmarking            | [@TBA](https://github.com/TBA)                   |


**Academic Mentor:** *Ms. Thu Le*
We sincerely thank our mentor for continuous guidance and insightful feedback throughout this study.

---

## 8 · Keywords

`Speech Emotion Recognition` · `Multimodal Learning` · `Conformer` · `Graph Attention Network (GAT)` · `Explainable AI (XAI)` · `IEMOCAP`

---

## 9 · License

This project is licensed under the
**Creative Commons Attribution – NonCommercial 4.0 International (CC BY-NC 4.0)** License.

You are free to share and adapt this work for **research and educational purposes**, provided that appropriate credit is given and the use is **non-commercial**.
See the [LICENSE](LICENSE) file for the full legal text.

<p align="center">
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://licensebuttons.net/l/by-nc/4.0/88x31.png" alt="CC BY-NC 4.0">
  </a>
</p>