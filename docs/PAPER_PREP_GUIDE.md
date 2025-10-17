# 🧾 **Paper Preparation and Submission Guide**

**Project:** *SLX02 – Enhancing Multimodal Speech Emotion Recognition via a Conformer–GAT Fusion Architecture with XAI*
**Organization:** *SpeedyLabX Research Group, FPT University*
**License:** *CC BY-NC 4.0*

---

## 1️⃣ Purpose

This guide provides a **step-by-step process** for preparing, formatting, and submitting the SLX02 research paper to its target venue (IJCAI 2026 or equivalent).
It ensures that all outputs—paper, figures, tables, and supplementary material—meet **academic and publisher standards**.

---

## 2️⃣ File and Directory Structure

```
paper/
├─ main.tex               # core paper
├─ sections/
│  ├─ 01_introduction.tex
│  ├─ 02_related_work.tex
│  ├─ 03_methodology.tex
│  ├─ 04_experiments.tex
│  ├─ 05_results_discussion.tex
│  └─ 06_conclusion.tex
├─ figures/
│  ├─ architecture.pdf
│  ├─ confusion_matrix.pdf
│  ├─ shap_summary.pdf
│  └─ attention_heatmap.pdf
├─ tables/
│  ├─ performance.tex
│  ├─ ablation.tex
│  └─ dataset_stats.tex
├─ references.bib
└─ appendix/
   ├─ ablation_details.pdf
   └─ xai_visuals.pdf
```

---

## 3️⃣ Paper Formatting

### For **IJCAI 2026**

* Template: [https://ijcai-24.org/formatting-guidelines/](https://ijcai-24.org/formatting-guidelines/)
* LaTeX class: `ijcai26.sty` (once available)
* Page limit: **7 pages main text + 1 page references**
* Font: **Times New Roman**, 10 pt
* Column: **two-column format**
* Spacing: do *not* modify margins or font size

### For **AAAI 2026**

* Template: `aaai26.sty`
* Max 8 pages text + 1 page references

Always validate using:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 4️⃣ Writing Checklist

| Section              | Key Questions                                   | Status |
| :------------------- | :---------------------------------------------- | :----- |
| Abstract             | Clear research gap and contribution             | ☐      |
| Introduction         | Motivation → Problem → Approach → Contributions | ☐      |
| Related Work         | Includes 2023–2025 literature                   | ☐      |
| Methodology          | Equations + architecture figure                 | ☐      |
| Experiments          | Dataset split, baselines, hyper-params          | ☐      |
| Results & Discussion | Tables + error analysis                         | ☐      |
| XAI Analysis         | Attention + SHAP/IG interpretation              | ☐      |
| Conclusion           | Key takeaway + future work                      | ☐      |
| References           | All citations complete & formatted              | ☐      |

---

## 5️⃣ Figures and Tables

### Figures

* Resolution ≥ 300 DPI (vector PDF/EPS preferred)
* Use consistent fonts: *Times New Roman*, 9 pt
* Color-blind-friendly palette: `#1b9e77`, `#d95f02`, `#7570b3`, `#e7298a`
* Caption below figure, format:

  ```
  Figure 1: Overview of the proposed Conformer–GAT fusion framework for multimodal SER.
  ```

### Tables

* Avoid vertical lines
* Center-align headers
* Caption above table:

  ```
  Table 2: Performance comparison on the IEMOCAP dataset.
  ```

### Example (LaTeX)

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{lccc}
\toprule
Model & WA & UA & F1 \\
\midrule
Conformer (Baseline) & 0.821 & 0.798 & 0.803 \\
Conformer + GAT (Ours) & \textbf{0.871} & \textbf{0.845} & \textbf{0.854} \\
\bottomrule
\end{tabular}
\caption{Performance comparison on IEMOCAP.}
\label{tab:main}
\end{table}
```

---

## 6️⃣ Referencing and BibTeX Style

### Reference format

Use **BibTeX** with IEEE/AAAI/IJCAI style:

```latex
\bibliographystyle{named}
\bibliography{references}
```

Example entry:

```bibtex
@article{khan2024mser,
  title   = {MSER: Multimodal Speech Emotion Recognition using Cross-Attention with Deep Fusion},
  author  = {Mustaqeem Khan and Wail Gueaieb and Abdulmotaleb El Saddik and Soonil Kwon},
  journal = {Expert Systems with Applications},
  year    = {2024},
  volume  = {245},
  pages   = {122946}
}
```

Cite using `\citep{}` or `\cite{}` depending on template.

---

## 7️⃣ Supplementary Materials

| Type              | Content                     | Format                               | Location                       |
| :---------------- | :-------------------------- | :----------------------------------- | :----------------------------- |
| Appendix A        | Ablation results            | PDF (≤ 3 pages)                      | appendix/                      |
| Appendix B        | XAI visualizations          | PDF images + captions                | appendix/                      |
| Appendix C        | Hyperparameter settings     | CSV or table TeX                     | appendix/                      |
| Code Zip          | Minimal reproducible subset | .zip (< 30 MB)                       | upload to OpenReview or Zenodo |
| Dataset Statement | Source + license info       | markdown → `docs/dataset_license.md` |                                |

---

## 8️⃣ Experiment → Paper Traceability

Every result in the paper must map to a logged experiment in `experiments/runs/`.

| Table/Figure                | Experiment ID                       | File Reference                      |
| :-------------------------- | :---------------------------------- | :---------------------------------- |
| Table 2 (Main Results)      | `2025-10-21_03_gat_layers_ablation` | `metrics.json`                      |
| Figure 3 (Confusion Matrix) | `EXP02`                             | `plots/confusion_matrix.png`        |
| Figure 5 (SHAP Summary)     | `EXP04`                             | `docs/xai_gallery/shap_summary.png` |

This traceability guarantees full reproducibility during peer review.

---

## 9️⃣ Camera-Ready Checklist

✅ All text within page limit
✅ Figures/tables vectorized
✅ Ablation & statistical results verified
✅ Proofread by ≥ 2 members
✅ Spell-checked (US English)
✅ Metadata updated (authors, affiliations)
✅ License (CC BY-NC 4.0) noted in repo
✅ BibTeX compiles without errors
✅ PDF generated without overfull boxes
✅ Overleaf linked to final commit hash
✅ Tag version → `v1.0-camera-ready`

---

## 🔟 Post-Submission Archiving

After acceptance:

1. Create `main` snapshot with tag:

   ```bash
   git tag -a v1.0-camera-ready -m "IJCAI 2026 camera-ready version"
   git push origin v1.0-camera-ready
   ```
2. Archive materials:

   * `paper/`
   * `figures/`, `tables/`, `appendix/`
   * `README.md`, `LICENSE`, and `CITATION.cff`
3. Upload to:

   * **Zenodo** (for DOI)
   * **OpenReview** (if required)
   * **Hugging Face Repo** (for model card)

---

## 11️⃣ Authorship & Acknowledgment Guidelines

| Contributor        | Authorship Status   | Contribution Domain              |
| :----------------- | :------------------ | :------------------------------- |
| Le Nguyen Gia Hung | First Author / Lead | Model Design · XAI · Writing     |
| Hoang Pham Gia Bao | Co-Author           | Text Model · Ablation · Drafting |
| Vo Tan Phat        | Co-Author           | Evaluation · Visualization       |
| Ms. Minh Thư       | Mentor              | Supervision · Final Review       |

*Authorship follows the Vancouver (2019) and ACM ethical standards.*

---

## 12️⃣ Citation and Repository Metadata

Provide an official citation file `CITATION.cff` at repo root:

```yaml
cff-version: 1.2.0
message: "If you use this code or dataset, please cite the following work."
title: "Enhancing Multimodal Speech Emotion Recognition via a Conformer–GAT Fusion Architecture with Explainable AI"
authors:
  - family-names: "Le Nguyen"
    given-names: "Gia Hung"
  - family-names: "Hoang Pham"
    given-names: "Gia Bao"
  - family-names: "Vo Tan"
    given-names: "Phat"
date-released: 2025-10-17
license: "CC-BY-NC-4.0"
repository-code: "https://github.com/SpeedyLabX/ser-conformer-gat-xai"
version: "1.0-camera-ready"
```

---

## 13️⃣ Submission Package Example

```
submission/
├─ paper.pdf
├─ supplementary.pdf
├─ code.zip
├─ README.txt
├─ LICENSE.txt
└─ CITATION.cff
```

ZIP size ≤ 50 MB, self-contained, builds on clean Overleaf commit.

---

## 14️⃣ Post-Acceptance Maintenance

After acceptance:

* Update README badges → “Accepted at IJCAI 2026”
* Release final code & model weights (under CC BY-NC 4.0)
* Announce DOI & paper link
* Update SpeedyLabX site / Hugging Face page

---

## 15️⃣ Contact

**Project Lead:** Hoang Pham Gia Bao ([BAOHOANG2005](https://github.com/BAOHOANG2005))
**Mentor:** Ms. Thu Le — FPT University
**Organization:** [SpeedyLabX Research Group](https://github.com/SpeedyLabX)

---

### 📄 Document Info

* **File:** `docs/PAPER_PREP_GUIDE.md`
* **Maintainer:** SpeedyLabX Core Writing Team
* **Last Updated:** October 2025
* **Applies To:** SLX02 Publication Cycle
* **License:** CC BY-NC 4.0