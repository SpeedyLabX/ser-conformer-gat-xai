# 🤝 **Contributor & Workflow Guide**

**Project:** *SLX02 – Enhancing Multimodal Speech Emotion Recognition via a Conformer–GAT Fusion Architecture with XAI*
**Organization:** *SpeedyLabX Research Group, FPT University*
**License:** *CC BY-NC 4.0*

---

## 1️⃣ Purpose

This document defines the **collaboration workflow**, **branch strategy**, and **review policy** for all contributors of SLX02.
It ensures that all code, documentation, and experiments follow the same academic and technical standards required for conference publication (IJCAI/AAAI 2026 target).

---

## 2️⃣ Prerequisites

Before contributing:

* Complete setup in [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md).
* Ensure **pre-commit** hooks are installed:

  ```bash
  pre-commit install
  pre-commit run --all-files
  ```
* Configure your Git identity:

  ```bash
  git config user.name "Your Name"
  git config user.email "your@fpt.edu.vn"
  ```

---

## 3️⃣ Repository Workflow

### 🔹 Branch Model

We use a **trunk-based workflow** with controlled merges.

| Branch   | Purpose                                   | Rules                       |
| :------- | :---------------------------------------- | :-------------------------- |
| `main`   | Stable, camera-ready or published version | Protected (no direct push)  |
| `dev`    | Integration & active development          | Protected (PR + 1 review)   |
| `feat/*` | New feature or module                     | Merged → `dev`              |
| `fix/*`  | Bug fix or refactor                       | Merged → `dev`              |
| `docs/*` | Documentation or README updates           | Merged → `dev`              |
| `exp/*`  | Experimental runs or ablation variants    | Merged → `dev` (after test) |

**Example:**

```
feat/audio-conformer
fix/xai-heatmap
docs/training-guide
exp/gat-ablations
```

---

## 4️⃣ Commit Convention

We follow **Conventional Commits** to generate changelogs automatically.

**Format:**

```
<type>(scope): <description>
```

**Examples:**

```bash
feat(model): add GAT fusion module
fix(train): resolve loss backward error
docs: update developer guide
refactor(audio): restructure conformer encoder
chore: update pre-commit dependencies
```

**Allowed types:**

| Type       | Purpose                                  |
| :--------- | :--------------------------------------- |
| `feat`     | Add new feature/module                   |
| `fix`      | Bug fix or logic correction              |
| `docs`     | Documentation updates                    |
| `refactor` | Internal improvement (no feature change) |
| `test`     | Unit/integration test                    |
| `chore`    | CI, configs, metadata updates            |
| `exp`      | Temporary ablation or experiment         |

---

## 5️⃣ Pull Request (PR) Workflow

### 🧭 Step-by-step

1. **Sync latest changes**

   ```bash
   git checkout dev
   git pull origin dev
   ```

2. **Create your branch**

   ```bash
   git checkout -b feat/new-fusion-head
   ```

3. **Develop and commit**

   ```bash
   git add .
   git commit -m "feat(model): implement new fusion head"
   ```

4. **Push to remote**

   ```bash
   git push -u origin feat/new-fusion-head
   ```

5. **Open a Pull Request**

   * Go to the repo on GitHub → “Compare & Pull Request”
   * Base branch: `dev`
   * Fill in PR template:

     ```
     ## Summary
     - Implemented GAT-based multimodal fusion
     - Added focal loss config

     ## Checklist
     - [x] Tests added
     - [x] Docs updated
     - [x] Lint & CI pass
     ```

6. **Review process**

   * At least **1 reviewer (lead)** approval for `dev`
   * At least **2 reviewers (core-admin)** for `main`

7. **Merge rules**

   * Use **Squash and Merge** to keep clean history
   * Delete feature branch after merge

---

## 6️⃣ Review Standards

**Reviewers check for:**

| Aspect           | Expected                                            |
| :--------------- | :-------------------------------------------------- |
| Code quality     | Clean, modular, typed, documented                   |
| Reproducibility  | Random seeds set, deterministic mode if possible    |
| Config alignment | YAML files follow structure in `/configs/base.yaml` |
| Documentation    | Clear commit + meaningful PR title                  |
| Ethics           | No copyrighted dataset leaks or personal data       |

---

## 7️⃣ Testing and CI

GitHub Actions run on every PR to `dev` and `main`:

* Linting via **Ruff + Black**
* Unit tests (`pytest --cov=src`)
* Coverage upload (Codecov)
* Docs build (mkdocs)

You can test locally before pushing:

```bash
ruff check .
black --check .
pytest -q --disable-warnings
```

---

## 8️⃣ Data & Artifacts Handling

* Large files (>100 MB) must use **DVC** or **Git LFS**.
* Do **NOT** commit:

  * `.pt`, `.ckpt`, `.wav`, `.zip`, or raw data.
* Artifacts (checkpoints, logs) → `.gitignore` + tracked via DVC if necessary.

---

## 9️⃣ Documentation Standards

All public documentation (e.g. README, Developer Guide, paper notes) must:

* Follow **academic tone** (no emoji-heavy or casual phrasing)
* Include **version/date**
* Use Markdown headings consistently:

  ```
  ## 1 · Section Title
  ### 1.1 Subsection
  ```
* Include diagram captions if visual figures are added.

---

## 🔟 Authorship and Credit Policy

As a research project, authorship for publications and datasets follows:

* **Substantial technical contribution** → co-author
* **Partial or supportive work** → acknowledgment

Final authorship list is approved by **Lead Researcher (Le Nguyen Gia Hung)** and **Academic Mentor (Ms. Minh Thư)**.

---

## 11️⃣ Security and Compliance

* All contributors must enable **2FA on GitHub**.
* Access to private repositories (data or models) is restricted to internal team members.
* Follow **FPT University research ethics guidelines** for dataset usage.

---

## 12️⃣ Example PR Lifecycle

```
📦 feat/audio-frontend
 ├── Push commits → open PR
 ├── CI checks pass
 ├── Review by @BAOHOANG2005
 ├── Reviewer approves ✅
 └── Squash & merge → dev branch
```

Then periodically:

```
dev → main
```

for stable checkpoints and paper-version synchronization.

---

## 13️⃣ Common Commands Summary

| Task                  | Command                               |
| :-------------------- | :------------------------------------ |
| Sync with remote      | `git pull origin dev`                 |
| Create feature branch | `git checkout -b feat/<name>`         |
| Run linting           | `pre-commit run --all-files`          |
| Run tests             | `pytest`                              |
| Open PR               | via GitHub web or `gh pr create`      |
| Merge PR              | “Squash and Merge” button (lead only) |

---

## 14️⃣ Contact

For internal questions:

* **Project Lead:** [@BAOHOANG2005](https://github.com/BAOHOANG2005)
* **Mentor:** *Ms. Thu Le*
* **Org:** [SpeedyLabX](https://github.com/SpeedyLabX)

---

### 🧾 Document Info

* **File:** `CONTRIBUTING.md`
* **Maintainer:** SpeedyLabX Core Team
* **Last Updated:** October 2025
* **Applies to:** All SLX02 collaborators
* **License:** CC BY-NC 4.0