[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17373616.svg)](https://doi.org/10.5281/zenodo.17373616)
[![Build Status](https://github.com/Vabi-tech/open-set-lid-prototype/actions/workflows/lint.yml/badge.svg)](https://github.com/Vabi-tech/open-set-lid-prototype/actions)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/Vabi-tech/open-set-lid-prototype/blob/main/colab.ipynb)


# Open-Set Detection Prototype — Multilingual Language Identification (Text)

This repo is a **minimal, reproducible prototype** for **open-set detection** in multilingual language identification (LID).
It trains a text classifier on a subset of languages (SEEN), then evaluates on **unseen languages (OOD)** and computes **confidence-based** and **feature-based** detection scores.

##  Methods Included
- **Model**: `distilbert-base-multilingual-cased` (Hugging Face)
- **Dataset**: Tatoeba (or FLORES-200) via `datasets` (switchable)
- **Scores**:
  - **Max Softmax Probability (MSP)**
  - **Energy Score** (log-sum-exp of logits)
  - **Mahalanobis Distance** (on penultimate features)
- **Metrics & Plots**:
  - Precision–Recall curves for OOD detection
  - Calibration: Reliability diagram + ECE

> Note: You’ll need internet access on first run to download the model/dataset.

##  Quickstart

```bash
# 1) Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train a multilingual LID classifier on SEEN languages
python src/train_classifier.py \  --dataset tatoeba \  --seen_langs en,es,de,fr,it \  --output_dir models/lid-seen

# 4) Extract logits and features for SEEN test and UNSEEN test languages
python src/extract_logits.py \  --dataset tatoeba \  --model_dir models/lid-seen \  --seen_langs en,es,de,fr,it \  --unseen_langs pt,ru,sv \  --out_path outputs/preds.jsonl

# 5) Compute OOD scores + plots (PR, calibration)
python scripts/plot_ood_and_calibration.py \  --preds outputs/preds.jsonl \  --out_dir outputs
```

##  Switching datasets
- For FLORES-200, pass `--dataset flores`. See `src/datasets.py` for details.

##  Repo layout
```
src/
  datasets.py            # dataset loaders (Tatoeba, FLORES)
  train_classifier.py    # fine-tune mDistilBERT for LID
  extract_logits.py      # get logits + features for SEEN/UNSEEN
  ood_scoring.py         # MSP, Energy, Mahalanobis
  calibration.py         # ECE + reliability bins
scripts/
  plot_ood_and_calibration.py  # produces PR curves + reliability diagrams
summary/
  one_page_summary.tex   # 1-page LaTeX summary template
  colab.ipynb # Interactive notebook (training → evaluation → visualization)
```

##  Citation-ready outputs
The `outputs/` folder will contain:
- `pr_energy.png`, `pr_msp.png`, `pr_mahalanobis.png`
- `reliability.png` — calibration plot
- `metrics.json` — AP, ECE, etc.
- `summary.pdf` — (optional) compiled 1-page summary (fill `summary/one_page_summary.tex`)

## Continuous Integration
This repository is verified via GitHub Actions (`flake8` lint) — ensuring every commit passes automated quality checks.

## Colab Notebook
A ready-to-run notebook (`colab.ipynb`) reproduces the training, extraction, and reliability plots in under 10 minutes on GPU.
