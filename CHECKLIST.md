# LungLens hackathon checklist

Status key: ✅ done and verified by running · 🟡 written, not yet verified · ⏳ not started · ⛔ blocked

Last updated: 2026-10-13, end of phase (b).

## Mandatory hackathon deliverables

| # | Requirement | Status | Where / evidence |
|---|---|---|---|
| 1 | Working prototype with user interaction (web app) | ⏳ | `app.py`, `pages/` (phase d) |
| 2 | Demo video, 2-3 min (Ribhu records; we write script + shot list) | ⏳ | `docs/demo_script.md` (phase f) |
| 3 | One-page project description PDF | ⏳ | `scripts/make_one_pager.py` -> `LungLens_OnePager.pdf` (phase f) |
| 4 | GitHub repository + "Code PDF" of all code | ⏳ | local git repo now; `scripts/make_code_pdf.py` (phase f); push needs a GitHub repo |
| 5 | `docs/WALKTHROUGH.md` (architecture, each file, judge Q&A) | ⏳ | phase f |

## Build phases

| Phase | Scope | Status | Evidence |
|---|---|---|---|
| a | Scaffolding, config, download script, patient-grouped split, data pipeline | ✅ | commit "Phase a"; `src/data.py` verified by the phase (b) training run |
| b | Model, training, evaluation, Colab notebook | ✅ trained + evaluated · 🟡 Colab notebook written, not run (no Colab here) | DenseNet121, 2 head + 9 fine-tune epochs on 4 CPU cores (106 min, early stop), best val AUROC 0.998; test results in `reports/metrics.json` |
| c | Grad-CAM, calibration (ECE before/after), OOD guard | ⏳ | |
| d | Streamlit app (5 pages), < 3 s CPU inference | ⏳ | |
| e | pytest suite | ⏳ | |
| f | README, WALKTHROUGH, demo script, PDFs | ⏳ | |

## Data and training rules from the brief

| Rule | Status | Evidence |
|---|---|---|
| Never commit data | ✅ | `.gitignore` excludes `data/`, `*.zip`, weights |
| `scripts/download_data.py` via Kaggle API | ✅ | ran end to end: 5,856 images extracted |
| Re-split train ~80/20 stratified, ignore 16-image val | ✅ | `src/split.py`, `reports/splits.csv`; train 4,168 / val 1,048 (both 74.3% pneumonia) |
| Group by patient (no patient in both train and val) | ✅ | 0 shared patients (printed by `python -m src.split`) |
| Test set untouched until final evaluation | ✅ | scored once in phase (b); threshold rule fixed on validation before that; nothing tuned on test |
| Class imbalance handled | ✅ | class-weighted cross-entropy, weights [1.944, 0.673] (printed by `src/train.py`) |
| X-ray-appropriate augmentation, no horizontal flip | ✅ | `src/data.py::build_transforms` (rotation ±7°, mild crop, brightness/contrast), used in training |
| Metrics on held-out test set: accuracy, sensitivity, specificity, precision, F1, AUROC + bootstrap 95% CIs | ✅ | `reports/metrics.json` (2,000 resamples) |
| Figures generated only from metrics.json | ✅ | `python -m src.evaluate --figures-only` redraws all of `reports/figures/` |
| Sensitivity-first threshold chosen on validation | ✅ | rule: highest threshold with val sensitivity >= 0.95, capped at 0.5 (cap approved by Ribhu 2026-10-13); the rule alone gave 0.996 |

## Results so far (copied from `reports/metrics.json`, phase b, uncalibrated)

Test set (624 X-rays) at the operating threshold 0.50: sensitivity 0.997 (95% CI 0.992-1.000), specificity 0.402
(0.339-0.465), AUROC 0.950 (0.929-0.969); 1 missed pneumonia, 140 false alarms on 234 normal X-rays. Validation
at the same threshold: sensitivity 1.000, specificity 0.863. The drop from validation to test is the main
honest finding so far; it suggests the official test folder differs from the training folder in ways our
validation split (drawn from the training folder) cannot show.
