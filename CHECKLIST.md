# LungLens hackathon checklist

Status key: ✅ done and verified by running · 🟡 written, not yet verified · ⏳ not started · ⛔ blocked

Last updated: 2026-10-13, end of phase (a).

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
| a | Scaffolding, config, download script, patient-grouped split, data pipeline | ✅ split + download run · 🟡 `src/data.py` not run (PyTorch cannot be installed in this environment yet) | commit "Phase a" |
| b | Model, training, evaluation, Colab notebook | ⛔ needs PyTorch | |
| c | Grad-CAM, calibration (ECE before/after), OOD guard | ⏳ | |
| d | Streamlit app (5 pages), < 3 s CPU inference | ⏳ | |
| e | pytest suite | ⏳ | |
| f | README, WALKTHROUGH, demo script, PDFs | ⏳ | |

## Data and training rules from the brief

| Rule | Status | Evidence |
|---|---|---|
| Never commit data | ✅ | `.gitignore` excludes `data/`, `*.zip`, weights |
| `scripts/download_data.py` via Kaggle API | ✅ | ran end to end: 5,856 images extracted |
| Re-split train ~80/20 stratified, ignore 16-image val | ✅ | `src/split.py`, `reports/splits.csv`; train 4,171 / val 1,045 (both 74.3% pneumonia) |
| Group by patient (no patient in both train and val) | ✅ | 0 shared patients (printed by `python -m src.split`) |
| Test set untouched until final evaluation | ✅ so far | only listed in splits.csv |
| Class imbalance handled | 🟡 | inverse-frequency weights in `src/data.py::class_weights` |
| X-ray-appropriate augmentation, no horizontal flip | 🟡 | `src/data.py::build_transforms` |
