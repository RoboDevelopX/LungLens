# LungLens: See what the AI sees

UnivaBio student hackathon 2026 · Theme: Artificial Intelligence for Human Health

## The problem

Pneumonia is the single largest infectious cause of death in children worldwide (WHO). A chest X-ray is the standard way to confirm it, but reading X-rays takes trained radiologists, and many clinics have too few. AI can help, yet most AI screening tools are black boxes: they give a confident answer even for a blurry photo, and never say why.

## Our solution

LungLens is a web app that screens a chest X-ray for pneumonia and shows its work. Upload an image and get:

- a **Normal / Pneumonia / Inconclusive** result with a **calibrated** probability,
- a **Grad-CAM heatmap** showing which regions drove the decision,
- a **plain-language explanation** of what the result does and does not mean, plus a clinician view.

## Key features

- **Honest confidence:** temperature scaling fitted on validation data; the calibration error is reported before and after on validation and test, including where it stays high. The app never shows 0% or 100%.
- **Sensitivity-first:** the threshold is set on validation data to catch at least 95% of pneumonia cases and is never above 50%; misses are costlier than false alarms in screening.
- **Inconclusive zone:** scores near the threshold say "please consult a clinician" instead of guessing.
- **Input check:** photos, documents and blank images are refused, not misread; other grayscale medical images remain the hardest case.
- **Sanity-checked explanations:** a model-randomisation test checks that the heatmaps reflect what the model learned, and an edge-attention check flags heatmaps that sit on image borders and text markers.
- **Model card:** limitations, bias and ethics stated openly in the app.

## How it works

DenseNet121 pretrained on ImageNet, fine-tuned in two stages on {{N_TRAIN}} pediatric chest X-rays (patient-grouped 80/20 split, class-weighted loss, X-ray-safe augmentation without flips). Grad-CAM is implemented from scratch with PyTorch hooks and cross-checked against a reference library. All metrics are computed once on the untouched {{N_TEST}}-image test set, with bootstrap 95% confidence intervals.

## Results (held-out test set)

{{RESULTS}}

## Limitations

Trained on children aged 1-5 from a single hospital (Guangzhou, China); no external validation, so performance will likely drop on adults and other hospitals' images. Many false alarms on the official test folder. Labels contain some noise. Only detects pneumonia patterns. No demographic data, so fairness across groups cannot be measured. **Educational screening aid, not a diagnostic device.**

## Tech stack

Python · PyTorch · torchvision · scikit-learn · Streamlit · Altair · matplotlib · reportlab · pytest · Kaggle "Chest X-Ray Images (Pneumonia)" (Kermany et al., 2018, CC BY 4.0)

## Author

Ribhu · github.com/RoboDevelopX/LungLens
