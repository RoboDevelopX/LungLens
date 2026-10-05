# 🫁 LungLens

**See what the AI sees.** Explainable chest X-ray screening for earlier pneumonia detection.

LungLens is a web app that looks at a chest X-ray and, in a few seconds, says how
likely it is that the patient has pneumonia. It also draws a **Grad-CAM heatmap**
over the X-ray showing which regions drove that decision, so a doctor can check
the AI's reasoning instead of trusting a black box.

Built by Ribhu ([@RoboDevelopX](https://github.com/RoboDevelopX)) for the
**UnivaBio hackathon** (theme: Artificial Intelligence for Human Health).

> ⚠️ **Screening aid, not a diagnosis.** LungLens is a student research prototype.
> It must not be used to make medical decisions. Always consult a qualified doctor.

![LungLens app screenshot](docs/app_screenshot.png)

## Why it matters

Pneumonia is the biggest infectious killer of young children: the WHO reports it
caused 14% of all deaths of children under five in 2019 (about 740,000 children).
A chest X-ray can confirm it, but many hospitals and clinics don't have a
radiologist available to read X-rays quickly. AI can read an X-ray in seconds, but
most AI tools give an answer without a reason, and doctors are right not to trust
what they can't check. LungLens shows its reasoning.

## Features

- **Upload an X-ray** (JPG or PNG) or try one of six built-in samples.
- **Instant result:** the probability of pneumonia and a clear verdict.
- **Grad-CAM heatmap** next to the original X-ray, with an adjustable strength slider.
- **Built-in guide** to reading the heatmap and spotting when the model may be using a shortcut.
- **Sanity check** that warns you if the upload looks like a colour photo rather than an X-ray.
- **Runs fully offline** after installation; X-rays never leave your computer.

## Results

Measured on the 624 test X-rays, which were never used for training or tuning:

| Metric | Value |
|---|---|
| Accuracy | 90.1% |
| Sensitivity (pneumonia cases caught) | 98.5% (384 of 390) |
| Specificity (healthy X-rays correctly cleared) | 76.1% (178 of 234) |
| AUC | 0.955 |

LungLens is tuned to rarely miss a sick child; the trade-off is some false alarms
on healthy X-rays, which a doctor then reviews.

| Confusion matrix | ROC curve |
|---|---|
| ![Confusion matrix](results/confusion_matrix.png) | ![ROC curve](results/roc_curve.png) |

**Grad-CAM examples** (top: original X-ray, bottom: what LungLens looked at):

![Grad-CAM examples](results/gradcam_examples.png)

## How it works

1. **Data:** 5,856 chest X-rays of children aged 1 to 5, labelled NORMAL or
   PNEUMONIA by physicians (Kermany et al., *Cell* 2018, CC BY 4.0).
2. **Model:** a DenseNet121 convolutional neural network, pretrained by the
   [TorchXRayVision](https://github.com/mlmed/torchxrayvision) project on
   hundreds of thousands of adult chest X-rays, with a new final layer that
   outputs one number: the probability of pneumonia (transfer learning).
3. **Training:** only the last dense block and the new layer are fine-tuned,
   for 5 epochs on an ordinary CPU. Validation images are held out by patient,
   so the model is never tested on a child it has already seen. The loss is
   weighted for the 3:1 class imbalance, and the alert threshold is chosen on
   validation data to balance sensitivity and specificity.
4. **Explainability:** Grad-CAM uses the gradients flowing into the network's
   last 7×7 feature maps to score how much each region of the X-ray mattered.
5. **App:** a Streamlit page that shows the verdict, probability and heatmap side by side.

## Run it yourself

You need Python 3.10 to 3.13 (3.12 recommended). New to Python? The
step-by-step guide for Windows and Mac is in [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md).

**Windows (Command Prompt), inside the project folder:**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Mac / Linux (Terminal), inside the project folder:**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. The trained model is already included in
`models/lunglens.pt`, so there's nothing to train.

**To reproduce everything from scratch** (optional):
```
python prepare_data.py   # downloads the dataset, about 1.2 GB
python train.py          # about 85 minutes on a 4-core CPU
python evaluate.py       # test-set metrics and charts into results/
```

## Project files

| File | What it does |
|---|---|
| `prepare_data.py` | Step 1. Downloads the dataset and unpacks it into image folders |
| `dataset.py` | Loads images and makes the patient-level train/validation split |
| `network.py` | The neural network and the image preprocessing |
| `train.py` | Step 2. Fine-tunes the network and saves `models/lunglens.pt` |
| `evaluate.py` | Step 3. Measures the model on the test set, saves charts to `results/` |
| `gradcam.py` | Builds the Grad-CAM heatmap and the coloured overlay |
| `app.py` | Step 4. The interactive web app |
| `models/` | The trained model (28 MB) |
| `samples/` | Six test-set X-rays for trying the app |
| `results/` | Test metrics, charts and training log |
| `docs/` | Run guide, one-page description, code PDF, demo video script, code walkthrough |

## Built with

Python · PyTorch · TorchXRayVision · scikit-learn · Streamlit · Matplotlib

## Limitations

- Trained on young children from a single hospital in China; performance on
  adults or on X-rays from other hospitals and machines is unknown.
- Only two classes (normal and pneumonia). It does not detect other conditions.
- Grad-CAM heatmaps are coarse (7×7 grid) and show where the model looked,
  not a medical outline of the infection.
- Not a medical device. It would need testing on data from other hospitals and
  regulatory approval before any clinical use.

## Credits

- Dataset: Kermany, D. S. et al. "Identifying Medical Diagnoses and Treatable
  Diseases by Image-Based Deep Learning." *Cell* 172(5), 2018.
- Pretrained weights: Cohen, J. P. et al. "TorchXRayVision: A library of chest
  X-ray datasets and models." MIDL 2022.
- Grad-CAM: Selvaraju, R. R. et al. "Grad-CAM: Visual Explanations from Deep
  Networks via Gradient-based Localization." ICCV 2017.
