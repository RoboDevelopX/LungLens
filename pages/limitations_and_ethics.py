"""Limitations & Ethics: the LungLens model card."""

import streamlit as st

from src import ui
from src.model import DISPLAY_NAMES

st.title("Limitations & ethics (model card)")
ui.disclaimer()

metrics = ui.get_metrics()
ood = ui.get_report("ood_eval.json")
facts = ui.get_split_facts()

st.header("Intended use")
st.markdown(
    """
- **What it is:** an educational prototype that shows how an AI screening aid for pneumonia on chest X-rays
  could work, and how its decisions can be explained and sanity-checked.
- **Who it is for:** students, hackathon judges and anyone learning about trustworthy medical AI.
- **Not for:** diagnosis, treatment decisions, triage of real patients, or any clinical use. It is not a
  medical device and has not been reviewed by any regulator.
"""
)

st.header("Model")
if metrics:
    ck = metrics["checkpoint"]
    st.markdown(
        f"- {DISPLAY_NAMES.get(ck['backbone'], ck['backbone'])} pretrained on ImageNet, fine-tuned for 2 classes (Normal, Pneumonia); "
        f"model fingerprint `{ck['fingerprint']}`.\n"
        f"- Probabilities calibrated with temperature scaling; operating threshold chosen on validation data for "
        f">= {metrics['operating_point']['target_sensitivity']:.0%} sensitivity and never above 50%; scores near the "
        "threshold are reported "
        f"as *Inconclusive*.\n- Explanations: Grad-CAM (own implementation, checked on the Sanity Check page)."
    )
else:
    st.info("No evaluated model yet: the details appear here after `python -m src.evaluate`.")

st.header("Training and evaluation data")
st.markdown(
    """
- **Source:** Kermany, Zhang & Goldbaum (2018), *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray
  Images for Classification*, Mendeley Data (CC BY 4.0), via the Kaggle copy "Chest X-Ray Images (Pneumonia)".
- **Population:** anterior-posterior chest X-rays of **pediatric patients aged one to five** from a **single
  hospital**, Guangzhou Women and Children's Medical Center, China.
- **Labels:** image-level Normal or Pneumonia, graded by physicians. No bounding boxes, no age, sex or other
  patient information.
- **Splits:** the official training folder was re-split 80/20 **by patient** into train and validation; the
  official test folder was used **once**, for the final numbers. The official 16-image validation folder is too
  small to be useful and is not used.
"""
)
if facts:
    n = facts["n"]
    st.caption(f"From reports/splits.csv: {n['train']:,} training, {n['val']:,} validation and {n['test']:,} test X-rays.")
if metrics:
    op = metrics["test"]["at_operating_threshold"]
    m, ci = op["metrics"], op["ci95"]
    st.markdown(
        f"**Test-set results** (n = {metrics['data']['n_test']}): sensitivity {m['sensitivity']:.3f} "
        f"[{ci['sensitivity'][0]:.3f}, {ci['sensitivity'][1]:.3f}], specificity {m['specificity']:.3f} "
        f"[{ci['specificity'][0]:.3f}, {ci['specificity'][1]:.3f}], AUROC {metrics['test']['auroc']:.3f} "
        f"[{ci['auroc'][0]:.3f}, {ci['auroc'][1]:.3f}]. These describe **this** dataset only."
    )

if facts:
    identity = (
        f"{facts['shared_ids']} patient IDs in the file names appear in both the official train and test folders "
        f"({facts['test_images_with_shared_id']} of the {facts['n']['test']} test images). "
        + ("In every case the infection type differs (bacteria vs virus), so these are probably different children "
           "numbered separately, but we cannot prove it."
           if facts["shared_ids_same_kind"] == 0 else
           f"{facts['shared_ids_same_kind']} of them share an infection type, so some test images may come from "
           "children the model was trained on.")
    )
    prevalence = (f"Probabilities were calibrated on validation data where {facts['val_pneumonia_share']:.0%} of images "
                  "show pneumonia.")
    if metrics and "calibration" in metrics:
        cal = metrics["calibration"]
        prevalence += (f" Even on this dataset's official test folder ({facts['test_pneumonia_share']:.0%} pneumonia) the "
                       f"calibration error after scaling is {cal['test_ece_after']:.3f}, against "
                       f"{cal['val_ece_after']:.3f} on validation, so read the percentages as a rough guide.")
else:
    identity = "File names suggest some patient IDs appear in both the official train and test folders."
    prevalence = "Probabilities were calibrated on validation data where most images show pneumonia."

st.header("Known limitations")
st.markdown(
    f"""
1. **Single hospital, single country.** Every image comes from one hospital's machines and protocols. Models
   trained this way often pick up site-specific cues and **lose accuracy on other hospitals' images**.
2. **Children only.** Adult chest X-rays look different (body size, heart shape, posture, common diseases).
   **Expect a performance drop on adults.**
3. **No external validation.** All numbers come from the same source as the training data. We have **not**
   tested LungLens on any independent dataset, so real-world performance is unknown.
4. **Label noise.** Labels come from physicians' grading of each image and, like most medical labels, contain
   some errors and borderline cases. We did not re-check or re-label any image, so some "mistakes" in our
   results may be label errors, and some "correct" answers may be wrong.
5. **Two classes only.** LungLens cannot recognise tuberculosis, COVID-19, effusions, tumours, fractures or any
   other finding. "Normal" means only "no pneumonia pattern found".
6. **Patient identity.** {identity}
7. **Calibration depends on prevalence.** {prevalence} Where pneumonia is rarer, as in most real screening
   settings, the true chance behind the same score is lower.
8. **Grad-CAM is coarse and partial.** It shows where evidence was found on a 7 x 7 grid, not why, and it can
   highlight markers or borders (see the Sanity Check page).
"""
)
if ood:
    lines = []
    for name, r in ood.items():
        lines.append(f"   - {name}: {r['n']} images, {100 * r['flagged_rate']:.1f}% flagged "
                     f"({r['reject']} refused, {r['warn']} warned)")
    st.markdown("9. **The input check is not perfect.** Measured on real images:\n" + "\n".join(lines))

st.header("Bias and fairness")
st.markdown(
    """
- The dataset has **no sex, ethnicity or socio-economic information**, so we **cannot measure** whether
  LungLens works equally well for different groups. Absence of evidence is not evidence of fairness.
- All patients come from one city in one country; populations with different disease patterns, nutrition or
  access to care are not represented.
- Images from cheaper or older X-ray machines, or portable bedside X-rays, may look different and may be
  handled worse, which could disadvantage low-resource settings, exactly where screening aids are most needed.
- **Automation bias:** people tend to over-trust confident software. LungLens counters this with calibrated
  probabilities, an Inconclusive zone, visible explanations and constant reminders that a clinician decides.
"""
)

st.header("Privacy")
st.markdown(
    "Uploaded images are processed in memory for the current session and are not saved to disk or sent anywhere "
    "by LungLens. Do not upload images that identify a person (names, dates, IDs burned into the image)."
)
