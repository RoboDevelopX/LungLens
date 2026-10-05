"""
app.py - The LungLens web app (Step 4).

Run:  streamlit run app.py
Then open the link it prints (usually http://localhost:8501).

What the user does
------------------
1. Uploads a chest X-ray (or picks one of the sample images).
2. LungLens shows its screening result: the chance of pneumonia and
   whether that is above the alert threshold.
3. Next to the X-ray it shows a Grad-CAM heatmap of where the model
   "looked" when deciding, so a clinician can check the reasoning.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from gradcam import gradcam, overlay
from network import load_model, preprocess

st.set_page_config(page_title="LungLens", page_icon="🫁", layout="wide")


@st.cache_resource          # load the model once, not on every click
def get_model():
    return load_model("models/lunglens.pt")


def looks_like_xray(image: Image.Image) -> bool:
    """
    A simple sanity check: X-rays are grayscale, so the red, green and
    blue values of each pixel are almost identical. A colourful photo
    is probably not an X-ray.
    """
    pixels = np.asarray(image.convert("RGB").resize((128, 128))).astype(float)
    colourfulness = np.abs(pixels[:, :, 0] - pixels[:, :, 1]).mean() + \
                    np.abs(pixels[:, :, 1] - pixels[:, :, 2]).mean()
    return colourfulness < 10


model, threshold = get_model()
metrics_file = Path("results/metrics.json")
metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else None

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.title("🫁 LungLens")
    st.write("**See what the AI sees.** Explainable chest X-ray screening "
             "for earlier pneumonia detection.")
    st.subheader("How it works")
    st.markdown(
        "1. A **DenseNet121** neural network, pretrained on hundreds of thousands of "
        "chest X-rays, was fine-tuned on 4,694 children's X-rays.\n"
        "2. It outputs the **probability of pneumonia**.\n"
        "3. **Grad-CAM** traces the decision back to the image and "
        "highlights the regions that drove it.")
    if metrics:
        st.subheader(f"Tested on {metrics['test_images']} unseen X-rays")
        c1, c2 = st.columns(2)
        c1.metric("Sensitivity", f"{metrics['sensitivity']:.1%}")
        c2.metric("Specificity", f"{metrics['specificity']:.1%}")
        c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        c2.metric("AUC", f"{metrics['auc']:.3f}")
        st.caption("Sensitivity: share of pneumonia cases caught. "
                   "Specificity: share of healthy X-rays correctly cleared.")

# ------------------------------------------------------------------ Main
st.title("LungLens: pneumonia screening you can see")
st.warning("⚠️ **Screening aid, not a diagnosis.** LungLens is a student research "
           "prototype trained on one public dataset of children's X-rays. It must not "
           "be used to make medical decisions. Always consult a qualified doctor.")

source = st.radio("Choose an X-ray", ["Try a sample", "Upload my own"], horizontal=True)
image = None
if source == "Upload my own":
    uploaded = st.file_uploader("Chest X-ray image (JPG or PNG, front view)",
                                type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
else:
    samples = sorted(Path("samples").glob("*.jpeg"))
    choice = st.selectbox("Sample X-rays (from the test set, never seen in training)",
                          samples, format_func=lambda p: p.stem.split("_")[0] + " example " + p.stem.split("_")[1])
    image = Image.open(choice)

if image is not None:
    image = image.convert("RGB")
    if not looks_like_xray(image):
        st.error("This image is in colour, so it is probably not a chest X-ray. "
                 "The result below is unlikely to mean anything.")

    with st.spinner("Analysing the X-ray..."):
        probability, heatmap = gradcam(model, preprocess(image), threshold)
    flagged = probability >= threshold

    # Three columns: the X-ray, the heatmap, and the result with its explanation.
    col_xray, col_heat, col_result = st.columns([1, 1, 1.1])

    with col_result:
        if flagged:
            st.error(f"### 🔴 Signs of pneumonia\n"
                     f"Pneumonia probability **{probability:.0%}**. Recommend review by a doctor.")
        else:
            st.success(f"### 🟢 No signs of pneumonia\n"
                       f"Pneumonia probability **{probability:.0%}**.")
        st.progress(probability, text=f"LungLens raises an alert at {threshold:.0%} or more")
        opacity = st.slider("Heatmap strength", 0.0, 1.0, 0.6, 0.05)
        with st.expander("How do I read the heatmap?", expanded=True):
            st.markdown(
                f"- The heatmap explains the answer LungLens gave "
                f"(**{'pneumonia' if flagged else 'normal'}**).\n"
                "- **Red and yellow** areas are where the network found the strongest "
                "evidence; uncoloured areas mattered little.\n"
                "- In pneumonia, infected lung tissue fills with fluid and shows up as "
                "white, cloudy patches (**consolidation**). A trustworthy result "
                "highlights areas inside the lungs.\n"
                "- If the heat sits on text labels, the image edges or outside the body, "
                "the model may be using a shortcut: treat the result with extra caution.\n"
                "- The map is built from a 7×7 grid, so it shows regions, not exact outlines.")

    col_xray.image(image, caption="Original X-ray", use_container_width=True)
    col_heat.image(overlay(image, heatmap, opacity),
                   caption="Grad-CAM: red = regions that drove the decision",
                   use_container_width=True)
else:
    st.info("Upload a chest X-ray or pick a sample to begin.")

st.caption("Data: Kermany et al., Cell 2018 (CC BY 4.0). Pretrained weights: TorchXRayVision. "
           "Built for the UnivaBio hackathon, AI for Human Health.")
