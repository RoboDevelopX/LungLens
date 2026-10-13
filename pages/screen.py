"""Screen an X-ray (home page): upload or pick a sample, then see the result,
the Grad-CAM heatmap and the plain-language explanation."""

import json

import streamlit as st

from src import ui
from src.explain import format_chance
from src.gradcam import overlay
from src.utils import repo_path

predictor = ui.get_predictor()
SAMPLES_DIR = repo_path("assets/samples")


@st.cache_data(max_entries=32, show_spinner="Analysing the X-ray...")
def analyse(image_bytes: bytes):
    """Run the full pipeline once per distinct image (slider moves reuse the cached result)."""
    image = ui.load_uploaded_image(image_bytes)
    if image is None:
        return None
    return ui.get_predictor().predict(image)


def load_samples() -> list[dict]:
    manifest = SAMPLES_DIR / "manifest.json"
    return json.loads(manifest.read_text()) if manifest.exists() else []


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🫁 LungLens")
st.markdown("#### See what the AI sees")
st.markdown(
    "Upload a chest X-ray. LungLens screens it for signs of pneumonia, shows **where** the AI looked, "
    "and explains in plain language what the result does and does not mean."
)
ui.disclaimer()
ui.demo_banner(predictor)

# ---------------------------------------------------------------------------
# 1. Choose an image
# ---------------------------------------------------------------------------
samples = load_samples()
choice = st.radio("Image source", ["Upload your own", "Try a sample image"], horizontal=True,
                  index=1 if samples else 0, disabled=not samples)

image_bytes, caption = None, None
if choice == "Upload your own":
    upload = st.file_uploader("Chest X-ray, frontal view (PNG or JPG)", type=["png", "jpg", "jpeg"])
    if upload is not None:
        image_bytes, caption = upload.getvalue(), upload.name
else:
    names = [s["title"] for s in samples]
    picked = samples[names.index(st.selectbox("Sample image", names))]
    image_bytes = (SAMPLES_DIR / picked["file"]).read_bytes()
    caption = picked["title"]
    st.caption(f"Source: {picked['attribution']}" + (f" · Dataset label: **{picked['label']}**" if picked.get("label") else ""))

if image_bytes is None:
    st.info("Upload an X-ray above, or switch to *Try a sample image*.")
    st.stop()

result = analyse(image_bytes)
if result is None:
    ui.status_card("not_xray", "This file could not be opened as an image",
                   '<p class="ll-sub">Please upload a PNG or JPG file.</p>')
    st.stop()

# ---------------------------------------------------------------------------
# 2. Input check (OOD guard)
# ---------------------------------------------------------------------------
if result.ood.verdict == "reject":
    reasons = "".join(f"<li>{r}</li>" for r in result.ood.reasons)
    ui.status_card(
        "not_xray",
        "This does not look like a chest X-ray, so LungLens will not analyse it",
        f"<ul>{reasons}</ul><p class='ll-sub'>A screening model only knows what it was trained on. Giving a "
        "confident answer for a photo or document would be misleading, so LungLens refuses instead.</p>",
    )
    st.image(ui.load_uploaded_image(image_bytes), caption=caption, width=320)
    st.stop()
if result.ood.verdict == "warn":
    st.warning("**Please read this result with caution.** " + " ".join(result.ood.reasons))

# ---------------------------------------------------------------------------
# 3. Result
# ---------------------------------------------------------------------------
exp, p = result.explanation, result.probability
left, right = st.columns([5, 7], gap="large")
with left:
    ui.status_card(
        exp.decision,
        exp.headline,
        f'<div class="ll-sub">Calibrated chance of pneumonia</div><div class="ll-prob">{format_chance(p)}</div>'
        + ui.probability_meter(p, predictor.low, predictor.high),
    )
    st.markdown("**What this means**")
    for paragraph in exp.layperson:
        st.markdown(paragraph)
    with st.expander("Clinician view"):
        for line in exp.clinician:
            st.markdown(f"- {line}")
        metrics = ui.get_metrics()
        if metrics:
            op = metrics["test"]["at_operating_threshold"]
            m, ci = op["metrics"], op["ci95"]
            st.markdown(
                f"- Held-out test set (n = {metrics['data']['n_test']}), at this threshold: sensitivity "
                f"{m['sensitivity']:.3f} (95% CI {ci['sensitivity'][0]:.3f}-{ci['sensitivity'][1]:.3f}), "
                f"specificity {m['specificity']:.3f} (95% CI {ci['specificity'][0]:.3f}-{ci['specificity'][1]:.3f})."
            )
        st.markdown("- Training data: pediatric (age 1-5) frontal chest X-rays from one hospital in Guangzhou; "
                    "no external validation. See *Limitations & Ethics*.")

with right:
    opacity = st.slider("Heatmap opacity", 0.0, 1.0, 0.55, 0.05,
                        help="Fade the Grad-CAM colours in and out over the X-ray.")
    blended = overlay(result.model_view, result.cam, opacity)
    side, single = st.tabs(["Side by side", "Overlay only"])
    with side:
        c1, c2 = st.columns(2)
        c1.image(result.model_view, caption="What the model sees (centre crop)", width="stretch")
        c2.image(blended, caption="Grad-CAM: where the AI looked", width="stretch")
    with single:
        st.image(blended, caption="Grad-CAM overlay", width="stretch")
    st.markdown('<div class="ll-legend"></div><div class="ll-scale"><span>little influence</span>'
                "<span>strong influence on the pneumonia score</span></div>", unsafe_allow_html=True)
    st.caption(
        "The heatmap explains the **pneumonia** score. It is coarse (a 7 x 7 grid, smoothed), so it shows "
        "regions, not exact outlines. The patient's right side is on the left of the image."
    )

st.caption(
    f"Analysed in {result.seconds:.2f} s on CPU · model "
    + ("DEMO_MODE (untrained)" if predictor.demo_mode else f"{ui.get_metrics()['checkpoint']['fingerprint']}")
    + " · How was this measured? See *Model Performance*."
)
