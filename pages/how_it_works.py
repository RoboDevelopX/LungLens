"""How It Works: the medical background and the AI pipeline, explained simply."""

import streamlit as st

from src import ui

st.title("How LungLens works")
ui.disclaimer()

# ---------------------------------------------------------------------------
st.header("1. Pneumonia on a chest X-ray")
left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown(
        """
Pneumonia is an infection that inflames the lungs' tiny air sacs (alveoli) and can fill them with
fluid or pus. According to the [World Health Organization](https://www.who.int/news-room/fact-sheets/detail/pneumonia),
it is the single largest infectious cause of death in children worldwide.

On an X-ray, **air looks dark** because X-rays pass through it easily, while **dense tissue and fluid look
white**. Healthy lungs are therefore mostly dark with fine white lines (blood vessels). In pneumonia,
infected areas fill with fluid and appear as **hazy or patchy white areas** (radiologists call them
*opacities* or *consolidation*), sometimes in one lobe (often bacterial) and sometimes spread more
diffusely (often viral).

Reading X-rays takes training, and many places have too few radiologists. That is the gap a
**screening aid** can help with: flag images that need a closer look, sooner.
"""
    )
with right:
    st.markdown("**How to read a chest X-ray image**")
    st.markdown(
        """
- You look at the patient **from the front**, so the patient's **right** lung is on the **left** of the image.
- The heart is the white shape in the middle, bulging to the image's right (patient's left).
- Dark = air, white = bone, heart, fluid.
- Letters such as **R** or **L** are markers added by the technician, not anatomy.
"""
    )

# ---------------------------------------------------------------------------
st.header("2. The pipeline, step by step")
st.graphviz_chart(
    """
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#eef4fc", color="#2a78d6", fontname="Helvetica", fontsize=11];
  edge [color="#898781"];
  upload  [label="Upload\\nX-ray"];
  guard   [label="Input check\\n(is it an X-ray?)", fillcolor="#fff8ea", color="#c98500"];
  model   [label="DenseNet121\\n(transfer learning)"];
  calib   [label="Temperature\\nscaling"];
  decide  [label="Decision\\nnormal / inconclusive /\\npneumonia"];
  cam     [label="Grad-CAM\\nheatmap"];
  explain [label="Plain-language\\nexplanation"];
  stop    [label="Refuse:\\nnot an X-ray", fillcolor="#f3f2ee", color="#6b6a65"];
  upload -> guard -> model -> calib -> decide -> explain;
  model -> cam -> explain;
  guard -> stop [style=dashed, label="fails"];
}
"""
)
st.markdown(
    """
1. **Input check.** Before trusting any prediction, LungLens checks that the image looks like a chest X-ray
   (grayscale, not blank, not a document, sensible shape), then asks the model whether the image feels
   familiar (an *energy score*). Photos and documents are refused instead of being given a confident wrong answer.
2. **The model.** A DenseNet121 network, first trained on 1.2 million everyday photos (ImageNet), then
   *fine-tuned* on about 4,200 labelled pediatric chest X-rays. This is **transfer learning**: the network
   already knows edges and textures, and only has to learn what pneumonia looks like.
3. **Calibration.** The raw score is converted into an honest probability with *temperature scaling*
   (see section 4).
4. **Decision.** Instead of forcing a yes/no answer, scores close to the decision line are reported as
   **Inconclusive, please consult a clinician**.
5. **Explanation.** Grad-CAM shows which regions drove the score, and the result is explained in plain words.
"""
)

# ---------------------------------------------------------------------------
st.header("3. What is Grad-CAM?")
st.markdown(
    """
*Gradient-weighted Class Activation Mapping* (Selvaraju et al., 2017) answers the question
**"which parts of the image pushed the pneumonia score up?"**

1. The network turns the X-ray into 1,024 small **feature maps** (7 x 7 grids), each reacting to some pattern.
2. We ask: if this feature map got stronger, how much would the pneumonia score change? That is the
   **gradient**, computed with the same maths used in training.
3. Each feature map gets an importance weight (its average gradient), and we add the maps up using those weights.
4. Negative parts are dropped (we only show evidence *for* pneumonia), and the 7 x 7 result is stretched over
   the image as a heatmap.

**What it is not:** Grad-CAM shows *where* the evidence was, not *why*, and it is not an outline of the
infection. Because it is only 7 x 7 cells, it is blurry by design. LungLens checks that the heatmaps are
meaningful on the **Sanity Check** page.
"""
)

# ---------------------------------------------------------------------------
st.header("4. Honest confidence (calibration)")
metrics = ui.get_metrics()
cal = metrics.get("calibration") if metrics else None
st.markdown(
    """
A model is **calibrated** when its confidence matches reality: of all the X-rays it calls "80% likely",
about 80% should truly show pneumonia. Modern neural networks are often **over-confident** (Guo et al., 2017),
so LungLens divides the model's raw scores by one number, the **temperature T**, fitted on validation images.
T above 1 softens over-confident scores; T below 1 sharpens under-confident ones. This changes the confidence
values, never the order of the images or which class is more likely, so AUROC and accuracy at 50% stay the same.
"""
)
if cal:
    direction = "above 1: the raw scores were over-confident and were softened" if cal["temperature"] > 1 else \
        "below 1: the raw scores were under-confident and were sharpened"
    st.markdown(
        f"For this model T = **{cal['temperature']:.2f}** ({direction}). The **expected calibration error (ECE)** "
        f"went from {cal['val_ece_before']:.3f} to {cal['val_ece_after']:.3f} on the validation images it was fitted "
        f"on, and from **{cal['test_ece_before']:.3f}** to **{cal['test_ece_after']:.3f}** on the held-out test set. "
        "Details on the *Model Performance* page."
    )
    if cal["test_ece_after"] > 0.05:
        st.markdown(
            "On the test set the error is still far from 0: the official test folder differs from the training "
            "folder (see *Limitations & ethics*), and calibration only carries over to similar images. So treat the "
            "percentage as a rough guide, not an exact chance."
        )

# ---------------------------------------------------------------------------
st.header("5. Why LungLens prefers false alarms to misses")
st.markdown(
    """
A screening tool makes two kinds of mistakes:

- a **false negative** tells a sick child "no pneumonia found", and treatment may be delayed;
- a **false positive** sends a healthy child for a doctor's review, which costs time but is safe.

So the decision threshold was chosen **on validation data** to catch at least **95% of pneumonia cases**
(*sensitivity-first*), accepting more false alarms (*lower specificity*). It is also never above **50%**:
an X-ray the model rates more likely pneumonia than not is never called normal. Around the threshold sits the
**Inconclusive** zone (40%-60% around a 50% threshold), where LungLens asks for a clinician instead of
guessing. The *Model Performance* page has a slider that shows the trade-off live.
"""
)
if metrics:
    opp = metrics["operating_point"]
    if opp.get("capped"):
        st.caption(
            f"For this model the 95% rule alone would have allowed a threshold of {opp['threshold_for_target']:.2f} "
            f"on validation data, so the 50% cap applies: the operating threshold is {opp['threshold']:.2f}."
        )
