"""Sanity Check: are the heatmaps meaningful, and where do they point? Numbers from reports/sanity.json."""

import streamlit as st

from src import ui

st.title("Sanity check: can we trust the heatmaps?")
ui.disclaimer()
st.markdown(
    """
A heatmap that *looks* convincing is not automatically a faithful explanation. Adebayo et al. (2018,
*Sanity Checks for Saliency Maps*) showed that some popular methods produce nearly the same picture even for a
model with **random weights**, i.e. they act like edge detectors. This page runs three checks on LungLens's
own Grad-CAM and reports the results, good or bad.
"""
)

sanity = ui.get_report("sanity.json")
if sanity is None:
    st.info("No sanity-check results yet. Run `python scripts/sanity_check.py` after evaluation.")
    st.stop()

# ---------------------------------------------------------------------------
st.header("Check 1: model randomisation")
rnd = sanity["randomization"]
st.markdown(
    f"""
We re-initialise parts of the trained network with **random weights** and compute Grad-CAM again on
{rnd['n_images']} test X-rays. If Grad-CAM explains what the model **learned**, the maps should change a lot.
We measure similarity with the **Spearman rank correlation** between the trained and randomised heatmaps
(1 = same ranking of pixels, 0 = unrelated).
"""
)
rows = []
for key, label in (("classifier", "Final layer randomised"), ("top", "Final layer + last dense block randomised")):
    r = rnd[key]
    rows.append({"Randomised part": label, "Mean correlation with trained map": f"{r['mean']:.2f}",
                 "Middle 90% of images": f"{r['p5']:.2f} to {r['p95']:.2f}"})
st.dataframe(rows, hide_index=True, width="stretch")
st.markdown(f"**Verdict:** {rnd['verdict']}")
path = ui.figure_path("sanity_randomization.png")
if path:
    st.image(str(path), caption="Same X-ray, Grad-CAM from the trained model vs randomised models", width="stretch")

# ---------------------------------------------------------------------------
st.header("Check 2: correct and incorrect predictions")
st.markdown(
    "Heatmaps for test X-rays the model got right and wrong. Errors are the most informative: "
    "a sensible model should fail on genuinely hard images, not because it looked at the wrong place."
)
path = ui.figure_path("sanity_examples.png")
if path:
    st.image(str(path), caption="True positives, true negatives, false positives and false negatives (test set)",
             width="stretch")

# ---------------------------------------------------------------------------
st.header("Check 3: is the model looking at the edges?")
b = sanity["border"]
st.markdown(
    f"""
Lung findings sit inside the image. Text markers (**R**, **L**), rulers and image borders sit at the edges.
We measure how much of each heatmap's weight falls in the **outer {int(100 * b['band'])}% band** of the image,
which covers {100 * b['uniform_share']:.0f}% of the pixels (so a heatmap spread evenly everywhere would score
{100 * b['uniform_share']:.0f}%).

- Median share of attention in the border band over all {b['n_images']} test X-rays: **{100 * b['median']:.0f}%**
- X-rays where **more than {100 * b['warn']:.0f}%** of the attention is in the border band: **{b['n_above_warn']}**
  ({100 * b['share_above_warn']:.1f}%). The app shows an extra caution message for these.
"""
)
st.markdown(f"**What we see:** {b['verdict']}")
by_outcome = b.get("median_by_outcome")
if by_outcome:
    st.dataframe([{"Outcome on the test set": k.capitalize(), "Median border share": f"{100 * v:.0f}%"}
                  for k, v in sorted(by_outcome.items(), key=lambda kv: -kv[1])], hide_index=True)
    called_normal = [v for k, v in by_outcome.items() if k in ("true negative", "false negative")]
    called_pneumonia = [v for k, v in by_outcome.items() if k in ("true positive", "false positive")]
    if called_normal and called_pneumonia and min(called_normal) > max(called_pneumonia):
        st.markdown(
            "Heatmaps lean most on the edges for X-rays the model calls **normal**. There the evidence for pneumonia "
            "is weak, and because every heatmap is rescaled to its own maximum, faint evidence near an edge is drawn "
            "just as red as strong evidence would be. That is why the app tells users that, for a normal result, red "
            "does not mean disease."
        )
path = ui.figure_path("border_attention.png")
if path:
    st.image(str(path), caption="Test X-rays with the most attention in the border band", width="stretch")
st.caption(
    "Honest limit: these checks catch obvious failures; they cannot prove the model uses medically "
    "correct features. Only a clinical study could show that."
)
