"""Model Performance: every number here is read from reports/metrics.json."""

import altair as alt
import pandas as pd
import streamlit as st

from src import ui
from src.model import DISPLAY_NAMES

st.title("Model performance")
ui.disclaimer()

metrics = ui.get_metrics()
if metrics is None:
    st.info("No evaluation results yet. Run `python -m src.evaluate` after training to create reports/metrics.json.")
    st.stop()

test, op = metrics["test"], metrics["test"]["at_operating_threshold"]
st.markdown(
    f"All results are on the **held-out test set** ({metrics['data']['n_test']} X-rays: "
    f"{metrics['data']['test_normal']} normal, {metrics['data']['test_pneumonia']} pneumonia) that was never used "
    f"for training, threshold choice or calibration. Brackets show **95% bootstrap confidence intervals** "
    f"({metrics['bootstrap']['n_resamples']:,} resamples). Generated {metrics['generated_at']} from model "
    f"`{metrics['checkpoint']['fingerprint']}`."
)

# ---------------------------------------------------------------------------
st.header("Headline metrics at the operating threshold")
names = {
    "sensitivity": ("Sensitivity (recall)", "Of children WITH pneumonia, the share LungLens flags."),
    "specificity": ("Specificity", "Of children WITHOUT pneumonia, the share LungLens clears."),
    "precision": ("Precision (PPV)", "Of all flags, the share that really were pneumonia."),
    "f1": ("F1 score", "Balance of precision and sensitivity."),
    "accuracy": ("Accuracy", "Share of all X-rays classified correctly."),
}
rows = []
for key, (label, meaning) in names.items():
    lo, hi = op["ci95"][key]
    rows.append({"Metric": label, "Value": f"{op['metrics'][key]:.3f}", "95% CI": f"{lo:.3f} to {hi:.3f}", "What it means": meaning})
lo, hi = op["ci95"]["auroc"]
rows.append({"Metric": "AUROC", "Value": f"{test['auroc']:.3f}", "95% CI": f"{lo:.3f} to {hi:.3f}",
             "What it means": "Chance a pneumonia X-ray outscores a normal one (any threshold)."})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
c = op["confusion"]
st.caption(
    f"Operating threshold = {op['threshold']:.3f} on calibrated P(pneumonia), chosen on validation data: the "
    f"highest threshold with sensitivity >= {metrics['operating_point']['target_sensitivity']:.2f}, capped at "
    f"{metrics['operating_point'].get('max_threshold', 1.0):.2f}. "
    f"On the test set it missed {c['fn']} of {c['fn'] + c['tp']} pneumonia X-rays and raised {c['fp']} false alarms "
    f"on {c['fp'] + c['tn']} normal ones. For comparison, a plain 0.5 threshold gives sensitivity "
    f"{test['at_0.5']['metrics']['sensitivity']:.3f} and specificity {test['at_0.5']['metrics']['specificity']:.3f}."
)

# ---------------------------------------------------------------------------
st.header("Try the trade-off yourself")
sweep = pd.DataFrame(test["threshold_sweep"])
threshold = st.slider("Decision threshold on P(pneumonia)", 0.0, 1.0, round(op["threshold"], 2), 0.01,
                      help="Lower = flag more X-rays (catch more pneumonia, more false alarms).")
row = sweep.iloc[(sweep["threshold"] - threshold).abs().argmin()]
c1, c2, c3 = st.columns(3)
c1.metric("Sensitivity", f"{row['sensitivity']:.1%}", help="Pneumonia cases caught")
c2.metric("Specificity", f"{row['specificity']:.1%}", help="Normal cases correctly cleared")
c3.metric("Precision", f"{row['precision']:.1%}", help="Flags that were really pneumonia")
n_pos, n_neg = metrics["data"]["test_pneumonia"], metrics["data"]["test_normal"]
st.markdown(
    f"At **{threshold:.2f}**, out of {n_pos} children with pneumonia LungLens would miss about "
    f"**{round((1 - row['sensitivity']) * n_pos)}**, and out of {n_neg} healthy children it would raise about "
    f"**{round((1 - row['specificity']) * n_neg)}** false alarms."
)
long = sweep.melt("threshold", value_vars=["sensitivity", "specificity"], var_name="Rate", value_name="value")
long["Rate"] = long["Rate"].str.capitalize()
lines = (
    alt.Chart(long)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("threshold:Q", title="Decision threshold"),
        y=alt.Y("value:Q", title="Rate on test set", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Rate:N", scale=alt.Scale(domain=["Sensitivity", "Specificity"], range=["#2a78d6", "#eb6834"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("threshold:Q", format=".2f"), "Rate:N", alt.Tooltip("value:Q", format=".1%")],
    )
)
rule = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(color="#52514e").encode(x="threshold:Q")
st.altair_chart((lines + rule).properties(height=300), width="stretch")
st.caption("The slider only changes this page's view. The app always uses the operating threshold chosen on validation data.")

# ---------------------------------------------------------------------------
st.header("Curves and confusion matrix")
cols = st.columns(3)
for col, (name, caption) in zip(cols, [("roc_curve.png", "ROC curve"), ("pr_curve.png", "Precision-recall curve"),
                                       ("confusion_matrix.png", "Confusion matrix at the operating threshold")]):
    path = ui.figure_path(name)
    if path:
        col.image(str(path), caption=caption, width="stretch")

# ---------------------------------------------------------------------------
st.header("Calibration: is the confidence honest?")
cal = metrics.get("calibration")
if cal:
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(
            pd.DataFrame(
                {
                    "Set": ["Validation (fit)", "Test (held out)"],
                    "ECE before": [f"{cal['val_ece_before']:.3f}", f"{cal['test_ece_before']:.3f}"],
                    "ECE after": [f"{cal['val_ece_after']:.3f}", f"{cal['test_ece_after']:.3f}"],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            f"Temperature **T = {cal['temperature']:.2f}** was fitted on validation data only. "
            "ECE (expected calibration error, 15 bins) is the average gap between confidence and accuracy; 0 is perfect."
        )
        if cal.get("note"):
            st.caption(cal["note"])
    path = ui.figure_path("reliability_diagram.png")
    if path:
        c2.image(str(path), caption="Reliability diagram, test set", width="stretch")

# ---------------------------------------------------------------------------
st.header("Training")
path = ui.figure_path("training_curves.png")
if path:
    st.image(str(path), caption="Loss and validation AUROC per epoch (vertical line: backbone unfrozen)", width="stretch")
ck = metrics["checkpoint"]
history = ui.get_report("train_history.json")
overall = ""  # the charts number epochs across both stages
if history and ck["best_stage"] == "finetune":
    n_head = sum(1 for e in history["epochs"] if e["stage"] == "head")
    overall = f", i.e. epoch {n_head + ck['best_epoch']} in the charts"
st.markdown(
    f"Backbone **{DISPLAY_NAMES.get(ck['backbone'], ck['backbone'])}**, best checkpoint from the *{ck['best_stage']}* "
    f"stage, epoch {ck['best_epoch']}{overall} (validation AUROC {ck['val_auroc_during_training']:.3f}). Training "
    "images: patient-grouped 80% of the official training folder; validation: the other 20% (no patient in both)."
)
