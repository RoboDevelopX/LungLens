"""Shared Streamlit helpers used by app.py and every page in pages/.

Keeping these in one place gives every page the same look, the same
disclaimer and ONE cached copy of the model.
"""

import io
import os

import streamlit as st
from PIL import Image

from src.predict import LungLensPredictor
from src.utils import load_config, load_json, repo_path

# One calm palette for the whole app (text colours all pass WCAG AA contrast on the page background).
INK, INK_2, MUTED, BORDER, SURFACE = "#0b0b0b", "#52514e", "#6b6a65", "#e1e0d9", "#fcfcfb"
STATUS = {  # accent colour + icon (+ the icon's text colour), so meaning never relies on colour alone
    "pneumonia": ("#d03b3b", "▲", "white"),
    "normal": ("#0c8a0c", "✓", "white"),
    "inconclusive": ("#e8a317", "?", INK),   # dark "?" on amber: white would be too faint
    "not_xray": ("#6b6a65", "✕", "white"),
}

CSS = f"""
<style>
html, body, [class*="css"] {{ font-size: 17px; }}
.block-container {{ max-width: 1150px; padding-top: 2rem; }}
.ll-banner {{ border: 1px solid {BORDER}; border-left: 5px solid #c98500; background: #fff8ea;
  padding: 0.7rem 1rem; border-radius: 6px; color: {INK}; margin: 0.4rem 0 1rem 0; }}
.ll-demo {{ border-left-color: #d03b3b; background: #fdf0f0; }}
.ll-card {{ border: 1px solid {BORDER}; border-left: 6px solid var(--accent); border-radius: 8px;
  background: white; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }}
.ll-card h3 {{ margin: 0 0 0.3rem 0; font-size: 1.35rem; color: {INK}; }}
.ll-icon {{ display: inline-block; width: 1.6rem; height: 1.6rem; line-height: 1.6rem; text-align: center;
  border-radius: 50%; background: var(--accent); color: var(--icon-ink); font-weight: 700; margin-right: 0.5rem; }}
.ll-prob {{ font-size: 2.6rem; font-weight: 650; color: {INK}; line-height: 1.1; }}
.ll-sub {{ color: {INK_2}; font-size: 0.92rem; }}
.ll-meter {{ position: relative; height: 14px; border-radius: 7px; overflow: hidden; display: flex; margin: 0.6rem 0 0.2rem 0; }}
.ll-meter div {{ height: 100%; }}
.ll-marker {{ position: relative; height: 20px; margin-top: -26px; }}
.ll-marker span {{ position: absolute; top: 0; width: 4px; height: 26px; margin-left: -2px; background: {INK};
  border: 2px solid white; border-radius: 3px; box-sizing: content-box; }}
.ll-scale {{ display: flex; justify-content: space-between; color: {INK_2}; font-size: 0.8rem; margin-top: 0.5rem; }}
.ll-legend {{ height: 10px; border-radius: 5px; background: linear-gradient(90deg, #ffffcc, #fd8d3c, #bd0026); }}
</style>
"""


def setup_page() -> None:
    """Called once by app.py before every page: browser-tab title, wide layout, shared CSS."""
    st.set_page_config(page_title="LungLens", page_icon="🫁", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model...")
def get_predictor() -> LungLensPredictor:
    """Load the model ONCE per server process (st.cache_resource), not once per click."""
    return LungLensPredictor.from_files(force_demo=os.environ.get("LUNGLENS_DEMO_MODE") == "1")


@st.cache_data
def get_metrics() -> dict | None:
    path = repo_path(load_config()["paths"]["reports_dir"]) / "metrics.json"
    return load_json(path) if path.exists() else None


@st.cache_data
def get_report(name: str) -> dict | None:
    path = repo_path(load_config()["paths"]["reports_dir"]) / name
    return load_json(path) if path.exists() else None


@st.cache_data
def get_split_facts() -> dict | None:
    """Facts about the data split, computed from reports/splits.csv (never typed by hand)."""
    path = repo_path(load_config()["paths"]["splits_csv"])
    if not path.exists():
        return None
    import pandas as pd

    df = pd.read_csv(path)
    df["kind"] = df["path"].map(lambda p: "bacteria" if "_bacteria_" in p else "virus" if "_virus_" in p else "normal")
    official_train, test = df[df["split"] != "test"], df[df["split"] == "test"]
    shared = set(official_train["patient_id"]) & set(test["patient_id"])
    same_kind = [p for p in shared
                 if set(official_train.loc[official_train["patient_id"] == p, "kind"])
                 & set(test.loc[test["patient_id"] == p, "kind"])]
    val = df[df["split"] == "val"]
    return {
        "n": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
        "val_pneumonia_share": float((val["label"] == 1).mean()),
        "test_pneumonia_share": float((test["label"] == 1).mean()),
        "shared_ids": len(shared),
        "shared_ids_same_kind": len(same_kind),
        "test_images_with_shared_id": int(test["patient_id"].isin(shared).sum()),
    }


def figure_path(name: str):
    path = repo_path(load_config()["paths"]["figures_dir"]) / name
    return path if path.exists() else None


def disclaimer() -> None:
    st.markdown(
        '<div class="ll-banner"><b>Educational screening aid, not a diagnostic device.</b> '
        "LungLens is a student project. It has not been clinically validated and must not be used "
        "to make medical decisions. Always consult a qualified clinician.</div>",
        unsafe_allow_html=True,
    )


def demo_banner(predictor: LungLensPredictor) -> None:
    if predictor.demo_mode:
        st.markdown(
            '<div class="ll-banner ll-demo"><b>DEMO_MODE: untrained model.</b> No trained weights were found '
            "(models/lunglens.pt) or demo mode was forced. Every result below is meaningless and exists only "
            "for interface development.</div>",
            unsafe_allow_html=True,
        )
    elif not predictor.fingerprint_ok:
        st.warning(
            "The model file does not match the one that produced reports/metrics.json. "
            "Re-run `python -m src.evaluate` so the reported numbers describe this model."
        )


def status_card(decision: str, headline: str, body_html: str = "") -> None:
    colour, icon, icon_ink = STATUS[decision]
    st.markdown(
        f'<div class="ll-card" style="--accent:{colour};--icon-ink:{icon_ink}"><h3><span class="ll-icon">{icon}</span>{headline}</h3>'
        f"{body_html}</div>",
        unsafe_allow_html=True,
    )


def probability_meter(p: float, low: float, high: float) -> str:
    """HTML bar: normal | inconclusive | pneumonia zones, with a marker at p."""
    zones = [
        (low, "#dff0df"),            # normal zone (light green tint)
        (high - low, "#fbefd3"),     # inconclusive zone (light amber tint)
        (1 - high, "#f7dcdc"),       # pneumonia zone (light red tint)
    ]
    bars = "".join(f'<div style="width:{100 * w:.2f}%;background:{c}"></div>' for w, c in zones)
    return (
        f'<div class="ll-meter">{bars}</div>'
        f'<div class="ll-marker"><span style="left:{100 * p:.2f}%"></span></div>'
        f'<div class="ll-scale"><span>0% · normal</span><span>inconclusive {100 * low:.0f}-{100 * high:.0f}%</span>'
        f"<span>pneumonia · 100%</span></div>"
    )


def load_uploaded_image(data: bytes) -> Image.Image | None:
    """Open uploaded bytes as an image; None if the file is not a readable image."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force decoding now, so a corrupt file fails here and not later
        return image
    except Exception:
        return None
