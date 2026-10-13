"""End-to-end: the single-image pipeline and the Streamlit pages (run headless with AppTest).

These use DEMO_MODE (an untrained network), so they test the plumbing, never the accuracy.
"""

from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.predict import DISPLAY_SIZE, LungLensPredictor

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def demo_predictor():
    return LungLensPredictor.from_files(force_demo=True)


def test_predict_returns_a_complete_result(demo_predictor, xray_like):
    result = demo_predictor.predict(xray_like)
    assert result.ood.verdict in ("ok", "warn")
    assert 0.0 <= result.probability <= 1.0
    assert result.cam.shape == (DISPLAY_SIZE, DISPLAY_SIZE)
    assert 0.0 <= result.cam.min() and result.cam.max() <= 1.0
    assert result.model_view.size == (DISPLAY_SIZE, DISPLAY_SIZE)
    assert result.explanation.decision in ("normal", "inconclusive", "pneumonia")


def test_predict_is_fast_enough_on_cpu(demo_predictor, xray_like):
    demo_predictor.predict(xray_like)  # warm-up (first call allocates memory)
    times = [demo_predictor.predict(xray_like).seconds for _ in range(3)]
    assert min(times) < 3.0, f"inference took {times}"


def test_predict_refuses_a_colour_photo(demo_predictor):
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 256, (400, 500, 3), dtype=np.uint8))
    result = demo_predictor.predict(photo)
    assert result.ood.verdict == "reject"
    assert result.probability is None and result.cam is None


def test_predictor_settings_come_from_the_metrics_file(cfg):
    """The app must use exactly the temperature, threshold and band that were evaluated."""
    from src.explain import inconclusive_band
    from src.model import LungLensNet

    model = LungLensNet("densenet121", pretrained=False).eval()
    evaluated = {"operating_point": {"threshold": 0.4321}, "calibration": {"temperature": 1.7},
                 "decision": {"low": 0.3321, "high": 0.5321}, "ood": {"energy_threshold": -2.5}}
    p = LungLensPredictor(cfg, model, evaluated)
    assert (p.temperature, p.threshold, p.low, p.high, p.energy_threshold) == (1.7, 0.4321, 0.3321, 0.5321, -2.5)

    # An older metrics file without calibration/decision/OOD sections still works (T = 1, band from config).
    p = LungLensPredictor(cfg, model, {"operating_point": {"threshold": 0.3}})
    assert p.temperature == 1.0 and p.energy_threshold is None
    assert (p.low, p.high) == inconclusive_band(0.3, cfg["decision"]["inconclusive_odds_factor"])


def test_weights_are_downloaded_when_missing_and_a_url_is_set(tmp_path, monkeypatch):
    """Cloud deployments fetch models/lunglens.pt from LUNGLENS_WEIGHTS_URL (tested with a local web server)."""
    import functools
    import http.server
    import threading

    from src.predict import download_weights_if_missing

    served = tmp_path / "release"
    served.mkdir()
    (served / "lunglens.pt").write_bytes(b"fake weights" * 1000)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.setenv(name, "127.0.0.1,localhost")
        monkeypatch.setenv("LUNGLENS_WEIGHTS_URL", f"http://127.0.0.1:{server.server_port}/lunglens.pt")
        target = tmp_path / "models" / "lunglens.pt"
        download_weights_if_missing(target)
        assert target.read_bytes() == b"fake weights" * 1000
        assert not target.with_suffix(".part").exists()
    finally:
        server.shutdown()


def test_split_facts_are_computed_from_the_split_file():
    import pandas as pd

    from src import ui

    facts = ui.get_split_facts()
    table = pd.read_csv(ROOT / "reports" / "splits.csv")
    assert facts["n"] == {split: int((table["split"] == split).sum()) for split in ("train", "val", "test")}
    assert sum(facts["n"].values()) == len(table)
    assert 0 <= facts["val_pneumonia_share"] <= 1
    assert facts["shared_ids_same_kind"] <= facts["shared_ids"]


@pytest.fixture
def demo_env(monkeypatch):
    monkeypatch.setenv("LUNGLENS_DEMO_MODE", "1")
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.mark.parametrize(
    "page,title",
    [("pages/screen.py", "LungLens"), ("pages/how_it_works.py", "How LungLens works"),
     ("pages/model_performance.py", "Model performance"), ("pages/sanity_check.py", "Sanity check"),
     ("pages/limitations_and_ethics.py", "Limitations & ethics")],
)
def test_every_page_renders_without_errors(demo_env, page, title):
    """Start the real entry point (app.py), then open each page from the navigation."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    at.switch_page(page).run()
    assert not at.exception, at.exception
    assert any(title in t.value for t in at.title), [t.value for t in at.title]


def test_home_page_analyses_a_sample_and_shows_the_demo_warning(demo_env):
    if not (ROOT / "assets/samples/manifest.json").exists():
        pytest.skip("no bundled samples")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "DEMO_MODE" in html                       # the untrained-model banner is visible
    assert "not a diagnostic device" in html         # the disclaimer is visible
    assert len(at.slider) >= 1                       # the opacity slider exists once a result is shown
