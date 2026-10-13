"""Out-of-distribution guard: obvious non-X-rays are refused, X-ray-like images pass."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.ood import OODResult, check_energy, check_image, energy_score


def test_xray_like_image_passes(cfg, xray_like):
    result = check_image(xray_like, cfg["ood"])
    assert result.verdict == "ok", result.reasons


def test_rgb_saved_xray_still_passes(cfg, xray_like):
    """283 real dataset files are stored as RGB with identical channels; they must not be refused."""
    assert check_image(xray_like.convert("RGB"), cfg["ood"]).verdict == "ok"


def test_colour_photo_is_rejected(cfg):
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 256, (400, 500, 3), dtype=np.uint8))
    result = check_image(photo, cfg["ood"])
    assert result.verdict == "reject"
    assert any("colour" in r for r in result.reasons)


def test_blank_image_is_rejected(cfg):
    assert check_image(Image.new("L", (512, 512), 128), cfg["ood"]).verdict == "reject"


def test_document_is_rejected(cfg):
    page = Image.new("L", (850, 1100), 255)
    draw = ImageDraw.Draw(page)
    for y in range(80, 1000, 28):
        draw.text((70, y), "lorem ipsum dolor sit amet " * 3, fill=0)
    result = check_image(page, cfg["ood"])
    assert result.verdict == "reject"
    assert any("white" in r for r in result.reasons)


def test_tiny_image_is_rejected(cfg, xray_like):
    assert check_image(xray_like.resize((100, 90)), cfg["ood"]).verdict == "reject"


def test_unusual_aspect_ratio_only_warns(cfg, xray_like):
    wide = xray_like.resize((1500, 500))  # width/height = 3.0
    result = check_image(wide, cfg["ood"])
    assert result.verdict == "warn"
    assert result.codes == ["unusual_shape"]


def test_every_message_has_a_stable_code(cfg, xray_like):
    """reports/ood_eval.json counts checks by code, so codes must not contain image-specific numbers."""
    tiny_and_wide = xray_like.resize((400, 100))  # too small AND unusual shape
    result = check_image(tiny_and_wide, cfg["ood"])
    assert result.codes == ["too_small", "unusual_shape"]
    assert len(result.codes) == len(result.reasons)
    odd = check_energy(OODResult("ok"), np.array([0.5, 0.2]), energy_threshold=-2.0)
    assert odd.codes == ["unusual_for_model"]


def test_energy_score_matches_definition():
    logits = np.array([[2.0, -1.0], [0.0, 0.0]])
    expected = -np.log(np.exp(logits).sum(axis=1))
    assert np.allclose(energy_score(logits), expected)


def test_energy_check_warns_only_above_threshold():
    logits = np.array([0.5, 0.2])  # energy ~ -1.05
    calm = check_energy(OODResult("ok"), logits, energy_threshold=0.0)
    assert calm.verdict == "ok"
    odd = check_energy(OODResult("ok"), logits, energy_threshold=-2.0)
    assert odd.verdict == "warn" and odd.reasons


def test_energy_check_never_overrides_a_reject():
    rejected = OODResult("reject", ["colour"])
    assert check_energy(rejected, np.array([5.0, 5.0]), energy_threshold=-100).verdict == "reject"


@pytest.mark.parametrize("value", [0, 255])
def test_all_black_or_all_white_rejected(cfg, value):
    assert check_image(Image.new("L", (600, 600), value), cfg["ood"]).verdict == "reject"
