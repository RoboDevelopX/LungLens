"""Grad-CAM: shape, range, hooks, a cross-check against the pytorch-grad-cam library,
and the helper measurements used by the Sanity Check page."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.gradcam import GradCAM, border_fraction, overlay, randomized_copy, rank_correlation, target_layer_for


@pytest.fixture
def x(cfg, xray_like):
    from src.data import build_transforms

    return build_transforms(cfg, train=False)(xray_like).unsqueeze(0)


@pytest.fixture
def explainer(tiny_model):
    """Grad-CAM on the shared test model; its hook is removed afterwards so tests stay independent."""
    cam = GradCAM(tiny_model, target_layer_for(tiny_model))
    yield cam
    cam.remove()


def test_cam_shape_and_range(explainer, x):
    cam, logits = explainer(x, class_idx=1)
    assert cam.shape == (224, 224)
    assert np.isfinite(cam).all()
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert cam.max() in (0.0, 1.0)   # rescaled so the strongest point is exactly 1 (or all zeros)
    assert logits.shape == (1, 2)


def test_hooks_capture_feature_map_and_gradient(explainer, x):
    explainer(x, class_idx=1)
    assert explainer.activations.shape == (1, 1024, 7, 7)
    assert explainer.gradients.shape == explainer.activations.shape


def test_default_class_is_the_predicted_one(explainer, x):
    cam_default, logits = explainer(x)
    predicted = int(logits.argmax())
    cam_explicit, _ = explainer(x, class_idx=predicted)
    assert np.allclose(cam_default, cam_explicit, atol=1e-6)


def test_works_inside_no_grad(explainer, x):
    with torch.no_grad():
        cam, _ = explainer(x, class_idx=1)
    assert cam.shape == (224, 224)


def test_attached_explainer_does_not_break_plain_inference(tiny_model, explainer, x):
    """Regression: the hook used to ask for a gradient even under no_grad, which crashed a plain forward pass."""
    with torch.no_grad():
        logits = tiny_model(x)
    assert logits.shape == (1, 2)


def test_matches_pytorch_grad_cam_library(tiny_model, explainer, x):
    """Cross-check ONLY: the app never uses this library; our own code must agree with it."""
    lib = pytest.importorskip("pytorch_grad_cam")
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    ours, _ = explainer(x, class_idx=1)
    with lib.GradCAM(model=tiny_model, target_layers=[target_layer_for(tiny_model)]) as reference:
        theirs = reference(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
    assert theirs.shape == ours.shape
    assert np.corrcoef(ours.ravel(), theirs.ravel())[0, 1] > 0.99
    assert np.abs(ours - theirs).max() < 0.05


def test_border_fraction():
    uniform = np.ones((100, 100))
    assert border_fraction(uniform, band=0.10) == pytest.approx(1 - 0.8 * 0.8)
    centre = np.zeros((100, 100))
    centre[40:60, 40:60] = 1
    assert border_fraction(centre) == 0.0
    edge = np.zeros((100, 100))
    edge[:5, :] = 1
    assert border_fraction(edge) == 1.0
    assert border_fraction(np.zeros((10, 10))) == 0.0


def test_overlay_keeps_size_and_is_rgb():
    image = Image.new("L", (300, 300), 100)
    cam = np.linspace(0, 1, 300 * 300).reshape(300, 300)
    out = overlay(image, cam, opacity=0.5)
    assert out.size == (300, 300) and out.mode == "RGB"
    # opacity 0 gives back the original gray image
    assert np.array_equal(np.asarray(overlay(image, cam, 0.0))[..., 0], np.asarray(image))


def test_randomized_copy_changes_weights_but_not_original(tiny_model):
    before = tiny_model.classifier.weight.clone()
    rand = randomized_copy(tiny_model, seed=1, layers="top")
    assert torch.equal(tiny_model.classifier.weight, before)
    assert not torch.equal(rand.classifier.weight, before)


def test_rank_correlation_bounds():
    a = np.arange(100.0)
    assert rank_correlation(a, a) == pytest.approx(1.0)
    assert rank_correlation(a, -a) == pytest.approx(-1.0)
