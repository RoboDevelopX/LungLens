"""Shared pytest fixtures. Tests use small synthetic images and an UNTRAINED model,
so they run anywhere in about a minute without the dataset or the trained weights."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make "src" importable

from src.model import LungLensNet  # noqa: E402
from src.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def tiny_model():
    """DenseNet121 with random weights (pretrained=False: no download needed)."""
    torch.manual_seed(0)
    return LungLensNet("densenet121", pretrained=False).eval()


@pytest.fixture
def xray_like():
    """A synthetic grayscale image with X-ray-like statistics: dark background,
    two brighter 'lung field' ellipses with texture, a bright 'spine' in the middle."""
    rng = np.random.default_rng(0)
    h, w = 600, 700
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 40.0)
    for cx in (0.32 * w, 0.68 * w):
        inside = ((xx - cx) / (0.17 * w)) ** 2 + ((yy - 0.5 * h) / (0.36 * h)) ** 2 < 1
        img[inside] = 110
    img[:, int(0.47 * w): int(0.53 * w)] = 200
    img += rng.normal(0, 18, size=img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")
