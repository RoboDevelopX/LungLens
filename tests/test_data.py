"""Data pipeline: tensor shapes, grayscale handling, class weights and the patient-grouped split."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data import build_transforms, class_weights
from src.split import grouped_stratified_split, patient_id


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("size,mode", [((1281, 888), "L"), ((900, 1100), "L"), ((640, 480), "RGB")])
def test_transform_output_shape(cfg, train, size, mode):
    image = Image.new(mode, size, color=128 if mode == "L" else (128, 128, 128))
    x = build_transforms(cfg, train=train)(image.convert("L"))
    assert x.shape == (3, 224, 224)
    assert x.dtype == torch.float32


def test_three_channels_are_identical_before_normalisation(cfg, xray_like):
    x = build_transforms(cfg, train=False)(xray_like)
    mean = torch.tensor(cfg["image"]["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["image"]["std"]).view(3, 1, 1)
    raw = x * std + mean  # undo normalisation
    assert torch.allclose(raw[0], raw[1], atol=1e-5) and torch.allclose(raw[1], raw[2], atol=1e-5)


def test_no_horizontal_flip_in_train_transforms(cfg):
    names = [type(t).__name__ for t in build_transforms(cfg, train=True).transforms]
    assert "RandomHorizontalFlip" not in names


def test_class_weights_inverse_frequency():
    labels = [0] * 25 + [1] * 75  # 1 normal : 3 pneumonia
    w = class_weights(labels)
    assert w.tolist() == pytest.approx([2.0, 2 / 3])
    # weighted counts are equal: the rare class is not drowned out
    assert 25 * w[0].item() == pytest.approx(75 * w[1].item())


@pytest.mark.parametrize(
    "name,expected",
    [
        ("person1374_bacteria_3506.jpeg", "person1374"),
        ("person27_virus_60.jpeg", "person27"),
        ("IM-0519-0001-0001.jpeg", "IM-0519"),
        ("NORMAL2-IM-0519-0001.jpeg", "NORMAL2-IM-0519"),
    ],
)
def test_patient_id_parsing(name, expected):
    assert patient_id(name) == expected


def test_patient_id_rejects_unknown_names():
    with pytest.raises(ValueError):
        patient_id("scan_001.jpeg")


def test_grouped_split_keeps_patients_together_and_is_stratified():
    rng = np.random.default_rng(0)
    rows = []
    for label, n_patients in ((0, 120), (1, 300)):
        for p in range(n_patients):
            for _ in range(rng.integers(1, 4)):  # 1-3 images per patient
                rows.append({"label": label, "patient_id": f"{label}-{p}"})
    df = pd.DataFrame(rows)
    split = grouped_stratified_split(df, val_fraction=0.2, seed=42)

    # no patient appears on both sides
    sides = df.assign(split=split).groupby("patient_id")["split"].nunique()
    assert (sides == 1).all()
    # each class gets roughly 20% val (whole patients make it slightly over, never under)
    for label in (0, 1):
        frac = (split[df["label"] == label] == "val").mean()
        assert 0.2 <= frac < 0.25
    # same seed -> same split
    assert split.equals(grouped_stratified_split(df, val_fraction=0.2, seed=42))
