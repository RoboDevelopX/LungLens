"""Build our train / val / test split and save it to reports/splits.csv.

Why we re-split:
  * The official `val` folder has only 16 images, far too few to pick a
    threshold or fit calibration, so we ignore it.
  * We move about 20% of the official `train` images into our own `val`.
  * The official `test` folder (624 images) is kept untouched for the final,
    one-time evaluation.

Why we split by PATIENT, not by image:
  Some patients have several X-rays. If one of a patient's images is in train
  and another in val, the model can "recognise the patient" and val scores
  look better than they really are (data leakage). So every image of a given
  patient goes to the same side of the split.

Patient IDs come from the file names:
  PNEUMONIA:  person1374_bacteria_3506.jpeg   -> "person1374"
  NORMAL:     IM-0519-0001-0001.jpeg          -> "IM-0519"
              NORMAL2-IM-0519-0001.jpeg       -> "NORMAL2-IM-0519"
  (We keep the "NORMAL2-" prefix: it marks a different batch of files, and we
  have no evidence that IM-0519 and NORMAL2-IM-0519 are the same person.)

Honest caveat (checked on the real file names): about half of the "personN"
IDs in train have both a *_bacteria_* and a *_virus_* file, so the bacteria and
virus files may be numbered independently and one "personN" may really be two
people. Grouping them together is the SAFE direction: it can only make the
split stricter, never leak a patient across train and val.

This file only needs numpy/pandas, so it runs before PyTorch is installed.

Usage:
    python -m src.split
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import data_dir, load_config, splits_csv

PERSON_RE = re.compile(r"^(person\d+)_")             # pneumonia files
NORMAL_RE = re.compile(r"^((?:NORMAL2-)?IM-\d+)-")   # normal files


def patient_id(filename: str) -> str:
    """Extract the patient ID from a Kermany file name (see module docstring)."""
    for pattern in (PERSON_RE, NORMAL_RE):
        match = pattern.match(filename)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot parse a patient ID from {filename!r}")


def list_images(root: Path, folder: str, classes: list[str]) -> pd.DataFrame:
    """One row per image in root/<folder>/<class>/, with its label and patient ID."""
    rows = []
    for label, cls in enumerate(classes):
        for path in sorted((root / folder / cls).glob("*.jpeg")):
            rows.append(
                {
                    "path": f"{folder}/{cls}/{path.name}",  # relative to the data folder
                    "label": label,                         # 0 = NORMAL, 1 = PNEUMONIA
                    "patient_id": patient_id(path.name),
                }
            )
    return pd.DataFrame(rows)


def grouped_stratified_split(df: pd.DataFrame, val_fraction: float, seed: int) -> pd.Series:
    """Return a 'train'/'val' label for every row of df.

    For each class separately (that is the "stratified" part), we shuffle the
    patients and move whole patients (the "grouped" part) into val until val
    holds about `val_fraction` of that class's images.
    """
    rng = np.random.default_rng(seed)
    assignment = pd.Series("train", index=df.index)

    for label in sorted(df["label"].unique()):
        class_rows = df[df["label"] == label]
        target = val_fraction * len(class_rows)

        # Sort first so the result never depends on file-system order, then shuffle.
        patients = rng.permutation(sorted(class_rows["patient_id"].unique()))

        n_val = 0
        for pid in patients:
            if n_val >= target:
                break
            rows_of_patient = class_rows.index[class_rows["patient_id"] == pid]
            assignment[rows_of_patient] = "val"
            n_val += len(rows_of_patient)

    return assignment


def build_splits(cfg: dict) -> pd.DataFrame:
    root = data_dir(cfg)
    classes = cfg["classes"]

    train_pool = list_images(root, "train", classes)
    train_pool["split"] = grouped_stratified_split(train_pool, cfg["split"]["val_fraction"], cfg["seed"])

    test = list_images(root, "test", classes)
    test["split"] = "test"

    return pd.concat([train_pool, test], ignore_index=True)


def summarise(df: pd.DataFrame, classes: list[str]) -> None:
    """Print image and patient counts per split, plus leakage checks."""
    print(f"{'split':<6} {'NORMAL':>7} {'PNEUMONIA':>10} {'total':>6} {'%pneu':>6} {'patients':>9}")
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        counts = part["label"].value_counts()
        n0, n1 = counts.get(0, 0), counts.get(1, 0)
        print(f"{split:<6} {n0:>7} {n1:>10} {len(part):>6} {100 * n1 / len(part):>5.1f}% {part['patient_id'].nunique():>9}")

    train_p = set(df.loc[df["split"] == "train", "patient_id"])
    val_p = set(df.loc[df["split"] == "val", "patient_id"])
    test_p = set(df.loc[df["split"] == "test", "patient_id"])
    print(f"patients in both train and val:  {len(train_p & val_p)}  (must be 0)")
    # The official test folder came from the dataset authors; we only report this.
    print(f"patient IDs in both train+val and test: {len((train_p | val_p) & test_p)}  (official split, reported only)")


def main() -> None:
    cfg = load_config()
    df = build_splits(cfg)
    out = splits_csv(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summarise(df, cfg["classes"])
    print(f"saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
