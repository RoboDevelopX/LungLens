"""Small shared helpers: config loading, paths, seeding and device choice.

Everything else in `src/` imports from here so that paths and settings are
defined in exactly one place (`config.yaml`).
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

# The repository root is the parent of this file's folder (src/ -> repo root).
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | str = ROOT / "config.yaml") -> dict:
    """Read config.yaml into a plain dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(relative: str) -> Path:
    """Turn a path from config.yaml (relative to the repo root) into an absolute Path."""
    return ROOT / relative


def data_dir(cfg: dict) -> Path:
    """Folder that holds the extracted dataset.

    The environment variable LUNGLENS_DATA_DIR wins over config.yaml, so the
    1+ GB of images can live outside the repository (e.g. on a faster disk).
    """
    override = os.environ.get("LUNGLENS_DATA_DIR")
    return Path(override) if override else repo_path(cfg["paths"]["data_dir"])


def splits_csv(cfg: dict) -> Path:
    """Location of splits.csv (file names + labels + split, no pixels).

    It is committed to git so anyone can reproduce exactly which images were
    used for training, validation and testing.
    """
    return repo_path(cfg["paths"]["splits_csv"])


def set_seed(seed: int) -> None:
    """Make runs repeatable: same seed -> same split, same augmentations, same init."""
    random.seed(seed)
    np.random.seed(seed)
    # torch is imported here (not at the top) so that data-preparation
    # scripts that only need numpy/pandas can run without PyTorch installed.
    import torch

    torch.manual_seed(seed)


def get_device():
    """Use a GPU when one exists (e.g. Colab T4), otherwise the CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path: Path) -> None:
    """Write a dict to pretty-printed JSON, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
