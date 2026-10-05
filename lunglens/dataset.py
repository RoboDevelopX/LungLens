"""
dataset.py - Loading the X-ray images and splitting them fairly.

Why a "patient-level" split?
----------------------------
Some children in the dataset have several X-rays. If one X-ray of a child
lands in the training set and another in the validation set, the model can
"recognise the child" rather than the disease, and our validation score
would look better than it really is. So we split by patient: all of a
child's X-rays go to the same side.
"""

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = Path("data/chest_xray")
LABELS = {"NORMAL": 0, "PNEUMONIA": 1}


def patient_id(filename: str) -> str:
    """
    Work out which child an image belongs to from its file name:
      person1000_bacteria_2931.jpeg -> "person1000"
      IM-0115-0001.jpeg             -> "IM-0115"
      NORMAL2-IM-1326-0001.jpeg     -> "NORMAL2-IM-1326"
    """
    stem = Path(filename).stem
    if stem.startswith("person"):
        return stem.split("_")[0]
    return stem.rsplit("-", 1)[0]


def list_images(split: str):
    """Return a list of (image_path, label) pairs for "train" or "test"."""
    items = []
    for class_name, label in LABELS.items():
        for path in sorted((DATA_DIR / split / class_name).glob("*.jpeg")):
            items.append((path, label))
    return items


def train_val_split(items, val_fraction: float = 0.1, seed: int = 42):
    """Hold out about 10% of the patients for validation."""
    patients = sorted({patient_id(p.name) for p, _ in items})
    random.Random(seed).shuffle(patients)           # fixed seed = same split every run
    val_patients = set(patients[: int(len(patients) * val_fraction)])
    train = [(p, y) for p, y in items if patient_id(p.name) not in val_patients]
    val = [(p, y) for p, y in items if patient_id(p.name) in val_patients]
    return train, val


class XrayDataset(Dataset):
    """A PyTorch Dataset: given an index, returns (image tensor, label)."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), float(label)
