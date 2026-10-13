"""PyTorch data pipeline: image transforms, Dataset and DataLoaders.

The split itself (which image goes to train / val / test) is decided once by
`src/split.py` and stored in reports/splits.csv. This file just reads that
table and turns images into tensors the model can eat.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils import data_dir, splits_csv


def build_transforms(cfg: dict, train: bool) -> transforms.Compose:
    """Image -> normalised 3x224x224 tensor.

    X-rays are single-channel, but the ImageNet-pretrained backbone expects 3
    channels, so we copy the gray channel three times (Grayscale(3)).

    Train-time augmentation is deliberately mild and X-ray appropriate:
      * small rotations (patients are never perfectly straight),
      * a slight random zoom/crop (different framing),
      * brightness/contrast jitter (different machines and exposures).
    There is NO horizontal flip: the heart sits on the patient's left, so a
    mirrored chest X-ray is anatomically wrong (it would look like dextrocardia)
    and the L/R side markers would be reversed.
    """
    img, aug = cfg["image"], cfg["augment"]
    normalise = [
        transforms.ToTensor(),  # PIL [0, 255] -> float tensor [0, 1], shape C x H x W
        transforms.Normalize(mean=img["mean"], std=img["std"]),
    ]
    if train:
        steps = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(img["resize"]),
            transforms.RandomRotation(aug["rotation_deg"]),
            transforms.RandomResizedCrop(img["crop"], scale=tuple(aug["crop_scale"]), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
        ]
    else:
        steps = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(img["resize"]),
            transforms.CenterCrop(img["crop"]),
        ]
    return transforms.Compose(steps + normalise)


class ChestXrayDataset(Dataset):
    """Rows of splits.csv -> (image_tensor, label) pairs."""

    def __init__(self, table: pd.DataFrame, root: Path, transform):
        self.paths = table["path"].tolist()
        self.labels = table["label"].tolist()
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        # .convert("L") = 8-bit grayscale; a few files in the dataset are stored as RGB.
        image = Image.open(self.root / self.paths[i]).convert("L")
        return self.transform(image), self.labels[i]


def load_split_table(cfg: dict) -> pd.DataFrame:
    path = splits_csv(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.split` first.")
    return pd.read_csv(path)


def class_weights(labels: list[int], n_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights for the loss: rare class -> bigger weight.

    weight_c = N / (n_classes * count_c). With about 1 NORMAL for every 3
    PNEUMONIA, a NORMAL mistake then costs about 3x more, so the model cannot
    score well by just predicting PNEUMONIA for everything.
    """
    counts = torch.bincount(torch.tensor(labels), minlength=n_classes).float()
    return len(labels) / (n_classes * counts)


def make_loaders(cfg: dict) -> dict:
    """Return {"train": DataLoader, "val": DataLoader, "test": DataLoader}."""
    table = load_split_table(cfg)
    root = data_dir(cfg)
    bs, workers = cfg["train"]["batch_size"], cfg["train"]["num_workers"]

    loaders = {}
    for split in ("train", "val", "test"):
        is_train = split == "train"
        ds = ChestXrayDataset(table[table["split"] == split], root, build_transforms(cfg, train=is_train))
        loaders[split] = DataLoader(ds, batch_size=bs, shuffle=is_train, num_workers=workers)
    return loaders
