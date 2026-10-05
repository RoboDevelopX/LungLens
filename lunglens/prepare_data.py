"""
prepare_data.py - Step 1 of LungLens: download and unpack the chest X-ray dataset.

The dataset
-----------
"Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for
Classification" by Kermany et al. (Cell, 2018). It contains 5,856 chest
X-rays of children aged 1 to 5 from Guangzhou Women and Children's Medical
Center, each labelled NORMAL or PNEUMONIA by expert physicians.

Google hosts a copy of it (used in the official Keras tutorial) as TFRecord
files, so we can download it without a Kaggle account.

What this script does
---------------------
1. Downloads four files: train/test images and train/test file paths.
2. Reads them record by record. Each "images" record is one JPEG file, and the
   matching "paths" record is its original file name, e.g.
   "./PNEUMONIA/person1_bacteria_1.jpeg". The folder name is the label.
3. Writes every image back out as a normal .jpeg file in:
       data/chest_xray/train/NORMAL/...
       data/chest_xray/train/PNEUMONIA/...
       data/chest_xray/test/NORMAL/...
       data/chest_xray/test/PNEUMONIA/...

Run it once:  python prepare_data.py
"""

import struct
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ChestXRay2017"
RAW_DIR = Path("data/raw")          # where the downloaded TFRecord files go
OUT_DIR = Path("data/chest_xray")   # where the unpacked .jpeg files go


def download(split: str, name: str) -> Path:
    """Download one TFRecord file (skips it if we already have it)."""
    target = RAW_DIR / split / f"{name}.tfrec"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{split}/{name}.tfrec"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, target)
    return target


def read_tfrecord(path: Path):
    """
    Yield the raw bytes of every record in a TFRecord file.

    A TFRecord file is just a list of records stored one after another.
    Each record is laid out as:
        8 bytes  - length of the data (little-endian unsigned integer)
        4 bytes  - checksum of the length (we skip it)
        N bytes  - the data itself
        4 bytes  - checksum of the data (we skip it)
    Reading it ourselves means we do not need to install TensorFlow.
    """
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if len(header) < 12:          # reached the end of the file
                return
            length = struct.unpack("<Q", header[:8])[0]
            data = f.read(length)
            f.read(4)                     # skip the data checksum
            yield data


def unpack(split: str) -> None:
    """Turn one split (train or test) into a folder of labelled .jpeg files."""
    images = read_tfrecord(download(split, "images"))
    paths = read_tfrecord(download(split, "paths"))

    counts = {"NORMAL": 0, "PNEUMONIA": 0}
    # The two files are in the same order, so we can walk them side by side.
    for jpeg_bytes, path_bytes in zip(images, paths):
        original = Path(path_bytes.decode())      # e.g. ./PNEUMONIA/person1_bacteria_1.jpeg
        label = original.parent.name              # "NORMAL" or "PNEUMONIA"
        out = OUT_DIR / split / label / original.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(jpeg_bytes)
        counts[label] += 1

    print(f"{split}: {counts['NORMAL']} normal, {counts['PNEUMONIA']} pneumonia")


if __name__ == "__main__":
    for split in ["train", "test"]:
        unpack(split)
    print(f"Done. Images are in {OUT_DIR}/")
