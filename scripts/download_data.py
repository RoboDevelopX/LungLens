"""Download and extract the Kermany chest X-ray dataset from Kaggle.

Dataset: "Chest X-Ray Images (Pneumonia)", Kaggle id
`paultimothymooney/chest-xray-pneumonia` (Kermany et al., Cell 2018, CC BY 4.0).

Credentials (any ONE of these works):
  1. New-style token:   export KAGGLE_API_TOKEN=...      (kaggle.com > Settings > API > Generate New Token)
  2. Legacy key pair:   export KAGGLE_USERNAME=...  KAGGLE_KEY=...
  3. Legacy file:       ~/.kaggle/kaggle.json  containing {"username": ..., "key": ...}

We call Kaggle's public REST API directly with `requests` (the same endpoint
the official `kaggle` CLI uses), so there is one fewer package to install.

Usage:
    python scripts/download_data.py                 # into data/ (or $LUNGLENS_DATA_DIR)
    python scripts/download_data.py --out /some/dir
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" is importable
from src.utils import data_dir, load_config  # noqa: E402

DATASET = "paultimothymooney/chest-xray-pneumonia"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"

# Only these folders are extracted. The zip also contains a duplicate copy
# (chest_xray/chest_xray/...) and macOS junk (__MACOSX/...), which we skip.
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")


def kaggle_auth() -> dict:
    """Return keyword arguments for requests.get() that authenticate with Kaggle."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return {"headers": {"Authorization": f"Bearer {token}"}}

    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        cred_file = Path.home() / ".kaggle" / "kaggle.json"
        if cred_file.exists():
            creds = json.loads(cred_file.read_text())
            user, key = creds.get("username"), creds.get("key")
    if user and key:
        return {"auth": (user, key)}  # HTTP basic auth, as the legacy API expects

    sys.exit(
        "No Kaggle credentials found. Set KAGGLE_API_TOKEN, or KAGGLE_USERNAME and "
        "KAGGLE_KEY, or create ~/.kaggle/kaggle.json. See the docstring at the top of this file."
    )


def download(zip_path: Path) -> None:
    """Stream the dataset zip to disk (about 2.5 GB) with a simple progress line."""
    print(f"Downloading {DATASET} ...")
    with requests.get(URL, stream=True, timeout=60, **kaggle_auth()) as r:
        if r.status_code == 401 or r.status_code == 403:
            sys.exit(f"Kaggle refused the credentials (HTTP {r.status_code}).")
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB at a time
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()


def extract(zip_path: Path, out_dir: Path) -> dict:
    """Extract chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg into out_dir.

    We build each destination path ourselves from known folder names instead of
    trusting paths inside the zip, so a malformed archive cannot write files
    outside out_dir.
    """
    counts = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            parts = name.split("/")
            # Expect exactly: chest_xray / <split> / <class> / <file>.jpeg
            if len(parts) != 4 or parts[0] != "chest_xray":
                continue
            _, split, cls, fname = parts
            if split not in SPLITS or cls not in CLASSES or not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            dest = out_dir / split / cls / Path(fname).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            counts[f"{split}/{cls}"] = counts.get(f"{split}/{cls}", 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="destination folder (default: data/ or $LUNGLENS_DATA_DIR)")
    parser.add_argument("--zip", type=Path, default=None, help="use an already-downloaded zip instead of downloading")
    parser.add_argument("--keep-zip", action="store_true", help="do not delete the zip after extracting")
    args = parser.parse_args()

    out_dir = args.out or data_dir(load_config())
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = args.zip or out_dir / "chest-xray-pneumonia.zip"
    if args.zip is None:
        download(zip_path)

    print(f"Extracting into {out_dir} ...")
    counts = extract(zip_path, out_dir)
    for key in sorted(counts):
        print(f"  {key:<20} {counts[key]:>5} images")
    print(f"  {'total':<20} {sum(counts.values()):>5} images")

    if args.zip is None and not args.keep_zip:
        zip_path.unlink()


if __name__ == "__main__":
    main()
