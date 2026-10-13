# LungLens: See what the AI sees

Explainable chest X-ray screening for earlier pneumonia detection, built for the
UnivaBio student hackathon (theme: Artificial Intelligence for Human Health).

> **Educational screening aid, not a diagnostic device.** Work in progress:
> this README is completed in phase (f) with results, screenshots and the
> architecture diagram. No metrics are reported until a real test-set run exists.

## Quickstart (so far)

```bash
pip install -r requirements.txt          # CPU torch: add --index-url https://download.pytorch.org/whl/cpu
python scripts/download_data.py          # needs Kaggle credentials, see below
python -m src.split                      # patient-grouped train/val split -> reports/splits.csv
```

### Kaggle credentials (any one)

1. `export KAGGLE_API_TOKEN=...` (kaggle.com > Settings > API > Generate New Token), or
2. `export KAGGLE_USERNAME=... KAGGLE_KEY=...`, or
3. `~/.kaggle/kaggle.json` with `{"username": "...", "key": "..."}`.

Set `LUNGLENS_DATA_DIR` to keep the 1.2 GB of images outside the repo.

## Dataset attribution

Kermany, D., Zhang, K., Goldbaum, M. (2018). *Labeled Optical Coherence Tomography (OCT)
and Chest X-Ray Images for Classification*, Mendeley Data, v2, doi:10.17632/rscbjbr9sj.2.
Licensed CC BY 4.0. Kaggle mirror: `paultimothymooney/chest-xray-pneumonia`.

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0, not included in this repository.
