"""Out-of-distribution (OOD) guard: "does this even look like a chest X-ray?"

A classifier trained on chest X-rays will still output a confident-looking
number for a cat photo. This guard runs BEFORE we trust any prediction and
has two layers:

Layer 1, image heuristics (cheap, explainable, no model needed):
    * too small                      -> reject
    * clearly coloured               -> reject  (X-rays are grayscale)
    * nearly blank / almost no contrast -> reject
    * mostly white (like a document) or mostly black -> reject
    * unusual shape (aspect ratio)   -> warn
Layer 2, the model's own "energy" score (Liu et al., 2020):
    energy = -logsumexp(logits). Familiar inputs get low (very negative)
    energy; if an image's energy is higher than 99% of our validation X-rays,
    we warn that the model finds it unusual.

Every threshold lives in config.yaml under `ood:` and was set from the real
training images (see `python -m src.ood --stats`).

Honest limit: other grayscale medical images (e.g. a hand X-ray or an MRI
slice) can pass the heuristics; see the measured rates in reports/ood_eval.json.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class OODResult:
    verdict: str                                   # "ok", "warn" or "reject"
    reasons: list[str] = field(default_factory=list)  # plain-language messages shown to the user
    stats: dict = field(default_factory=dict)
    codes: list[str] = field(default_factory=list)    # short name of each check that fired, for counting


def image_stats(image: Image.Image) -> dict:
    """Simple numbers that describe an image. Computed on a 256 px copy for speed."""
    w, h = image.size
    small = image.convert("RGB")
    small.thumbnail((256, 256))
    rgb = np.asarray(small, dtype=np.float32)
    gray = rgb.mean(axis=2)
    colour_diff = (np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])).mean() / 2
    return {
        "width": w,
        "height": h,
        "aspect_ratio": w / h,
        "colour_diff": float(colour_diff),            # 0 for a perfectly gray image (0-255 scale)
        "mean": float(gray.mean()),
        "std": float(gray.std()),                     # contrast
        "frac_white": float((gray > 235).mean()),     # documents and screenshots are mostly white
        "frac_black": float((gray < 20).mean()),
    }


def check_image(image: Image.Image, ocfg: dict) -> OODResult:
    """Layer 1: heuristics on the raw image. `ocfg` is config.yaml's `ood` section."""
    s = image_stats(image)
    rejects, warnings = [], []  # (code, message) pairs

    if min(s["width"], s["height"]) < ocfg["min_side_px"]:
        rejects.append(("too_small", f"The image is very small ({s['width']} x {s['height']} pixels). "
                                     "Please upload the original X-ray file."))
    if s["colour_diff"] > ocfg["max_color_diff"]:
        rejects.append(("colour", "The image is in colour. Chest X-rays are grayscale, so this does not look like an X-ray."))
    if s["std"] < ocfg["min_std"]:
        rejects.append(("blank", "The image is almost blank (very little contrast)."))
    if s["frac_white"] > ocfg["max_frac_white"]:
        rejects.append(("mostly_white", "Most of the image is white, like a document or screenshot, not an X-ray."))
    if s["frac_black"] > ocfg["max_frac_black"]:
        rejects.append(("mostly_black", "Most of the image is black. It may be cropped, inverted or not an X-ray."))
    lo, hi = ocfg["aspect_ratio_range"]
    if not lo <= s["aspect_ratio"] <= hi:
        warnings.append(("unusual_shape", f"The image shape (width/height = {s['aspect_ratio']:.2f}) is unusual for a chest X-ray."))

    verdict = "reject" if rejects else ("warn" if warnings else "ok")
    fired = rejects + warnings
    return OODResult(verdict, [msg for _, msg in fired], s, [code for code, _ in fired])


def energy_score(logits: np.ndarray) -> np.ndarray:
    """Energy = -logsumexp(logits) for each row. Higher = less familiar to the model."""
    z = np.asarray(logits, dtype=np.float64)
    m = z.max(axis=1)
    return -(m + np.log(np.exp(z - m[:, None]).sum(axis=1)))


def check_energy(result: OODResult, logits: np.ndarray, energy_threshold: float | None) -> OODResult:
    """Layer 2: add a warning if the model's energy score is above the validation threshold."""
    if energy_threshold is None or result.verdict == "reject":
        return result
    energy = float(energy_score(np.asarray(logits).reshape(1, -1))[0])
    result.stats["energy"] = energy
    if energy > energy_threshold:
        result.reasons.append("The model finds this image unusual compared with the X-rays it was trained on.")
        result.codes.append("unusual_for_model")
        if result.verdict == "ok":
            result.verdict = "warn"
    return result


# ---------------------------------------------------------------------------
# Command line: measure real X-rays (to set thresholds) and test the guard
# ---------------------------------------------------------------------------
def _percentiles(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {f"p{q:g}": round(float(np.percentile(v, q)), 4) for q in (0.5, 1, 5, 50, 95, 99, 99.5)}


def dataset_stats(cfg: dict) -> dict:
    """Heuristic statistics over all TRAIN images (thresholds are chosen from these)."""
    import pandas as pd

    from src.utils import data_dir, splits_csv

    table = pd.read_csv(splits_csv(cfg))
    paths = [data_dir(cfg) / p for p in table.loc[table["split"] == "train", "path"]]
    stats = [image_stats(Image.open(p)) for p in paths]
    keys = ("aspect_ratio", "colour_diff", "mean", "std", "frac_white", "frac_black")
    summary = {k: _percentiles([s[k] for s in stats]) for k in keys}
    summary["min_side_px"] = _percentiles([min(s["width"], s["height"]) for s in stats])
    summary["n_images"] = len(stats)
    return summary


def evaluate_guard(cfg: dict, sources: dict) -> dict:
    """Run the full guard (heuristics + energy) on each named list of image paths.

    Returns, per source, how many images were passed ("ok"), warned or rejected,
    and how often each check fired ("checks_fired"; one image can fire several).
    """
    import torch

    from src.data import build_transforms
    from src.model import load_checkpoint
    from src.utils import load_json, repo_path

    metrics = load_json(repo_path(cfg["paths"]["reports_dir"]) / "metrics.json")
    energy_threshold = metrics["ood"]["energy_threshold"]
    model, _ = load_checkpoint(repo_path(cfg["paths"]["model"]))
    transform = build_transforms(cfg, train=False)

    results = {}
    for name, paths in sources.items():
        counts = {"ok": 0, "warn": 0, "reject": 0}
        fired = {}
        for path in paths:
            image = Image.open(path)
            result = check_image(image, cfg["ood"])
            if result.verdict != "reject":
                with torch.no_grad():
                    logits = model(transform(image.convert("L")).unsqueeze(0)).numpy()
                result = check_energy(result, logits[0], energy_threshold)
            counts[result.verdict] += 1
            for code in result.codes:
                fired[code] = fired.get(code, 0) + 1
        n = len(paths)
        results[name] = {
            "n": n,
            **counts,
            "flagged_rate": round((counts["warn"] + counts["reject"]) / n, 4) if n else None,
            "reject_rate": round(counts["reject"] / n, 4) if n else None,
            "checks_fired": dict(sorted(fired.items(), key=lambda kv: -kv[1])),
        }
        print(f"{name:<28} n={n:<5} ok={counts['ok']:<5} warn={counts['warn']:<5} reject={counts['reject']:<5}")
    return results


def main() -> None:
    import argparse
    from pathlib import Path

    from src.utils import load_config, repo_path, save_json

    parser = argparse.ArgumentParser(description="Measure X-ray statistics or test the OOD guard.")
    parser.add_argument("--stats", action="store_true", help="print heuristic statistics of the train X-rays")
    parser.add_argument("--eval", nargs="*", metavar="NAME=FOLDER",
                        help="test the guard on folders of images, e.g. photos=/path/to/photos")
    args = parser.parse_args()
    cfg = load_config()

    if args.stats:
        summary = dataset_stats(cfg)
        save_json(summary, repo_path(cfg["paths"]["reports_dir"]) / "ood_xray_stats.json")
        for k, v in summary.items():
            print(k, v)

    if args.eval is not None:
        import pandas as pd

        from src.utils import data_dir, splits_csv

        table = pd.read_csv(splits_csv(cfg))
        # In-distribution control: the real test X-rays should almost all pass.
        sources = {"chest X-rays (test set)": [data_dir(cfg) / p for p in table.loc[table["split"] == "test", "path"]]}
        for item in args.eval:
            name, folder = item.split("=", 1)
            files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            sources[name] = files
        save_json(evaluate_guard(cfg, sources), repo_path(cfg["paths"]["reports_dir"]) / "ood_eval.json")


if __name__ == "__main__":
    main()
