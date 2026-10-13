"""One X-ray in, one explained screening result out.

This is the pipeline the app runs for every uploaded image:

    1. OOD guard, layer 1 (image heuristics)  -> may stop here with "not an X-ray"
    2. Model forward pass                      -> 2 logits
    3. OOD guard, layer 2 (energy score)       -> may add a warning
    4. Temperature scaling                     -> calibrated P(pneumonia)
    5. Decision with an "Inconclusive" band    -> normal / inconclusive / pneumonia
    6. Grad-CAM for the PNEUMONIA class        -> heatmap
    7. Plain-language explanation

It has no Streamlit code, so it can be tested and reused from the command line:
    python -m src.predict path/to/xray.jpeg
"""

import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.calibration import softmax_probs
from src.data import build_transforms
from src.explain import Explanation, band_from_config, explain
from src.gradcam import GradCAM, border_fraction, target_layer_for
from src.model import LungLensNet, file_fingerprint, load_checkpoint
from src.ood import OODResult, check_energy, check_image
from src.utils import load_config, load_json, repo_path

DISPLAY_SIZE = 896  # the heatmap is shown on a sharper copy of exactly the region the model saw


def download_weights_if_missing(path) -> None:
    """Cloud deployments: models/ is not in git, so fetch the weights once from LUNGLENS_WEIGHTS_URL
    (for example the file attached to a GitHub release). Does nothing if the file exists or no URL is set.
    The app still compares the file's fingerprint with reports/metrics.json and warns on a mismatch."""
    import os

    url = os.environ.get("LUNGLENS_WEIGHTS_URL")
    if path.exists() or not url:
        return
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(partial, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    partial.rename(path)  # only a complete download gets the real name


@dataclass
class Prediction:
    ood: OODResult
    probability: float | None = None          # calibrated P(pneumonia)
    explanation: Explanation | None = None
    cam: np.ndarray | None = None             # DISPLAY_SIZE x DISPLAY_SIZE, values in [0, 1]
    model_view: Image.Image | None = None     # the centre-cropped X-ray the model actually saw
    border_share: float | None = None
    seconds: float = 0.0


class LungLensPredictor:
    def __init__(self, cfg: dict, model: LungLensNet, metrics: dict | None, demo_mode: bool = False):
        self.cfg = cfg
        self.model = model.eval()
        self.demo_mode = demo_mode
        self.transform = build_transforms(cfg, train=False)
        # Same geometry as the model's Resize(256) + CenterCrop(224), just 4x larger for display.
        scale = DISPLAY_SIZE / cfg["image"]["crop"]
        self.display_transform = transforms.Compose(
            [transforms.Resize(round(cfg["image"]["resize"] * scale)), transforms.CenterCrop(DISPLAY_SIZE)]
        )
        self.cam = GradCAM(self.model, target_layer_for(self.model))
        # One model is shared by every visitor of the app. Grad-CAM stores the last
        # feature map on the object, so we let one request use the model at a time.
        self._lock = threading.Lock()
        self.fingerprint_ok = True  # set by from_files(): do weights match reports/metrics.json?

        # Settings fitted on validation data by src/evaluate.py (defaults only in demo mode).
        if metrics is not None:
            self.temperature = metrics.get("calibration", {}).get("temperature", 1.0)
            self.threshold = metrics["operating_point"]["threshold"]
            self.energy_threshold = metrics.get("ood", {}).get("energy_threshold")
        else:
            self.temperature, self.threshold, self.energy_threshold = 1.0, 0.5, None
        decision = (metrics or {}).get("decision")
        if decision:  # use exactly the band that was evaluated on the test set
            self.low, self.high = decision["low"], decision["high"]
        else:
            self.low, self.high = band_from_config(self.threshold, cfg["decision"])

    @classmethod
    def from_files(cls, cfg: dict | None = None, force_demo: bool = False) -> "LungLensPredictor":
        """Load models/lunglens.pt + reports/metrics.json, or fall back to DEMO_MODE."""
        cfg = cfg or load_config()
        ckpt_path = repo_path(cfg["paths"]["model"])
        metrics_path = repo_path(cfg["paths"]["reports_dir"]) / "metrics.json"
        if not force_demo:
            download_weights_if_missing(ckpt_path)
        if force_demo or not ckpt_path.exists() or not metrics_path.exists():
            # DEMO_MODE: an UNTRAINED network, only so the interface can be developed
            # without the weights. Its outputs are meaningless and the app says so.
            torch.manual_seed(0)
            return cls(cfg, LungLensNet(cfg["model"]["backbone"], pretrained=False), None, demo_mode=True)
        model, _ = load_checkpoint(ckpt_path)
        predictor = cls(cfg, model, load_json(metrics_path))
        predictor.fingerprint_ok = file_fingerprint(ckpt_path) == load_json(metrics_path)["checkpoint"]["fingerprint"]
        return predictor

    def predict(self, image: Image.Image) -> Prediction:
        start = time.perf_counter()

        # 1. Heuristic guard on the raw upload.
        ood = check_image(image, self.cfg["ood"])
        if ood.verdict == "reject":
            return Prediction(ood=ood, seconds=time.perf_counter() - start)

        gray = image.convert("L")
        x = self.transform(gray).unsqueeze(0)                     # 1 x 3 x 224 x 224

        # 2 + 6. One forward/backward pass gives both the logits and the Grad-CAM map.
        with self._lock:
            cam_224, logits = self.cam(x, class_idx=1)            # explain the PNEUMONIA score
        logits = logits.numpy()

        # 3. Energy-based guard (adds a warning, never blocks).
        ood = check_energy(ood, logits[0], self.energy_threshold)

        # 4. Calibrated probability.
        p = float(softmax_probs(logits, self.temperature)[0, 1])

        # Heatmap at display resolution, on exactly the region the model saw.
        cam = np.asarray(Image.fromarray(cam_224.astype(np.float32)).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR))
        cam = np.clip(cam, 0, 1)
        border = border_fraction(cam_224)

        # 5 + 7. Decision and explanation.
        text = explain(
            p,
            self.low,
            self.high,
            cam=cam_224,
            border_share=border,
            border_warn=self.cfg["explain"]["border_warn"],
            temperature=self.temperature,
            ood_reasons=ood.reasons,
        )
        return Prediction(
            ood=ood,
            probability=p,
            explanation=text,
            cam=cam,
            model_view=self.display_transform(gray),
            border_share=border,
            seconds=time.perf_counter() - start,
        )


def benchmark(predictor: LungLensPredictor, image: Image.Image, runs: int) -> dict:
    """Time the full pipeline (guard + model + Grad-CAM + explanation) on one image, after one warm-up run."""
    import platform

    predictor.predict(image)  # warm-up: the first call pays one-off start-up costs
    seconds = [predictor.predict(image).seconds for _ in range(runs)]
    return {
        "runs": runs,
        "median_s": round(float(np.median(seconds)), 3),
        "max_s": round(float(np.max(seconds)), 3),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "machine": f"{platform.machine()}, {platform.system()}, Python {platform.python_version()}, torch {torch.__version__}",
        "demo_mode": predictor.demo_mode,
    }


def main() -> None:
    import argparse

    from src.utils import save_json

    parser = argparse.ArgumentParser(description="Screen one chest X-ray from the command line.")
    parser.add_argument("image")
    parser.add_argument("--demo", action="store_true", help="force DEMO_MODE (untrained model)")
    parser.add_argument("--benchmark", type=int, metavar="RUNS",
                        help="time RUNS predictions on this image and save reports/timing.json")
    args = parser.parse_args()

    predictor = LungLensPredictor.from_files(force_demo=args.demo)
    if args.benchmark:
        timing = benchmark(predictor, Image.open(args.image), args.benchmark)
        save_json(timing, repo_path(predictor.cfg["paths"]["reports_dir"]) / "timing.json")
        print(timing)
        return
    result = predictor.predict(Image.open(args.image))
    print(f"input check: {result.ood.verdict}  {result.ood.reasons}")
    if result.explanation:
        print(f"P(pneumonia) = {result.probability:.3f}  ->  {result.explanation.decision}")
        print(result.explanation.headline)
        for line in result.explanation.clinician:
            print("  -", line)
    print(f"time: {result.seconds:.2f} s" + ("   [DEMO_MODE: untrained model]" if predictor.demo_mode else ""))


if __name__ == "__main__":
    main()
