"""Sanity checks for the Grad-CAM explanations -> reports/sanity.json + figures.

Three checks, all on the held-out TEST images:
  1. Model randomisation (Adebayo et al., 2018): randomise the final layer, then
     also the last dense block, and measure how much the heatmaps change.
  2. Example heatmaps for true/false positives/negatives.
  3. Border attention: how much heatmap weight sits in the outer image band,
     where text markers and borders live (a sign of shortcut learning).

Run after `python -m src.evaluate` (it needs the calibrated threshold):
    python scripts/sanity_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" is importable
from src.calibration import softmax_probs  # noqa: E402
from src.data import build_transforms  # noqa: E402
from src.gradcam import GradCAM, border_fraction, overlay, randomized_copy, rank_correlation, target_layer_for  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.utils import data_dir, load_config, load_json, repo_path, save_json  # noqa: E402

N_RANDOMIZATION = 50
BAND = 0.10


def p_text(p: float) -> str:
    """P(pneumonia) for figure titles, never "1.00" or "0.00": the model is never certain."""
    return "P(pneumonia) > 0.99" if p >= 0.995 else ("P(pneumonia) < 0.01" if p < 0.005 else f"P(pneumonia) = {p:.2f}")


def verdict_randomization(mean_corr: float) -> str:
    if mean_corr < 0.3:
        return (f"Pass. With randomised weights the heatmaps change almost completely (mean correlation "
                f"{mean_corr:.2f}), so Grad-CAM reflects what the model learned, not just image edges.")
    if mean_corr < 0.6:
        return (f"Partial pass. Randomising the weights changes the heatmaps substantially (mean correlation "
                f"{mean_corr:.2f}) but some structure remains, partly from the image itself.")
    return (f"Concern. Heatmaps stay similar after randomisation (mean correlation {mean_corr:.2f}); "
            "they may reflect image structure more than the learned model.")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    reports = repo_path(cfg["paths"]["reports_dir"])
    figures = repo_path(cfg["paths"]["figures_dir"])
    metrics = load_json(reports / "metrics.json")
    temperature = metrics["calibration"]["temperature"]
    threshold = metrics["operating_point"]["threshold"]
    border_warn = cfg["explain"]["border_warn"]

    model, _ = load_checkpoint(repo_path(cfg["paths"]["model"]))
    # Randomised copies are made BEFORE any hook is attached to the trained model.
    rand_classifier = randomized_copy(model, seed=1, layers="classifier")
    rand_top = randomized_copy(model, seed=1, layers="top")
    cams = {name: GradCAM(m, target_layer_for(m)) for name, m in
            (("trained", model), ("classifier", rand_classifier), ("top", rand_top))}

    transform = build_transforms(cfg, train=False)
    display = build_transforms(cfg, train=False).transforms[:3]  # Grayscale, Resize, CenterCrop (no tensor)

    preds = pd.read_csv(reports / "predictions.csv")
    test = preds[preds["split"] == "test"].reset_index(drop=True)
    test["p"] = softmax_probs(test[["logit_normal", "logit_pneumonia"]].to_numpy(), temperature)[:, 1]
    test["pred"] = (test["p"] >= threshold).astype(int)
    test["outcome"] = np.select(
        [(test.label == 1) & (test.pred == 1), (test.label == 0) & (test.pred == 0),
         (test.label == 0) & (test.pred == 1), (test.label == 1) & (test.pred == 0)],
        ["true positive", "true negative", "false positive", "false negative"],
        default="",  # never used: the four cases cover every image (a string default keeps numpy happy)
    )

    def load(path):
        image = Image.open(data_dir(cfg) / path).convert("L")
        view = image
        for step in display:
            view = step(view)
        return transform(image).unsqueeze(0), view

    # --- Check 3 (and cache): trained-model heatmap + border share for EVERY test image ---------
    print(f"computing Grad-CAM for {len(test)} test images ...", flush=True)
    trained_maps, shares = {}, []
    for i, path in enumerate(test["path"]):
        x, _ = load(path)
        cam, _ = cams["trained"](x, class_idx=1)
        trained_maps[i] = cam
        shares.append(border_fraction(cam, BAND))
    test["border_share"] = shares

    uniform = 1 - (1 - 2 * BAND) ** 2
    above = test["border_share"] > border_warn
    by_outcome = test.groupby("outcome")["border_share"].median().round(3).to_dict()
    median = float(test["border_share"].median())
    if median > uniform:
        border_verdict = (f"Warning sign. The median border share ({100 * median:.0f}%) is above the {100 * uniform:.0f}% "
                          "expected from evenly spread attention, so the model leans on the image edges more than it should.")
    else:
        border_verdict = (f"Mostly reassuring. The median border share ({100 * median:.0f}%) is below the "
                          f"{100 * uniform:.0f}% expected from evenly spread attention, so most heatmaps concentrate "
                          f"inside the image. {int(above.sum())} test X-rays ({100 * above.mean():.1f}%) still put more "
                          f"than {100 * border_warn:.0f}% of their attention on the edges, and the app adds a caution "
                          "message for results like those.")
    border = {
        "band": BAND,
        "uniform_share": round(uniform, 4),
        "n_images": int(len(test)),
        "median": round(median, 4),
        "mean": round(float(test["border_share"].mean()), 4),
        "warn": border_warn,
        "n_above_warn": int(above.sum()),
        "share_above_warn": round(float(above.mean()), 4),
        "median_by_outcome": by_outcome,
        "verdict": border_verdict,
    }

    # --- Check 1: randomisation on a fixed random subset ------------------------------------------
    # Only images whose trained heatmap is not flat: an all-zero map (no evidence for pneumonia at all,
    # common for clear normals) has no pixel ranking to compare with.
    rng = np.random.default_rng(cfg["seed"])
    eligible = [i for i in range(len(test)) if trained_maps[i].max() > trained_maps[i].min()]
    subset = rng.choice(eligible, size=min(N_RANDOMIZATION, len(eligible)), replace=False)
    corr = {"classifier": [], "top": []}
    for i in subset:
        x, _ = load(test.loc[i, "path"])
        for name in ("classifier", "top"):
            cam, _ = cams[name](x, class_idx=1)
            corr[name].append(rank_correlation(trained_maps[i], cam))

    def summary(values):
        values = np.asarray(values)
        # A constant (e.g. all-zero) randomised map has no ranking: Spearman is undefined (NaN).
        # We count it as 0 ("the heatmap changed completely") and report how often that happened.
        v = np.nan_to_num(values, nan=0.0)
        return {"mean": round(float(v.mean()), 4), "p5": round(float(np.percentile(v, 5)), 4),
                "p95": round(float(np.percentile(v, 95)), 4), "n_constant_maps": int(np.isnan(values).sum())}

    randomization = {"n_images": int(len(subset)), "n_flat_trained_maps_skipped": int(len(test) - len(eligible)),
                     "classifier": summary(corr["classifier"]), "top": summary(corr["top"])}
    randomization["verdict"] = verdict_randomization(randomization["top"]["mean"])

    save_json({"randomization": randomization, "border": border}, reports / "sanity.json")
    print("randomization:", randomization)
    print("border:", {k: v for k, v in border.items() if k != "verdict"})

    # --- Figures -----------------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})

    def show(ax, img, title):
        ax.imshow(img, cmap="gray" if img.mode == "L" else None)
        ax.set_title(title, color="#0b0b0b")
        ax.axis("off")

    # Randomisation figure: 3 example rows x 4 columns.
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for row, i in enumerate(subset[:3]):
        x, view = load(test.loc[i, "path"])
        show(axes[row, 0], view, f"X-ray ({'pneumonia' if test.loc[i, 'label'] else 'normal'})")
        show(axes[row, 1], overlay(view, trained_maps[i], 0.6), "trained model")
        for col, name, title in ((2, "classifier", "final layer randomised"), (3, "top", "+ last block randomised")):
            cam, _ = cams[name](x, class_idx=1)
            show(axes[row, col], overlay(view, cam, 0.6), title)
    fig.tight_layout()
    fig.savefig(figures / "sanity_randomization.png", dpi=130, facecolor="white")
    plt.close(fig)

    # Correct vs incorrect examples: up to 3 per outcome, the most confident of each kind.
    outcomes = ["true positive", "true negative", "false positive", "false negative"]
    fig, axes = plt.subplots(len(outcomes), 3, figsize=(8, 11))
    for r, outcome in enumerate(outcomes):
        rows = test[test["outcome"] == outcome]
        rows = rows.sort_values("p", ascending=outcome in ("true negative", "false negative"))
        for c in range(3):
            ax = axes[r, c]
            ax.axis("off")
            if c == 0 and len(rows) == 0:
                ax.text(0.0, 0.5, f"no {outcome}s on the test set", fontsize=9, color="#52514e")
            if c < len(rows):
                i = rows.index[c]
                _, view = load(test.loc[i, "path"])
                show(ax, overlay(view, trained_maps[i], 0.6), f"{outcome}\n{p_text(test.loc[i, 'p'])}")
    fig.tight_layout()
    fig.savefig(figures / "sanity_examples.png", dpi=130, facecolor="white")
    plt.close(fig)

    # Highest border attention: the 6 most edge-focused heatmaps.
    top_border = test.sort_values("border_share", ascending=False).head(6)
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    for ax, (i, row) in zip(axes.ravel(), top_border.iterrows()):
        _, view = load(row["path"])
        show(ax, overlay(view, trained_maps[i], 0.6),
             f"{row['outcome']}\nborder share {100 * row['border_share']:.0f}%")
    fig.tight_layout()
    fig.savefig(figures / "border_attention.png", dpi=130, facecolor="white")
    plt.close(fig)
    # One clean example for the one-pager: a confident true positive whose heatmap stays inside the image.
    good = test[(test["outcome"] == "true positive") & (test["border_share"] < uniform)].sort_values("p", ascending=False)
    if len(good):
        i = good.index[0]
        _, view = load(test.loc[i, "path"])
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        show(axes[0], view, "Chest X-ray (test set)")
        show(axes[1], overlay(view, trained_maps[i], 0.6), f"Grad-CAM, {p_text(test.loc[i, 'p'])}")
        fig.tight_layout()
        fig.savefig(figures / "onepager_example.png", dpi=150, facecolor="white")
        plt.close(fig)
    print("figures: sanity_randomization.png, sanity_examples.png, border_attention.png, onepager_example.png")


if __name__ == "__main__":
    main()
