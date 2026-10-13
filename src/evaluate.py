"""Final evaluation on the held-out test set, plus all report figures.

What this script does, in order:
  1. Runs the trained model once over our val and test images and saves the
     raw scores to reports/predictions.csv.
  2. Picks the operating threshold on VAL ONLY: the highest threshold that
     still catches >= 95% of pneumonia cases (sensitivity-first, because a
     screening tool should rarely miss a sick patient), capped at 0.5 so an
     X-ray the model rates more likely pneumonia than not is never "normal".
  3. Applies that fixed threshold to TEST and reports accuracy, sensitivity,
     specificity, precision, F1 and AUROC, each with a bootstrap 95% CI.
  4. Writes everything to reports/metrics.json, then draws every figure from
     that file (never from numbers typed by hand).

Usage:
    python -m src.evaluate                  # full evaluation (needs data + models/lunglens.pt)
    python -m src.evaluate --figures-only   # just redraw figures from reports/metrics.json
"""

import argparse
import datetime as dt
import platform

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.data import make_loaders
from src.model import file_fingerprint, load_checkpoint
from src.utils import get_device, load_config, load_json, repo_path, save_json

N_BOOTSTRAP = 2000


# ---------------------------------------------------------------------------
# 1. Running the model
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits N x 2, labels N) for every image in the loader, in order."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        all_logits.append(model(images.to(device)).cpu())
        all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def softmax_pneumonia(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """P(pneumonia) from the two logits. temperature=1 means 'uncalibrated'."""
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)  # subtract the max for numerical stability
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


# ---------------------------------------------------------------------------
# 2. Metrics (positive class = PNEUMONIA = 1)
# ---------------------------------------------------------------------------
def confusion_counts(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = probs >= threshold
    return {
        "tp": int(np.sum(pred & (labels == 1))),   # sick, flagged
        "fn": int(np.sum(~pred & (labels == 1))),  # sick, missed  <- the costly error in screening
        "tn": int(np.sum(~pred & (labels == 0))),  # healthy, cleared
        "fp": int(np.sum(pred & (labels == 0))),   # healthy, flagged (false alarm)
    }


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Threshold-dependent metrics. A zero denominator gives 0.0 instead of a crash."""
    c = confusion_counts(labels, probs, threshold)
    tp, fn, tn, fp = c["tp"], c["fn"], c["tn"], c["fp"]

    def ratio(a, b):
        return a / b if b else 0.0

    sensitivity = ratio(tp, tp + fn)   # a.k.a. recall: share of sick patients we catch
    precision = ratio(tp, tp + fp)     # a.k.a. PPV: share of alarms that are real
    return {
        "accuracy": ratio(tp + tn, len(labels)),
        "sensitivity": sensitivity,
        "specificity": ratio(tn, tn + fp),  # share of healthy patients correctly cleared
        "precision": precision,
        "f1": ratio(2 * precision * sensitivity, precision + sensitivity),
    }


def bootstrap_ci(labels: np.ndarray, probs: np.ndarray, threshold: float, n: int = N_BOOTSTRAP, seed: int = 42) -> dict:
    """95% confidence intervals by the percentile bootstrap.

    We redraw the test set n times WITH replacement (same size), recompute every
    metric on each redraw, and report the 2.5th and 97.5th percentiles. The
    spread shows how much the numbers could move with a different sample of
    patients, which matters with only 624 test images.
    """
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in ("accuracy", "sensitivity", "specificity", "precision", "f1", "auroc")}
    for _ in range(n):
        idx = rng.integers(0, len(labels), size=len(labels))
        y, p = labels[idx], probs[idx]
        if len(np.unique(y)) < 2:  # AUROC needs both classes (practically never happens at n=624)
            continue
        for k, v in binary_metrics(y, p, threshold).items():
            samples[k].append(v)
        samples["auroc"].append(roc_auc_score(y, p))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()}


def threshold_for_sensitivity(labels: np.ndarray, probs: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity on (labels, probs) is still >= target.

    Sort the pneumonia cases' probabilities from high to low; to catch k of
    them we must set the threshold at the k-th highest. We need
    k = ceil(target * number_of_pneumonia_cases).
    """
    positive_probs = np.sort(probs[labels == 1])[::-1]
    k = int(np.ceil(target * len(positive_probs)))
    return float(positive_probs[k - 1])


def choose_threshold(labels: np.ndarray, probs: np.ndarray, target: float, max_threshold: float) -> float:
    """The operating threshold: sensitivity-first, then capped.

    A very confident model can reach the sensitivity target with a threshold
    near 1, which would call an X-ray "normal" even at 80% probability of
    pneumonia. The cap stops that; lowering a threshold can only raise
    sensitivity, so the target still holds.
    """
    return min(threshold_for_sensitivity(labels, probs, target), max_threshold)


def threshold_sweep(labels: np.ndarray, probs: np.ndarray, steps: int = 101) -> list[dict]:
    """Sensitivity/specificity at thresholds 0.00, 0.01, ..., 1.00 (feeds the app's slider)."""
    rows = []
    for t in np.linspace(0, 1, steps):
        m = binary_metrics(labels, probs, t)
        rows.append({"threshold": round(float(t), 2), **{k: round(m[k], 4) for k in ("sensitivity", "specificity", "precision")}})
    return rows


def rounded(d: dict, digits: int = 4) -> dict:
    return {k: round(v, digits) if isinstance(v, float) else v for k, v in d.items()}


def evaluate_at(labels, probs, threshold, seed) -> dict:
    return {
        "threshold": float(threshold),  # full precision: the app must reproduce these decisions exactly
        "metrics": rounded(binary_metrics(labels, probs, threshold)),
        "ci95": {k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in bootstrap_ci(labels, probs, threshold, seed=seed).items()},
        "confusion": confusion_counts(labels, probs, threshold),
    }


# ---------------------------------------------------------------------------
# 3. The full evaluation run
# ---------------------------------------------------------------------------
def run_evaluation(cfg: dict) -> dict:
    device = get_device()
    ckpt_path = repo_path(cfg["paths"]["model"])
    model, ckpt = load_checkpoint(ckpt_path, device)
    loaders = make_loaders(cfg)

    # Run the model once per split and keep the raw scores.
    outputs = {}
    for split in ("val", "test"):
        logits, labels = predict_logits(model, loaders[split], device)
        outputs[split] = (logits, labels)

    table = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": split,
                    "path": loaders[split].dataset.paths,
                    "label": labels,
                    "logit_normal": logits[:, 0].round(5),
                    "logit_pneumonia": logits[:, 1].round(5),
                }
            )
            for split, (logits, labels) in outputs.items()
        ]
    )
    table.to_csv(repo_path(cfg["paths"]["reports_dir"]) / "predictions.csv", index=False)

    val_logits, val_labels = outputs["val"]
    test_logits, test_labels = outputs["test"]
    val_probs = softmax_pneumonia(val_logits)
    test_probs = softmax_pneumonia(test_logits)

    # Operating threshold: chosen on val, then frozen before looking at test.
    target, cap = cfg["decision"]["target_sensitivity"], cfg["decision"]["max_threshold"]
    threshold_for_target = threshold_for_sensitivity(val_labels, val_probs, target)
    threshold = choose_threshold(val_labels, val_probs, target, cap)
    val_at_threshold = binary_metrics(val_labels, val_probs, threshold)

    fpr, tpr, roc_thr = roc_curve(test_labels, test_probs)
    prec, rec, _ = precision_recall_curve(test_labels, test_probs)
    seed = cfg["seed"]

    metrics = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "All numbers computed by src/evaluate.py on the held-out official test set. Nothing is hand-entered.",
        "checkpoint": {
            "path": cfg["paths"]["model"],
            "fingerprint": file_fingerprint(ckpt_path),
            "backbone": ckpt["backbone"],
            "best_stage": ckpt.get("stage"),
            "best_epoch": ckpt.get("epoch"),
            "val_auroc_during_training": round(float(ckpt.get("val_auroc", float("nan"))), 4),
        },
        "data": {
            "n_val": int(len(val_labels)),
            "n_test": int(len(test_labels)),
            "test_normal": int(np.sum(test_labels == 0)),
            "test_pneumonia": int(np.sum(test_labels == 1)),
        },
        "probabilities": {"temperature": 1.0, "calibrated": False},
        "operating_point": {
            "rule": f"highest threshold with validation sensitivity >= {target}, capped at {cap}",
            "target_sensitivity": target,
            "max_threshold": cap,
            "threshold_for_target": float(threshold_for_target),
            "capped": bool(threshold_for_target > cap),
            "threshold": float(threshold),  # full precision: the app must reproduce these decisions exactly
            "val_metrics": rounded(val_at_threshold),
        },
        "test": {
            "auroc": round(float(roc_auc_score(test_labels, test_probs)), 4),
            "auprc": round(float(average_precision_score(test_labels, test_probs)), 4),
            "at_operating_threshold": evaluate_at(test_labels, test_probs, threshold, seed),
            "at_0.5": evaluate_at(test_labels, test_probs, 0.5, seed),
            "roc_curve": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
            "pr_curve": {"precision": prec.round(4).tolist(), "recall": rec.round(4).tolist()},
            "threshold_sweep": threshold_sweep(test_labels, test_probs),
        },
        "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": seed, "method": "percentile, resampling test images with replacement"},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
    }
    save_json(metrics, repo_path(cfg["paths"]["reports_dir"]) / "metrics.json")
    return metrics


def print_summary(metrics: dict) -> None:
    op = metrics["test"]["at_operating_threshold"]
    print(f"checkpoint {metrics['checkpoint']['fingerprint']}  ({metrics['checkpoint']['backbone']})")
    opp = metrics["operating_point"]
    print(f"threshold for {opp['target_sensitivity']:.0%} val sensitivity: {opp['threshold_for_target']:.4f}; "
          f"operating threshold (after the {opp['max_threshold']} cap): {opp['threshold']:.4f}"
          f"  -> val sensitivity {opp['val_metrics']['sensitivity']:.3f}, specificity {opp['val_metrics']['specificity']:.3f}")
    print(f"TEST (n={metrics['data']['n_test']}) at the operating threshold:")
    for k in ("accuracy", "sensitivity", "specificity", "precision", "f1", "auroc"):
        value = metrics["test"]["auroc"] if k == "auroc" else op["metrics"][k]
        lo, hi = op["ci95"][k]
        print(f"  {k:<12} {value:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  confusion    {op['confusion']}")
    half = metrics["test"]["at_0.5"]
    print(f"for comparison at threshold 0.5: sensitivity {half['metrics']['sensitivity']:.3f}, "
          f"specificity {half['metrics']['specificity']:.3f}, accuracy {half['metrics']['accuracy']:.3f}")


# ---------------------------------------------------------------------------
# 4. Figures: drawn ONLY from reports/metrics.json (+ train_history.json)
# ---------------------------------------------------------------------------
# Calm, colour-blind-checked palette shared with the app.
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def make_figures(metrics: dict, history: dict | None, out_dir) -> list:
    import matplotlib

    matplotlib.use("Agg")  # draw to files, no screen needed
    import matplotlib.pyplot as plt

    out_dir = repo_path(out_dir) if isinstance(out_dir, str) else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    test = metrics["test"]
    op = test["at_operating_threshold"]

    def save(fig, name):
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=160, facecolor=SURFACE)
        plt.close(fig)
        saved.append(path)

    # ROC curve with the operating point marked.
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    ax.plot(test["roc_curve"]["fpr"], test["roc_curve"]["tpr"], color=BLUE, linewidth=2, solid_capstyle="round")
    sens, spec = op["metrics"]["sensitivity"], op["metrics"]["specificity"]
    ax.plot([1 - spec], [sens], "o", markersize=8, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"operating point\nsens {sens:.2f}, spec {spec:.2f}", (1 - spec, sens), xytext=(12, -34),
                textcoords="offset points", fontsize=8.5, color=INK_2)
    lo, hi = op["ci95"]["auroc"]
    ax.set_title(f"ROC curve, test set (AUROC {test['auroc']:.3f}, 95% CI {lo:.3f}-{hi:.3f})", fontsize=9.5)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    save(fig, "roc_curve.png")

    # Precision-recall curve with the no-skill baseline (= share of pneumonia in test).
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    _style_axes(ax)
    prevalence = metrics["data"]["test_pneumonia"] / metrics["data"]["n_test"]
    ax.axhline(prevalence, color=AXIS, linewidth=1)
    ax.text(0.02, prevalence + 0.015, f"no-skill baseline ({prevalence:.2f})", fontsize=8, color=MUTED)
    ax.plot(test["pr_curve"]["recall"], test["pr_curve"]["precision"], color=BLUE, linewidth=2)
    ax.set_title(f"Precision-recall curve, test set (AUPRC {test['auprc']:.3f})", fontsize=9.5)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1.02)
    save(fig, "pr_curve.png")

    # Confusion matrix at the operating threshold.
    c = op["confusion"]
    grid = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(grid, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#184f95"]))
    names = [["true negative", "false positive"], ["false negative", "true positive"]]
    for i in range(2):
        for j in range(2):
            dark_cell = grid[i, j] > grid.max() / 2
            ax.text(j, i, f"{grid[i, j]}\n{names[i][j]}", ha="center", va="center", fontsize=10,
                    color="white" if dark_cell else INK)
    ax.set_xticks([0, 1], ["Normal", "Pneumonia"])
    ax.set_yticks([0, 1], ["Normal", "Pneumonia"])
    ax.set_xlabel("Predicted", color=INK_2)
    ax.set_ylabel("Actual", color=INK_2)
    ax.tick_params(colors=INK_2, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Confusion matrix, test set (threshold {op['threshold']:.2f})", fontsize=9.5, color=INK)
    save(fig, "confusion_matrix.png")

    # Sensitivity / specificity trade-off across thresholds.
    sweep = pd.DataFrame(test["threshold_sweep"])
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    _style_axes(ax)
    ax.plot(sweep["threshold"], sweep["sensitivity"], color=BLUE, linewidth=2, label="Sensitivity")
    ax.plot(sweep["threshold"], sweep["specificity"], color=ORANGE, linewidth=2, label="Specificity")
    ax.axvline(op["threshold"], color=INK_2, linewidth=1)
    ax.text(op["threshold"] - 0.015, 0.72, "operating\nthreshold", fontsize=8, color=INK_2, ha="right")
    ax.set_xlabel("Decision threshold on P(pneumonia)")
    ax.set_ylabel("Rate on test set")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=2, labelcolor=INK_2)
    ax.set_title("Sensitivity vs specificity trade-off", fontsize=9.5)
    save(fig, "threshold_tradeoff.png")

    # Training curves (validation AUROC and losses per epoch).
    if history and history.get("epochs"):
        h = pd.DataFrame(history["epochs"])
        h["n"] = np.arange(1, len(h) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
        for ax in axes:
            _style_axes(ax)
        axes[0].plot(h["n"], h["train_loss"], color=BLUE, linewidth=2, marker="o", markersize=4, label="Train loss")
        axes[0].plot(h["n"], h["val_loss"], color=ORANGE, linewidth=2, marker="o", markersize=4, label="Val loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_title("Loss (weighted cross-entropy)", fontsize=9.5)
        axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2)
        axes[1].plot(h["n"], h["val_auroc"], color=BLUE, linewidth=2, marker="o", markersize=4)
        best = h["val_auroc"].idxmax()
        axes[1].plot([h.loc[best, "n"]], [h.loc[best, "val_auroc"]], "o", markersize=8, color=BLUE,
                     markeredgecolor=SURFACE, markeredgewidth=2)
        axes[1].annotate(f"best {h.loc[best, 'val_auroc']:.3f}", (h.loc[best, "n"], h.loc[best, "val_auroc"]),
                         xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK_2)
        low, high = h["val_auroc"].min(), h["val_auroc"].max()
        axes[1].set_ylim(low - 0.05 * (high - low), high + 0.25 * (high - low))  # headroom for the label
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Validation AUROC", fontsize=9.5)
        n_head = int((h["stage"] == "head").sum())
        for ax in axes:
            if 0 < n_head < len(h):
                ax.axvline(n_head + 0.5, color=AXIS, linewidth=1)
        save(fig, "training_curves.png")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LungLens on the held-out test set.")
    parser.add_argument("--figures-only", action="store_true", help="redraw figures from reports/metrics.json")
    args = parser.parse_args()

    cfg = load_config()
    reports = repo_path(cfg["paths"]["reports_dir"])
    if args.figures_only:
        metrics = load_json(reports / "metrics.json")
    else:
        metrics = run_evaluation(cfg)
    print_summary(metrics)

    history_path = reports / "train_history.json"
    history = load_json(history_path) if history_path.exists() else None
    for path in make_figures(metrics, history, cfg["paths"]["figures_dir"]):
        print(f"figure: {path.relative_to(repo_path('.'))}")


if __name__ == "__main__":
    main()
