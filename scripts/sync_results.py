"""Copy the latest numbers from reports/*.json into README.md and docs/WALKTHROUGH.md.

Two generated blocks, so the documents can never drift from the real run:
  * <!-- RESULTS:START --> ... <!-- RESULTS:END -->  (README and WALKTHROUGH):
    a results table built from reports/metrics.json (+ sanity/OOD reports).
  * <!-- QA:START --> ... <!-- QA:END -->  (WALKTHROUGH): the judge questions,
    rendered from docs/templates/judge_qa.md, where every {{name}} is a number
    computed by doc_numbers() below.

Usage (after evaluation, the OOD evaluation, the sanity check and the benchmark):
    python scripts/sync_results.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import DISPLAY_NAMES  # noqa: E402
from src.utils import load_json  # noqa: E402


def results_markdown() -> str:
    m = load_json(ROOT / "reports" / "metrics.json")
    op = m["test"]["at_operating_threshold"]
    half = m["test"]["at_0.5"]
    names = [("sensitivity", "Sensitivity (recall)"), ("specificity", "Specificity"), ("precision", "Precision (PPV)"),
             ("f1", "F1"), ("accuracy", "Accuracy")]
    same = abs(op["threshold"] - 0.5) < 1e-9  # then a separate "at 0.5" column would only repeat the numbers
    lines = [
        f"Held-out official test set: **{m['data']['n_test']} X-rays** ({m['data']['test_normal']} normal, "
        f"{m['data']['test_pneumonia']} pneumonia), evaluated once. 95% CIs from {m['bootstrap']['n_resamples']:,} "
        f"bootstrap resamples. Model `{m['checkpoint']['fingerprint']}` ({DISPLAY_NAMES.get(m['checkpoint']['backbone'], m['checkpoint']['backbone'])}).",
        "",
        f"| Metric | At the operating threshold ({op['threshold']:.2f}) | 95% CI |" + ("" if same else " At threshold 0.5 |"),
        "|---|---|---|" + ("" if same else "---|"),
    ]
    for key, label in names:
        lo, hi = op["ci95"][key]
        lines.append(f"| {label} | **{op['metrics'][key]:.3f}** | {lo:.3f} to {hi:.3f} |"
                     + ("" if same else f" {half['metrics'][key]:.3f} |"))
    lo, hi = op["ci95"]["auroc"]
    lines.append(f"| AUROC | **{m['test']['auroc']:.3f}** | {lo:.3f} to {hi:.3f} |" + ("" if same else " (threshold-free) |"))
    c = op["confusion"]
    lines += [
        "",
        f"- **Confusion matrix** at the operating threshold: {c['tp']} true positives, {c['fn']} missed pneumonia, "
        f"{c['tn']} true negatives, {c['fp']} false alarms.",
    ]
    cal, d = m.get("calibration"), m.get("decision")
    if cal:
        lines.append(f"- **Calibration** (ECE, 15 bins; 0 is perfect): validation {cal['val_ece_before']:.3f} -> "
                     f"{cal['val_ece_after']:.3f}, test {cal['test_ece_before']:.3f} -> {cal['test_ece_after']:.3f} "
                     f"after temperature scaling (T = {cal['temperature']:.2f}, fitted on validation).")
    if d:
        lines.append(f"- **Inconclusive zone** [{d['low']:.2f}, {d['high']:.2f}]: {100 * d['test_inconclusive_rate']:.1f}% "
                     f"of test X-rays; accuracy on the remaining ones {d['test_accuracy_when_decided']:.3f}.")
    sanity = ROOT / "reports" / "sanity.json"
    if sanity.exists():
        s = load_json(sanity)
        lines.append(f"- **Grad-CAM model randomisation:** {s['randomization']['verdict']}")
        lines.append(f"- **Grad-CAM attention on the image edges:** {s['border']['verdict']}")
    ood_eval = ROOT / "reports" / "ood_eval.json"
    if ood_eval.exists():
        lines += ["", "Input check, share of images warned or refused:", ""]
        for name, r in load_json(ood_eval).items():
            lines.append(f"- {name}: {100 * r['flagged_rate']:.1f}% of {r['n']} ({r['reject']} refused, {r['warn']} warned)")
    lines += ["", "![ROC curve](reports/figures/roc_curve.png) ![Confusion matrix](reports/figures/confusion_matrix.png)"]
    return "\n".join(lines)


def replace_block(path: Path, marker: str, content: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"(<!-- {marker}:START -->)(.*?)(<!-- {marker}:END -->)", re.S)
    if not pattern.search(text):
        print(f"skip {path.name}: no {marker} markers")
        return
    path.write_text(pattern.sub(lambda mt: f"{mt.group(1)}\n{content}\n{mt.group(3)}", text), encoding="utf-8")
    print(f"updated {path.relative_to(ROOT)}")


def ci(pair) -> str:
    return f"{pair[0]:.3f} to {pair[1]:.3f}"


def doc_numbers() -> dict:
    """Every number the judge Q&A quotes, already formatted, read from the report files."""
    reports = ROOT / "reports"
    m = load_json(reports / "metrics.json")
    op, half = m["test"]["at_operating_threshold"], m["test"]["at_0.5"]
    k, c = op["metrics"], op["confusion"]
    splits = pd.read_csv(reports / "splits.csv")
    history = load_json(reports / "train_history.json")["epochs"]
    v = {
        "n_train": f"{(splits['split'] == 'train').sum():,}", "n_val": f"{(splits['split'] == 'val').sum():,}",
        "n_test": m["data"]["n_test"], "test_normal": m["data"]["test_normal"], "test_pneumonia": m["data"]["test_pneumonia"],
        "threshold": f"{op['threshold']:.3f}", "target_sensitivity": f"{m['operating_point']['target_sensitivity']:.0%}",
        "threshold_for_target": f"{m['operating_point']['threshold_for_target']:.3f}",
        "max_threshold": f"{m['operating_point']['max_threshold']:.2f}",
        "val_sens": f"{m['operating_point']['val_metrics']['sensitivity']:.3f}",
        "val_spec": f"{m['operating_point']['val_metrics']['specificity']:.3f}",
        "auroc": f"{m['test']['auroc']:.3f}", "auroc_ci": ci(op["ci95"]["auroc"]), "auprc": f"{m['test']['auprc']:.3f}",
        "tp": c["tp"], "fn": c["fn"], "tn": c["tn"], "fp": c["fp"],
        "sens_05": f"{half['metrics']['sensitivity']:.3f}", "spec_05": f"{half['metrics']['specificity']:.3f}",
        "fn_05": half["confusion"]["fn"], "fp_05": half["confusion"]["fp"],
        "backbone": DISPLAY_NAMES.get(m["checkpoint"]["backbone"], m["checkpoint"]["backbone"]), "best_stage": m["checkpoint"]["best_stage"],
        "best_epoch": m["checkpoint"]["best_epoch"], "n_epochs": len(history),
        "train_minutes": f"{sum(e['seconds_total'] for e in history) / 60:.0f}",
        "fingerprint": m["checkpoint"]["fingerprint"], "n_bootstrap": f"{m['bootstrap']['n_resamples']:,}",
    }
    for key in ("sensitivity", "specificity", "precision", "f1", "accuracy"):
        short = {"sensitivity": "sens", "specificity": "spec", "precision": "prec"}.get(key, key)
        v[short], v[f"{short}_ci"] = f"{k[key]:.3f}", ci(op["ci95"][key])
    if "calibration" in m:
        cal, d, ood = m["calibration"], m["decision"], m["ood"]
        # Per-image calibrated probabilities on test, recomputed from the saved logits.
        preds = pd.read_csv(reports / "predictions.csv")
        test = preds[preds["split"] == "test"]
        z = test[["logit_normal", "logit_pneumonia"]].to_numpy() / cal["temperature"]
        p = 1 / (1 + np.exp(z[:, 0] - z[:, 1]))
        labels = test["label"].to_numpy()
        v["fp_conf90"] = int(((labels == 0) & (p >= 0.9)).sum())  # normal X-rays called pneumonia with >= 90%
        missed = p[(labels == 1) & (p < op["threshold"])]
        in_band = int(((missed >= d["low"]) & (missed <= d["high"])).sum())
        if len(missed) == 0:
            v["missed_in_band"] = "It missed no pneumonia X-ray at all."
        elif len(missed) == 1:
            v["missed_in_band"] = (f"The single missed pneumonia X-ray scored {missed[0]:.2f}, "
                                   + ("inside the zone, so the app would have asked for a clinician instead of saying normal."
                                      if in_band else "below the zone, so the app would have called it normal."))
        else:
            v["missed_in_band"] = (f"{in_band} of the {len(missed)} missed pneumonia X-rays score inside the zone, "
                                   "where the app asks for a clinician instead of saying normal.")
        v.update({
            "temperature": f"{cal['temperature']:.2f}",
            "ece_val_before": f"{cal['val_ece_before']:.3f}", "ece_val_after": f"{cal['val_ece_after']:.3f}",
            "ece_test_before": f"{cal['test_ece_before']:.3f}", "ece_test_after": f"{cal['test_ece_after']:.3f}",
            "low": f"{d['low']:.3f}", "high": f"{d['high']:.3f}", "odds_factor": f"{d['odds_factor']:g}",
            "inconclusive_rate": f"{100 * d['test_inconclusive_rate']:.1f}%",
            "acc_decided": f"{d['test_accuracy_when_decided']:.3f}",
            "energy_flag_rate": f"{100 * ood['test_xray_energy_flag_rate']:.1f}%",
        })
    if (reports / "ood_eval.json").exists():
        v["ood_list"] = "\n".join(f"  - {name}: {100 * r['flagged_rate']:.1f}% of {r['n']} flagged "
                                  f"({r['reject']} refused, {r['warn']} warned)"
                                  for name, r in load_json(reports / "ood_eval.json").items())
    if (reports / "sanity.json").exists():
        s = load_json(reports / "sanity.json")
        r, b = s["randomization"], s["border"]
        v.update({
            "rand_n": r["n_images"], "rand_top": f"{r['top']['mean']:.2f}", "rand_classifier": f"{r['classifier']['mean']:.2f}",
            "rand_verdict": r["verdict"], "border_median": f"{100 * b['median']:.0f}%",
            "border_uniform": f"{100 * b['uniform_share']:.0f}%", "border_above": f"{100 * b['share_above_warn']:.1f}%",
            "border_n_above": b["n_above_warn"], "border_verdict": b["verdict"],
            "border_tn": f"{100 * b['median_by_outcome']['true negative']:.0f}%",
            "border_tp": f"{100 * b['median_by_outcome']['true positive']:.0f}%",
        })
    if (reports / "timing.json").exists():
        t = load_json(reports / "timing.json")
        n = t["torch_threads"]
        v.update({"infer_median": f"{t['median_s']:.2f} s", "infer_max": f"{t['max_s']:.2f} s", "infer_runs": t["runs"],
                  "infer_threads": f"{n} CPU thread" + ("" if n == 1 else "s")})
    shared = set(splits.loc[splits["split"] != "test", "patient_id"]) & set(splits.loc[splits["split"] == "test", "patient_id"])
    v["shared_ids"] = len(shared)
    v["shared_test_images"] = int(splits.loc[splits["split"] == "test", "patient_id"].isin(shared).sum())
    return v


def render(template: str, values: dict) -> str:
    """Replace every {{name}} with values[name]; an unknown name is an error, never left in silently."""
    def value(match):
        name = match.group(1)
        if name not in values:
            raise KeyError(f"no value for {{{{{name}}}}} (is its report file missing?)")
        return str(values[name])

    return re.sub(r"\{\{(\w+)\}\}", value, template)


if __name__ == "__main__":
    block = results_markdown()
    for doc in (ROOT / "README.md", ROOT / "docs" / "WALKTHROUGH.md"):
        replace_block(doc, "RESULTS", block.replace("](reports/", "](../reports/") if doc.parent.name == "docs" else block)
    template = ROOT / "docs" / "templates" / "judge_qa.md"
    if template.exists():
        qa = render(template.read_text(encoding="utf-8"), doc_numbers())
        replace_block(ROOT / "docs" / "WALKTHROUGH.md", "QA", qa.strip())
