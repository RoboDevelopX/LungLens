"""Threshold logic, metrics and calibration maths."""

import numpy as np
import pytest

from src.calibration import expected_calibration_error, fit_temperature, softmax_probs
from src.evaluate import (binary_metrics, bootstrap_ci, choose_threshold, confusion_counts,
                          threshold_for_sensitivity, threshold_sweep)
from src.explain import attention_zones, band_from_config, decide, explain, format_chance, inconclusive_band


# --- decision bands --------------------------------------------------------
@pytest.mark.parametrize(
    "p,expected",
    [(0.0, "normal"), (0.39, "normal"), (0.40, "inconclusive"), (0.5, "inconclusive"),
     (0.60, "inconclusive"), (0.61, "pneumonia"), (1.0, "pneumonia")],
)
def test_decide_three_zones(p, expected):
    assert decide(p, low=0.40, high=0.60) == expected


def test_decide_rejects_inverted_band():
    with pytest.raises(ValueError):
        decide(0.5, low=0.7, high=0.3)


def test_band_is_40_to_60_percent_around_a_threshold_of_one_half():
    assert inconclusive_band(0.5, 1.5) == (pytest.approx(0.4), pytest.approx(0.6))


@pytest.mark.parametrize("threshold", [0.01, 0.05, 0.3, 0.5, 0.93, 0.99])
def test_band_never_runs_past_0_or_1_so_all_three_answers_stay_possible(threshold):
    low, high = inconclusive_band(threshold, 1.5)
    assert 0 < low < threshold < high < 1
    assert decide(low / 2, low, high) == "normal"
    assert decide((high + 1) / 2, low, high) == "pneumonia"


def test_band_can_be_fixed_with_two_configurable_thresholds():
    odds_only = {"inconclusive_odds_factor": 1.5, "inconclusive_low": None, "inconclusive_high": None}
    assert band_from_config(0.5, odds_only) == inconclusive_band(0.5, 1.5)
    fixed = {"inconclusive_odds_factor": 1.5, "inconclusive_low": 0.3, "inconclusive_high": 0.7}
    assert band_from_config(0.5, fixed) == (0.3, 0.7)
    with pytest.raises(ValueError):  # a band that does not contain the operating threshold makes no sense
        band_from_config(0.8, fixed)


def test_odds_factor_of_one_means_a_plain_threshold():
    low, high = inconclusive_band(0.5, 1.0)
    assert decide(0.49, low, high) == "normal" and decide(0.51, low, high) == "pneumonia"


# --- sensitivity-first threshold ---------------------------------------------
def test_threshold_is_capped_so_likely_pneumonia_is_never_called_normal():
    # A very confident model: every pneumonia case scores >= 0.9, so the 95% rule alone would pick about 0.9.
    labels = np.array([0] * 50 + [1] * 50)
    probs = np.concatenate([np.linspace(0.0, 0.3, 50), np.linspace(0.9, 1.0, 50)])
    assert threshold_for_sensitivity(labels, probs, 0.95) > 0.9
    t = choose_threshold(labels, probs, target=0.95, max_threshold=0.5)
    assert t == 0.5
    assert binary_metrics(labels, probs, t)["sensitivity"] >= 0.95  # lowering a threshold keeps the target


def test_threshold_reaches_target_sensitivity_and_is_highest_possible():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 200 + [1] * 300)
    probs = np.clip(np.concatenate([rng.normal(0.3, 0.15, 200), rng.normal(0.7, 0.15, 300)]), 0, 1)
    t = threshold_for_sensitivity(labels, probs, target=0.95)
    assert binary_metrics(labels, probs, t)["sensitivity"] >= 0.95
    # any higher threshold (the next pneumonia score above t) drops below the target
    higher = np.min(probs[(labels == 1) & (probs > t)])
    assert binary_metrics(labels, probs, higher)["sensitivity"] < 0.95


def test_binary_metrics_on_known_confusion():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert confusion_counts(labels, probs, 0.5) == {"tp": 3, "fn": 1, "tn": 5, "fp": 1}
    m = binary_metrics(labels, probs, 0.5)
    assert m["sensitivity"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(5 / 6)
    assert m["precision"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.8)


def test_metrics_do_not_crash_without_positive_predictions():
    labels, probs = np.array([0, 1]), np.array([0.1, 0.2])
    assert binary_metrics(labels, probs, 0.99)["precision"] == 0.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 300)
    probs = np.clip(labels * 0.4 + rng.random(300) * 0.6, 0, 1)
    ci = bootstrap_ci(labels, probs, 0.5, n=300)
    point = binary_metrics(labels, probs, 0.5)
    for k in ("sensitivity", "specificity", "accuracy"):
        assert ci[k][0] <= point[k] <= ci[k][1]


def test_threshold_sweep_is_monotonic():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, 200)
    probs = rng.random(200)
    sweep = threshold_sweep(labels, probs)
    sens = [r["sensitivity"] for r in sweep]
    spec = [r["specificity"] for r in sweep]
    assert len(sweep) == 101
    assert all(a >= b for a, b in zip(sens, sens[1:]))  # raising the threshold never catches more
    assert all(a <= b for a, b in zip(spec, spec[1:]))


# --- calibration ----------------------------------------------------------------
def test_temperature_recovers_known_overconfidence():
    """Make labels from logits/2, then show the model the raw logits: the right fix is T = 2."""
    rng = np.random.default_rng(3)
    margin = rng.normal(0, 3, 20000)
    labels = (rng.random(20000) < 1 / (1 + np.exp(-margin / 2))).astype(int)
    logits = np.stack([np.zeros_like(margin), margin], axis=1)
    assert fit_temperature(logits, labels) == pytest.approx(2.0, abs=0.15)


def test_temperature_scaling_never_changes_the_ranking():
    rng = np.random.default_rng(4)
    logits = rng.normal(0, 2, (100, 2))
    p1, p2 = softmax_probs(logits, 1.0)[:, 1], softmax_probs(logits, 3.0)[:, 1]
    assert np.array_equal(np.argsort(p1), np.argsort(p2))


def test_ece_is_zero_for_perfect_calibration_and_large_for_overconfidence():
    # 100 predictions at 80% confidence, exactly 80 correct -> perfectly calibrated
    probs = np.tile([0.2, 0.8], (100, 1))
    labels = np.array([1] * 80 + [0] * 20)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)
    # 99% confident but only 50% correct -> ECE about 0.49
    probs = np.tile([0.01, 0.99], (100, 1))
    labels = np.array([1] * 50 + [0] * 50)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.49, abs=1e-6)


# --- explanation text -------------------------------------------------------------
def test_explanation_matches_decision_and_mentions_uncertainty():
    for p, decision in ((0.1, "normal"), (0.5, "inconclusive"), (0.9, "pneumonia")):
        e = explain(p, 0.4, 0.6)
        assert e.decision == decision
        assert e.layperson and e.clinician
    assert "clinician" in explain(0.5, 0.4, 0.6).headline.lower()
    assert "not a diagnosis" in " ".join(explain(0.9, 0.4, 0.6).layperson)
    # For a normal result the heatmap is rescaled to its own maximum: the text must say red is not disease.
    assert "does not mean disease" in " ".join(explain(0.1, 0.4, 0.6).layperson)


def test_border_caution_is_added_only_when_attention_is_on_the_edges():
    calm = explain(0.9, 0.4, 0.6, border_share=0.2, border_warn=0.5)
    edgy = explain(0.9, 0.4, 0.6, border_share=0.8, border_warn=0.5)
    assert not any("edges" in t for t in calm.layperson)
    assert any("edges" in t for t in edgy.layperson)


def test_attention_zones_sum_to_one_and_find_the_hot_zone():
    cam = np.zeros((90, 90))
    cam[70:90, 0:20] = 1  # lower part of the image's left side
    zones = attention_zones(cam)
    assert sum(zones.values()) == pytest.approx(1.0)
    assert max(zones, key=zones.get).startswith("lower zone, image left")


@pytest.mark.parametrize("p, shown", [(0.0, "<1%"), (0.004, "<1%"), (0.005, "1%"), (0.481, "48%"),
                                      (0.994, "99%"), (0.995, ">99%"), (1.0, ">99%")])
def test_chance_is_never_shown_as_0_or_100_percent(p, shown):
    assert format_chance(p) == shown


def test_empty_heatmap_is_explained_instead_of_pointing_at_a_zone():
    """A confident normal X-ray can have no evidence FOR pneumonia anywhere: Grad-CAM is then all zeros."""
    exp = explain(0.001, 0.4, 0.6, cam=np.zeros((224, 224)), border_share=0.0)
    assert any("heatmap is empty" in line for line in exp.layperson)
    assert not any("largest share" in line or "border band" in line for line in exp.clinician)
