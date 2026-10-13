"""Turn numbers into a decision and a plain-language explanation.

Kept separate from the Streamlit code so that the wording and the decision
rules can be unit-tested and read in one place.

Decision rule (all probabilities are CALIBRATED P(pneumonia)):

    p <  low                -> "normal"        (no pneumonia pattern found)
    low <= p <= high        -> "inconclusive"  (too close to call; see a clinician)
    p >  high               -> "pneumonia"     (pneumonia pattern found)

The operating threshold is chosen on validation data (src/evaluate.py): the
highest threshold that still catches >= 95% of pneumonia, capped at 0.5.
low/high surround it: probabilities whose odds are within a factor of 1.5 of
the threshold's odds (40%-60% around 0.5), or two fixed values if
inconclusive_low/inconclusive_high are set in config.yaml.
"""

from dataclasses import dataclass

import numpy as np

ZONE_ROWS = ("upper", "middle", "lower")
# Radiology convention: the patient's RIGHT side is shown on the LEFT of the image.
ZONE_COLS = ("image left (patient's right)", "image right (patient's left)")


def inconclusive_band(threshold: float, odds_factor: float) -> tuple[float, float]:
    """The two edges of the 'Inconclusive' zone around the threshold.

    We work with ODDS = p / (1 - p): the zone holds every probability whose odds
    are within `odds_factor` of the threshold's odds. Around 0.5 (odds 1) with
    factor 1.5 that is odds 0.67-1.5, i.e. 40%-60%. Unlike a fixed +/- 0.10 it
    can never run past 0% or 100%, so all three answers stay possible wherever
    the threshold is. odds_factor = 1 means no zone (a plain threshold).
    """
    if odds_factor < 1:
        raise ValueError("odds_factor must be >= 1")
    threshold = min(max(threshold, 1e-6), 1 - 1e-6)  # odds of exactly 0 or 1 are undefined
    odds = threshold / (1 - threshold)
    low_odds, high_odds = odds / odds_factor, odds * odds_factor
    return low_odds / (1 + low_odds), high_odds / (1 + high_odds)


def band_from_config(threshold: float, decision_cfg: dict) -> tuple[float, float]:
    """The band to use: the two fixed values from config.yaml if both are set, else the odds-factor band."""
    low, high = decision_cfg.get("inconclusive_low"), decision_cfg.get("inconclusive_high")
    if low is None or high is None:
        return inconclusive_band(threshold, decision_cfg["inconclusive_odds_factor"])
    if not 0.0 <= low <= threshold <= high <= 1.0:
        raise ValueError(f"need 0 <= inconclusive_low <= operating threshold ({threshold:.3f}) <= inconclusive_high <= 1")
    return float(low), float(high)


def decide(p: float, low: float, high: float) -> str:
    """Map a calibrated probability to 'normal', 'inconclusive' or 'pneumonia'."""
    if low > high:
        raise ValueError("low must be <= high")
    if p < low:
        return "normal"
    if p > high:
        return "pneumonia"
    return "inconclusive"


def attention_zones(cam: np.ndarray) -> dict:
    """Share of heatmap weight in each of six zones (3 rows x 2 columns)."""
    h, w = cam.shape
    total = cam.sum()
    shares = {}
    for i, row in enumerate(ZONE_ROWS):
        for j, col in enumerate(ZONE_COLS):
            block = cam[i * h // 3 : (i + 1) * h // 3, j * w // 2 : (j + 1) * w // 2]
            shares[f"{row} zone, {col}"] = float(block.sum() / total) if total > 0 else 0.0
    return shares


@dataclass
class Explanation:
    decision: str        # "pneumonia", "normal", "inconclusive" or "not_xray"
    headline: str
    layperson: list[str]
    clinician: list[str]


def explain(
    p: float,
    low: float,
    high: float,
    cam: np.ndarray | None = None,
    border_share: float | None = None,
    border_warn: float = 0.5,
    temperature: float | None = None,
    ood_reasons: list[str] | None = None,
) -> Explanation:
    """Build the text shown in the app's explanation panel."""
    decision = decide(p, low, high)
    pct = f"{100 * p:.0f}%"

    if decision == "pneumonia":
        headline = "Signs consistent with pneumonia"
        lay = [
            f"The AI found patterns that often appear in pneumonia. It estimates a {pct} chance that this X-ray shows pneumonia.",
            "This is a screening flag, not a diagnosis. Please show this X-ray to a doctor, who will also consider symptoms, an examination and other tests.",
        ]
    elif decision == "normal":
        headline = "No signs of pneumonia found"
        lay = [
            f"The AI did not find the patterns it links to pneumonia. It estimates a {pct} chance of pneumonia.",
            "This does not rule out pneumonia or any other illness. If the person feels unwell, please see a doctor.",
        ]
    else:
        headline = "Inconclusive: please consult a clinician"
        lay = [
            f"The AI's estimate ({pct}) is too close to its decision line to give a reliable answer either way.",
            "Rather than guess, LungLens asks a clinician to review this X-ray.",
        ]

    if decision == "normal":
        lay.append(
            "The coloured overlay shows where the AI found the most pneumonia-like evidence, even though overall it "
            "was too weak to raise a flag. The colours are scaled to this image's strongest area, so red here does "
            "not mean disease."
        )
    else:
        lay.append(
            "The coloured overlay shows which areas pushed the pneumonia score up: the stronger the red, the bigger "
            "that area's influence. It shows where the evidence came from, not why, and it is not a map of the infection."
        )

    clin = [
        f"Calibrated P(pneumonia) = {p:.3f}" + (f" (temperature T = {temperature:.2f})" if temperature else ""),
        f"Decision bands: normal < {low:.2f} <= inconclusive <= {high:.2f} < pneumonia",
    ]

    if cam is not None:
        zones = attention_zones(cam)
        top_zone, share = max(zones.items(), key=lambda kv: kv[1])
        clin.append(f"Grad-CAM: largest share of attention ({100 * share:.0f}%) in the {top_zone}")
    if border_share is not None:
        clin.append(f"Grad-CAM weight in the outer 10% border band: {100 * border_share:.0f}%")
        if border_share > border_warn:
            lay.append(
                "Caution: much of the AI's attention is at the edges of the image, where text labels and borders sit, "
                "rather than on the lungs. Treat this result with extra care."
            )
    if ood_reasons:
        clin.extend(f"Input check: {r}" for r in ood_reasons)

    return Explanation(decision, headline, lay, clin)
