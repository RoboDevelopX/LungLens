"""Temperature scaling and calibration error (ECE).

"Calibrated" means: among all X-rays the model gives about 80% confidence,
about 80% should really be correct. Deep networks are usually OVER-confident,
so we soften their scores with one number, the temperature T (Guo et al., 2017):

    p = softmax(logits / T)      T > 1 softens, T < 1 sharpens, T = 1 changes nothing

T is fitted on the VALIDATION set by minimising the negative log-likelihood.
Because dividing both logits by the same positive T never changes which one
is bigger, temperature scaling never changes a prediction's ranking: AUROC
stays identical; only the confidence values change.
"""

import numpy as np
import torch


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Find the T that minimises cross-entropy on (logits, labels). Returns T as a float."""
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)

    # Optimise log(T) instead of T so that T = exp(log_t) can never go negative.
    log_t = torch.zeros(1, requires_grad=True)  # start at T = 1
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits_t / log_t.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def softmax_probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """N x 2 logits -> N x 2 probabilities, after dividing by the temperature."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)  # numerical stability; does not change the result
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> list[dict]:
    """Group predictions by confidence and compare confidence with accuracy per group.

    confidence = probability of the predicted class (always between 0.5 and 1
    for two classes). A perfectly calibrated model has accuracy == confidence
    in every bin. These rows are what the reliability diagram plots.
    """
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            rows.append(
                {
                    "lo": round(float(lo), 4),
                    "hi": round(float(hi), 4),
                    "count": int(in_bin.sum()),
                    "confidence": float(confidence[in_bin].mean()),
                    "accuracy": float(correct[in_bin].mean()),
                }
            )
    return rows


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE = weighted average gap between accuracy and confidence across bins.

    ECE = sum over bins of (bin size / N) * |accuracy(bin) - confidence(bin)|.
    0 is perfect; 0.05 means confidences are off by 5 percentage points on average.
    """
    n = len(labels)
    return float(sum(r["count"] / n * abs(r["accuracy"] - r["confidence"]) for r in reliability_bins(probs, labels, n_bins)))
