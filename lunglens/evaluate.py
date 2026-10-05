"""
evaluate.py - Step 3 of LungLens: measure the model honestly.

The 624 test X-rays were never used for training or for choosing the
threshold, so they show how the model would do on new children.

Run:  python evaluate.py
Output: results/metrics.json, results/confusion_matrix.png,
        results/roc_curve.png, results/gradcam_examples.png

The numbers that matter for screening
-------------------------------------
- Sensitivity (recall): of the children who HAVE pneumonia, how many did we
  flag? A screening tool must keep this high: a miss means a sick child is
  sent home.
- Specificity: of the healthy children, how many did we correctly clear?
  Low specificity means false alarms and extra work for doctors.
- AUC: how well the scores separate sick from healthy over every possible
  threshold. 1.0 is perfect, 0.5 is a coin flip.
"""

import json
import random

import matplotlib
matplotlib.use("Agg")                    # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from dataset import XrayDataset, list_images
from gradcam import gradcam, overlay
from network import eval_transform, load_model, preprocess
from train import predict_probabilities


def main():
    model, threshold = load_model()
    test_items = list_images("test")
    loader = DataLoader(XrayDataset(test_items, eval_transform), batch_size=32, num_workers=3)
    probs, labels = predict_probabilities(model, loader)
    predicted = (probs >= threshold).astype(int)

    # Confusion matrix: rows = truth, columns = prediction.
    tn, fp, fn, tp = confusion_matrix(labels, predicted).ravel()
    metrics = {
        "test_images": int(len(labels)),
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "auc": roc_auc_score(labels, probs),
        "true_positive": tp, "false_negative": fn,
        "true_negative": tn, "false_positive": fp,
    }
    metrics = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, np.integer) else v)
               for k, v in metrics.items()}
    print(json.dumps(metrics, indent=2))
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Confusion matrix picture ----------------------------------------
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    matrix = np.array([[tn, fp], [fn, tp]])
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=16,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Normal", "Pneumonia"])
    ax.set_yticks([0, 1], ["Normal", "Pneumonia"])
    ax.set_xlabel("LungLens prediction")
    ax.set_ylabel("Doctor's label")
    ax.set_title("Test set (624 X-rays)")
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=200)

    # ---- ROC curve --------------------------------------------------------
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot(fpr, tpr, linewidth=2, label=f"LungLens (AUC = {metrics['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
    ax.set_xlabel("False alarm rate (1 - specificity)")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC curve, test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("results/roc_curve.png", dpi=200)

    # ---- A few Grad-CAM examples -----------------------------------------
    random.seed(0)
    picks = random.sample([i for i in test_items if i[1] == 0], 2) + \
            random.sample([i for i in test_items if i[1] == 1], 2)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for col, (path, label) in enumerate(picks):
        image = Image.open(path).convert("RGB")
        probability, heat = gradcam(model, preprocess(image), threshold)
        axes[0, col].imshow(image, cmap="gray")
        axes[0, col].set_title(f"Doctor: {'Pneumonia' if label else 'Normal'}")
        axes[1, col].imshow(overlay(image, heat))
        axes[1, col].set_title(f"LungLens: {probability:.0%} pneumonia")
        for row in range(2):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig("results/gradcam_examples.png", dpi=150)
    print("Saved plots to results/")


if __name__ == "__main__":
    main()
