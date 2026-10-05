"""
train.py - Step 2 of LungLens: fine-tune the model on the X-rays.

Run:  python train.py
Output: models/lunglens.pt (the trained weights + chosen decision threshold)
        results/training_log.json

How training works, in one paragraph
------------------------------------
We show the model a batch of 32 X-rays, it guesses a pneumonia score for
each, and we measure how wrong the guesses are with a "loss" function.
Back-propagation then works out, for every adjustable number (weight) in the
network, which direction would make the loss smaller, and the optimiser
nudges each weight a little in that direction. One pass over all training
images is an "epoch". After each epoch we check the model on validation
images it never trains on, and keep the best version.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import XrayDataset, list_images, train_val_split
from network import LungLensNet, eval_transform, train_transform

EPOCHS = 5
BATCH_SIZE = 32
SEED = 42


def predict_probabilities(model, loader):
    """Run the model over a whole dataset and return (probabilities, labels)."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():                       # no learning here, so skip gradients
        for images, y in loader:
            probs.append(torch.sigmoid(model(images)).numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def best_threshold(probs, labels):
    """
    A probability above the threshold means "flag as pneumonia".
    Instead of blindly using 0.5 we pick the threshold that maximises
    sensitivity + specificity on the validation set (Youden's J statistic),
    so the model is balanced between missing sick children and raising
    false alarms on healthy ones.
    """
    best_t, best_j = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        predicted = probs >= t
        sensitivity = (predicted & (labels == 1)).sum() / (labels == 1).sum()
        specificity = (~predicted & (labels == 0)).sum() / (labels == 0).sum()
        if sensitivity + specificity - 1 > best_j:
            best_t, best_j = float(t), sensitivity + specificity - 1
    return best_t


def main():
    torch.manual_seed(SEED)
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # ---- Data -------------------------------------------------------------
    train_items, val_items = train_val_split(list_images("train"))
    n_pneumonia = sum(y for _, y in train_items)
    n_normal = len(train_items) - n_pneumonia
    print(f"Training on {len(train_items)} images ({n_normal} normal, {n_pneumonia} pneumonia)")
    print(f"Validating on {len(val_items)} images")

    train_loader = DataLoader(XrayDataset(train_items, train_transform),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    val_loader = DataLoader(XrayDataset(val_items, eval_transform),
                            batch_size=BATCH_SIZE, num_workers=3)

    # ---- Model ------------------------------------------------------------
    model = LungLensNet(pretrained=True)
    model.freeze_early_layers()

    # The dataset has about 3x more pneumonia than normal images. Without a
    # correction the model could score well by mostly saying "pneumonia".
    # pos_weight < 1 makes each pneumonia example count less, so both
    # classes matter equally in the loss.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_normal / n_pneumonia))

    # AdamW optimiser. The new classifier layer starts from random numbers,
    # so it gets a larger learning rate than the pretrained layers.
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.features.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    # Gradually lower the learning rate over training (cosine schedule).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---- Training loop ----------------------------------------------------
    log, best_auc = [], 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start, running_loss = time.time(), 0.0
        for step, (images, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()                     # clear last step's gradients
            loss = loss_fn(model(images), labels.float())
            loss.backward()                           # back-propagation
            optimizer.step()                          # nudge the weights
            running_loss += loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running_loss / step:.4f}", flush=True)
        scheduler.step()

        # Check how well we do on the validation patients.
        val_probs, val_labels = predict_probabilities(model, val_loader)
        val_auc = roc_auc_score(val_labels, val_probs)
        entry = {"epoch": epoch, "train_loss": running_loss / len(train_loader),
                 "val_auc": float(val_auc), "minutes": (time.time() - start) / 60}
        log.append(entry)
        print(entry, flush=True)

        # Keep the best model so far (highest validation AUC).
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model_state": model.state_dict(),
                        "threshold": best_threshold(val_probs, val_labels),
                        "epoch": epoch, "val_auc": float(val_auc)},
                       "models/lunglens.pt")
            print(f"  saved new best model (val AUC {val_auc:.4f})")

    Path("results/training_log.json").write_text(json.dumps(log, indent=2))
    print("Training finished.")


if __name__ == "__main__":
    main()
