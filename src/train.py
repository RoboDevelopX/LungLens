"""Train LungLens with two-stage transfer learning.

Stage 1 (head_epochs):     backbone frozen, only the new 2-class layer learns.
Stage 2 (finetune_epochs): everything unfrozen, lower learning rate, cosine
                           decay, early stopping on validation AUROC.

The best epoch (highest val AUROC) is saved to models/lunglens.pt and the
per-epoch numbers go to reports/train_history.json. The test set is never
touched here; that happens once, in src/evaluate.py.

Usage:
    python -m src.train                               # full run with config.yaml settings
    python -m src.train --max-batches 3 --head-epochs 1 --finetune-epochs 1   # quick smoke test
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.data import class_weights, load_split_table, make_loaders
from src.model import LungLensNet, save_checkpoint
from src.utils import get_device, load_config, repo_path, save_json, set_seed


def run_epoch(model, loader, loss_fn, device, optimizer=None, max_batches=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise just evaluates.

    Returns (mean loss, pneumonia probabilities, true labels) for the images seen.
    """
    training = optimizer is not None
    model.train(training)  # train mode: dropout/batch-norm update; eval mode: frozen statistics
    total_loss, all_probs, all_labels = 0.0, [], []

    with torch.set_grad_enabled(training):
        for i, (images, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu())  # P(pneumonia)
            all_labels.append(labels.cpu())

    probs, labels = torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()
    return total_loss / len(labels), probs, labels


def safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    """AUROC, or NaN when only one class is present (possible in tiny smoke tests)."""
    return float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan")


def main() -> None:
    cfg = load_config()
    tcfg = cfg["train"]
    parser = argparse.ArgumentParser(description="Train the LungLens classifier.")
    parser.add_argument("--backbone", default=cfg["model"]["backbone"], choices=["densenet121", "resnet50"])
    parser.add_argument("--head-epochs", type=int, default=tcfg["head_epochs"])
    parser.add_argument("--finetune-epochs", type=int, default=tcfg["finetune_epochs"])
    parser.add_argument("--max-batches", type=int, default=None, help="limit batches per epoch (smoke tests only)")
    parser.add_argument("--out", default=cfg["paths"]["model"], help="where to save the best checkpoint")
    parser.add_argument("--history", default="reports/train_history.json")
    args = parser.parse_args()

    set_seed(cfg["seed"])
    device = get_device()
    print(f"device={device}  backbone={args.backbone}  threads={torch.get_num_threads()}")

    loaders = make_loaders(cfg)
    table = load_split_table(cfg)
    weights = class_weights(table.loc[table["split"] == "train", "label"].tolist())
    print(f"class weights [NORMAL, PNEUMONIA] = {weights.tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    model = LungLensNet(args.backbone, pretrained=cfg["model"]["pretrained"]).to(device)

    history, best_auroc, epochs_without_gain = [], -1.0, 0
    stages = [
        # (name, number of epochs, backbone trainable?, learning rate)
        ("head", args.head_epochs, False, tcfg["head_lr"]),
        ("finetune", args.finetune_epochs, True, tcfg["finetune_lr"]),
    ]

    for stage, n_epochs, backbone_trainable, lr in stages:
        if n_epochs == 0:
            continue
        model.set_backbone_trainable(backbone_trainable)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=tcfg["weight_decay"])
        # Cosine schedule: learning rate glides from lr down towards 0 over the stage.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            lr_now = optimizer.param_groups[0]["lr"]  # read before the scheduler changes it
            train_loss, train_probs, train_labels = run_epoch(
                model, loaders["train"], loss_fn, device, optimizer, args.max_batches
            )
            t_train = time.time() - t0
            val_loss, val_probs, val_labels = run_epoch(model, loaders["val"], loss_fn, device, None, args.max_batches)
            t_total = time.time() - t0
            scheduler.step()

            val_auroc = safe_auroc(val_labels, val_probs)
            row = {
                "stage": stage,
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": round(train_loss, 4),
                "train_auroc": round(safe_auroc(train_labels, train_probs), 4),
                "val_loss": round(val_loss, 4),
                "val_auroc": round(val_auroc, 4),
                "val_acc_at_0.5": round(float(((val_probs >= 0.5) == val_labels).mean()), 4),
                "seconds_train": round(t_train, 1),
                "seconds_total": round(t_total, 1),
            }
            history.append(row)
            print(
                f"[{stage} {epoch}/{n_epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_auroc={val_auroc:.4f} val_acc@0.5={row['val_acc_at_0.5']:.4f} "
                f"time={t_train:.0f}s train + {t_total - t_train:.0f}s val",
                flush=True,
            )

            if val_auroc > best_auroc:
                best_auroc, epochs_without_gain = val_auroc, 0
                save_checkpoint(
                    model,
                    repo_path(args.out),
                    extra={"stage": stage, "epoch": epoch, "val_auroc": val_auroc, "config": cfg},
                )
                print(f"  saved new best checkpoint (val AUROC {val_auroc:.4f})", flush=True)
            else:
                epochs_without_gain += 1

            save_json({"backbone": args.backbone, "best_val_auroc": best_auroc, "epochs": history}, repo_path(args.history))

            # Early stopping only applies once the whole network is being fine-tuned.
            if stage == "finetune" and epochs_without_gain >= tcfg["early_stop_patience"]:
                print(f"  early stopping: no val AUROC gain for {epochs_without_gain} epochs", flush=True)
                break

    print(f"done. best val AUROC = {best_auroc:.4f}; checkpoint at {args.out}")


if __name__ == "__main__":
    main()
