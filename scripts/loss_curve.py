import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

def plot_histories(metric_paths, labels=None, save_dir="reports/plots", save_name=None, show=True):
    metric_paths = [Path(p) for p in metric_paths]
    labels = labels or [p.parent.name for p in metric_paths]

    curves = []
    for p in metric_paths:
        with open(p, "r", encoding="utf-8") as f:
            h = json.load(f)
        curves.append({
            "train_loss": h.get("train_loss", []),
            "val_loss":   h.get("val_loss", []),
            "train_dice": h.get("train_dice", []),
            "val_dice":   h.get("val_dice", []),
            "best_epoch": h.get("best_epoch", None),
            "best_val_dice": h.get("best_val_dice", None),
        })

    plt.figure(figsize=(12,4))

    # loss
    plt.subplot(1,2,1)
    for c, lab in zip(curves, labels):
        x1 = range(1, len(c["train_loss"])+1)
        x2 = range(1, len(c["val_loss"])+1)
        plt.plot(x1, c["train_loss"], label=f"{lab} Train")
        plt.plot(x2, c["val_loss"], linestyle="--", label=f"{lab} Val")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()

    # dice
    plt.subplot(1,2,2)
    for c, lab in zip(curves, labels):
        x1 = range(1, len(c["train_dice"])+1)
        x2 = range(1, len(c["val_dice"])+1)
        plt.plot(x1, c["train_dice"], label=f"{lab} Train")
        plt.plot(x2, c["val_dice"], linestyle="--", label=f"{lab} Val")
    plt.xlabel("Epochs"); plt.ylabel("Dice"); plt.title("Dice"); plt.legend()

    plt.tight_layout()

    if save_name:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_dir) / save_name
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved → {out_path}")

    if show:
        plt.show()
    else:
        plt.close()