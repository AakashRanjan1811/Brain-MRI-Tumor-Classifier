"""Figures generated from results/metrics.json (can be re-run without retraining)."""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .config import CLASS_DISPLAY, CLASSES

NAMES = {"baseline_cnn": "Baseline CNN (original architecture)", "efficientnet_b0": "EfficientNetB0 (transfer learning)"}


def plot_training_curves(results, out_path):
    models = [m for m in NAMES if m in results["models"]]
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.8 * len(models)), squeeze=False)
    for r, m in enumerate(models):
        h = results["models"][m]["history"]
        ep = np.arange(1, len(h["loss"]) + 1)
        for c, (key, label) in enumerate((("accuracy", "Accuracy"), ("loss", "Loss"))):
            ax = axes[r, c]
            ax.plot(ep, h[key], label="Train", color="#1f77b4")
            ax.plot(ep, h["val_" + key], label="Validation", color="#ff7f0e")
            if "finetune" in h.get("phase", []):
                start = h["phase"].index("finetune") + 0.5
                ax.axvline(start, color="gray", ls="--", lw=1)
                ax.text(start, ax.get_ylim()[1], " fine-tuning starts", va="top", fontsize=8, color="gray")
            ax.set_title(f"{NAMES[m]}: {label}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_confusion_matrices(results, out_path):
    models = [m for m in NAMES if m in results["models"]]
    labels = [CLASS_DISPLAY[c] for c in CLASSES]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, m in zip(axes[0], models):
        cm = np.array(results["models"][m]["test_metrics"]["confusion_matrix"])
        pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if pct[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(CLASSES)), labels, rotation=20)
        ax.set_yticks(range(len(CLASSES)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        acc = results["models"][m]["test_metrics"]["accuracy"]
        ax.set_title(f"{NAMES[m]}\nTest accuracy {acc:.2%}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def make_all(results_dir):
    with open(os.path.join(results_dir, "metrics.json")) as f:
        results = json.load(f)
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_training_curves(results, os.path.join(fig_dir, "training_curves.png"))
    plot_confusion_matrices(results, os.path.join(fig_dir, "confusion_matrices.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    make_all(ap.parse_args().results_dir)
