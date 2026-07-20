"""
Plot helper for Homework 5 CSV outputs.

Expected inputs:
- Results_Problem_1/problem1_history.csv
- Results_Problem_1/problem1_summary.csv
- Results_Problem_2/problem2_history.csv
- Results_Problem_2/problem2_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "Results_Problem_1"
P2 = ROOT / "Results_Problem_2"


def maybe_plot_history(csv_path: Path, output_name: str, title: str) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name, group in df.groupby("model_name"):
        if "val_loss" in group.columns:
            axes[0].plot(group["epoch"], group["val_loss"], label=f"{model_name} val")
        if "train_loss" in group.columns:
            axes[0].plot(group["epoch"], group["train_loss"], linestyle="--", label=f"{model_name} train")
        if "val_accuracy_pct" in group.columns:
            axes[1].plot(group["epoch"], group["val_accuracy_pct"], label=f"{model_name} val")
        if "train_accuracy_pct" in group.columns:
            axes[1].plot(group["epoch"], group["train_accuracy_pct"], linestyle="--", label=f"{model_name} train")

    axes[0].set_title(f"{title}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)

    axes[1].set_title(f"{title}: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(csv_path.parent / output_name, dpi=200)
    plt.close(fig)


def main() -> None:
    maybe_plot_history(P1 / "problem1_history.csv", "problem1_loss_curves.png", "Homework 5 Problem 1 Training Curves")
    maybe_plot_history(P2 / "problem2_history.csv", "problem2_training_curves.png", "Homework 5 Problem 2 Training Curves")


if __name__ == "__main__":
    main()
