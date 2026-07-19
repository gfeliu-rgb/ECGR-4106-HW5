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

    plt.figure(figsize=(10, 6))
    for model_name, group in df.groupby("model_name"):
        if "val_loss" in group.columns:
            plt.plot(group["epoch"], group["val_loss"], label=f"{model_name} val")
        if "train_loss" in group.columns:
            plt.plot(group["epoch"], group["train_loss"], linestyle="--", label=f"{model_name} train")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(csv_path.parent / output_name, dpi=200)
    plt.close()


def main() -> None:
    maybe_plot_history(P1 / "problem1_history.csv", "problem1_loss_curves.png", "Homework 5 Problem 1 Loss Curves")
    maybe_plot_history(P2 / "problem2_history.csv", "problem2_training_curves.png", "Homework 5 Problem 2 Loss Curves")


if __name__ == "__main__":
    main()
