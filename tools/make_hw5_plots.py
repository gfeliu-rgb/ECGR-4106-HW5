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

DISPLAY_NAMES = {
    "vit_patch4_dim256_d4_h4": "ViT-A\np4 d256",
    "vit_patch4_dim512_d8_h8": "ViT-B\np4 d512",
    "vit_patch8_dim256_d4_h4": "ViT-C\np8 d256",
    "vit_patch8_dim512_d8_h8": "ViT-D\np8 d512",
    "resnet18": "ResNet-18",
    "swin_tiny_pretrained": "Swin-Tiny\npretrained",
    "swin_small_pretrained": "Swin-Small\npretrained",
    "swin_scratch": "Scratch\nSwin",
}


def display_name(model_name: str) -> str:
    return DISPLAY_NAMES.get(str(model_name), str(model_name))


def maybe_plot_history(csv_path: Path, output_name: str, title: str) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name, group in df.groupby("model_name"):
        if "val_loss" in group.columns:
            axes[0].plot(group["epoch"], group["val_loss"], marker="o", label=f"{model_name} val")
        if "train_loss" in group.columns:
            axes[0].plot(group["epoch"], group["train_loss"], marker="x", alpha=0.72, label=f"{model_name} train")
        if "val_accuracy_pct" in group.columns:
            axes[1].plot(group["epoch"], group["val_accuracy_pct"], marker="o", label=f"{model_name} val")
        if "train_accuracy_pct" in group.columns:
            axes[1].plot(group["epoch"], group["train_accuracy_pct"], marker="x", alpha=0.72, label=f"{model_name} train")

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


def annotate_points(
    ax: plt.Axes,
    model_names: pd.Series,
    labels: list[str],
    x_values: pd.Series,
    y_values: pd.Series,
    offsets: dict[str, tuple[int, int]] | None = None,
) -> None:
    offsets = offsets or {}
    for model_name, label, x_value, y_value in zip(model_names, labels, x_values, y_values, strict=True):
        ax.annotate(
            label.replace("\n", " "),
            (x_value, y_value),
            textcoords="offset points",
            xytext=offsets.get(str(model_name), (5, 5)),
            fontsize=7,
        )


def plot_problem1_tradeoffs() -> None:
    summary_path = P1 / "problem1_summary.csv"
    if not summary_path.exists():
        return

    df = pd.read_csv(summary_path)
    order = [
        "vit_patch4_dim256_d4_h4",
        "vit_patch4_dim512_d8_h8",
        "vit_patch8_dim256_d4_h4",
        "vit_patch8_dim512_d8_h8",
        "resnet18",
    ]
    df["order"] = df["model_name"].map({name: idx for idx, name in enumerate(order)})
    df = df.sort_values("order")
    labels = [display_name(name) for name in df["model_name"]]
    params_m = pd.to_numeric(df["parameter_count"], errors="coerce") / 1e6
    flops_g = pd.to_numeric(df["flops_forward"], errors="coerce") / 1e9
    accuracy = pd.to_numeric(df["test_accuracy_pct"], errors="coerce")
    time_sec = pd.to_numeric(df["train_time_per_epoch_sec"], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(labels, accuracy, color="#4c78a8")
    axes[0, 0].set_title("Final Test Accuracy")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(0, max(accuracy) + 8)
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(labels, time_sec, color="#f58518")
    axes[0, 1].set_title("Training Time")
    axes[0, 1].set_ylabel("Seconds / Epoch")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].scatter(params_m, accuracy, s=90, color="#54a24b")
    axes[1, 0].set_title("Accuracy vs. Parameter Count")
    axes[1, 0].set_xlabel("Parameters (millions)")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_ylim(0, max(accuracy) + 8)
    axes[1, 0].grid(alpha=0.25)
    annotate_points(
        axes[1, 0],
        df["model_name"],
        labels,
        params_m,
        accuracy,
        {
            "vit_patch4_dim512_d8_h8": (6, -12),
            "vit_patch8_dim512_d8_h8": (6, 6),
            "resnet18": (6, 6),
        },
    )

    axes[1, 1].scatter(flops_g, accuracy, s=90, color="#e45756")
    axes[1, 1].set_title("Accuracy vs. Estimated FLOPs")
    axes[1, 1].set_xlabel("FLOPs / Forward (billions, log scale)")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlim(min(flops_g) * 0.7, max(flops_g) * 1.8)
    axes[1, 1].set_ylim(0, max(accuracy) + 8)
    axes[1, 1].grid(alpha=0.25)
    annotate_points(
        axes[1, 1],
        df["model_name"],
        labels,
        flops_g,
        accuracy,
        {
            "vit_patch4_dim512_d8_h8": (6, 6),
            "vit_patch8_dim512_d8_h8": (6, -14),
        },
    )

    fig.suptitle("Homework 5 Problem 1: Accuracy, Runtime, and Compute Tradeoffs", fontsize=14)
    fig.tight_layout()
    fig.savefig(P1 / "problem1_tradeoffs.png", dpi=200)
    plt.close(fig)


def plot_problem2_tradeoffs() -> None:
    summary_path = P2 / "problem2_summary.csv"
    if not summary_path.exists():
        return

    df = pd.read_csv(summary_path)
    order = ["swin_tiny_pretrained", "swin_small_pretrained", "swin_scratch"]
    df["order"] = df["model_name"].map({name: idx for idx, name in enumerate(order)})
    df = df.sort_values("order")
    labels = [display_name(name) for name in df["model_name"]]
    total_params_m = pd.to_numeric(df["total_parameter_count"], errors="coerce") / 1e6
    trainable_params_m = pd.to_numeric(df["parameter_count"], errors="coerce") / 1e6
    accuracy = pd.to_numeric(df["test_accuracy_pct"], errors="coerce")
    time_sec = pd.to_numeric(df["train_time_per_epoch_sec"], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(labels, accuracy, color="#4c78a8")
    axes[0, 0].set_title("Final Test Accuracy")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(0, max(accuracy) + 8)
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(labels, time_sec, color="#f58518")
    axes[0, 1].set_title("Training Time")
    axes[0, 1].set_ylabel("Seconds / Epoch")
    axes[0, 1].tick_params(axis="x", rotation=15)
    axes[0, 1].grid(axis="y", alpha=0.25)

    x_positions = range(len(labels))
    width = 0.35
    axes[1, 0].bar([x - width / 2 for x in x_positions], total_params_m, width=width, label="total", color="#54a24b")
    axes[1, 0].bar([x + width / 2 for x in x_positions], trainable_params_m, width=width, label="trainable", color="#b279a2")
    axes[1, 0].set_title("Total vs. Trainable Parameters")
    axes[1, 0].set_ylabel("Parameters (millions, log scale)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(list(x_positions), labels)
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].scatter(trainable_params_m, accuracy, s=110, color="#e45756")
    axes[1, 1].set_title("Accuracy vs. Trainable Parameters")
    axes[1, 1].set_xlabel("Trainable parameters (millions, log scale)")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlim(min(trainable_params_m) * 0.7, max(trainable_params_m) * 1.5)
    axes[1, 1].set_ylim(0, max(accuracy) + 8)
    axes[1, 1].grid(alpha=0.25)
    annotate_points(
        axes[1, 1],
        df["model_name"],
        labels,
        trainable_params_m,
        accuracy,
        {
            "swin_tiny_pretrained": (6, -14),
            "swin_small_pretrained": (6, 6),
            "swin_scratch": (6, 0),
        },
    )

    fig.suptitle("Homework 5 Problem 2: Fine-Tuning vs. Scratch Tradeoffs", fontsize=14)
    fig.tight_layout()
    fig.savefig(P2 / "problem2_tradeoffs.png", dpi=200)
    plt.close(fig)


def main() -> None:
    maybe_plot_history(P1 / "problem1_history.csv", "problem1_loss_curves.png", "Homework 5 Problem 1 Training Curves")
    maybe_plot_history(P2 / "problem2_history.csv", "problem2_training_curves.png", "Homework 5 Problem 2 Training Curves")
    plot_problem1_tradeoffs()
    plot_problem2_tradeoffs()


if __name__ == "__main__":
    main()
