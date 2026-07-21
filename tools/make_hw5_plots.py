"""
Plot helper for Homework 5 CSV outputs.

Expected inputs:
- Results_Problem_1/problem1_history.csv
- Results_Problem_1/problem1_summary.csv
- Results_Problem_2/problem2_history.csv
- Results_Problem_2/problem2_summary.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    "swin_tiny_cached_extended": "Swin-Tiny\ncached head",
    "swin_small_cached_extended": "Swin-Small\ncached head",
}


def display_name(model_name: str) -> str:
    return DISPLAY_NAMES.get(str(model_name), str(model_name))


def readable_epoch_ticks(epochs: list[int | float]) -> list[int | float]:
    if len(epochs) <= 20:
        return epochs
    first = int(min(epochs))
    last = int(max(epochs))
    ticks = [first]
    ticks.extend(range(((first // 10) + 1) * 10, last + 1, 10))
    if ticks[-1] != last:
        ticks.append(last)
    return ticks


def readable_label_epochs(epochs: list[int | float]) -> set[int | float]:
    if len(epochs) <= 12:
        return set(epochs)
    first = int(min(epochs))
    last = int(max(epochs))
    labels = {first, last}
    step = 10 if len(epochs) > 40 else 5
    labels.update(range(((first // step) + 1) * step, last + 1, step))
    return labels


def maybe_plot_history(csv_path: Path, output_name: str, title: str) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = sorted(df["epoch"].dropna().unique())
    epoch_ticks = readable_epoch_ticks(epochs)
    for model_name, group in df.groupby("model_name"):
        if "val_loss" in group.columns:
            axes[0].plot(group["epoch"], group["val_loss"], marker="o", markersize=6, linewidth=1.8, label=f"{model_name} val")
        if "train_loss" in group.columns:
            axes[0].plot(group["epoch"], group["train_loss"], marker="x", markersize=6, linewidth=1.8, alpha=0.72, label=f"{model_name} train")
        if "val_accuracy_pct" in group.columns:
            axes[1].plot(group["epoch"], group["val_accuracy_pct"], marker="o", markersize=6, linewidth=1.8, label=f"{model_name} val")
        if "train_accuracy_pct" in group.columns:
            axes[1].plot(group["epoch"], group["train_accuracy_pct"], marker="x", markersize=6, linewidth=1.8, alpha=0.72, label=f"{model_name} train")

    axes[0].set_title(f"{title}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(epoch_ticks)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)

    axes[1].set_title(f"{title}: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xticks(epoch_ticks)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(csv_path.parent / output_name, dpi=200)
    plt.close(fig)


def maybe_plot_epoch_details(csv_path: Path, output_name: str, title: str) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns:
        return

    model_names = list(df["model_name"].drop_duplicates())
    cols = min(3, len(model_names))
    rows = math.ceil(len(model_names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows), squeeze=False)
    epochs = sorted(df["epoch"].dropna().unique())
    epoch_ticks = readable_epoch_ticks(epochs)

    for ax, model_name in zip(axes.flat, model_names, strict=False):
        group = df[df["model_name"] == model_name].sort_values("epoch")
        ax.plot(
            group["epoch"],
            group["train_accuracy_pct"],
            marker="x",
            markersize=7,
            linewidth=2.0,
            label="train accuracy",
            color="#f58518",
        )
        ax.plot(
            group["epoch"],
            group["val_accuracy_pct"],
            marker="o",
            markersize=7,
            linewidth=2.0,
            label="test accuracy",
            color="#4c78a8",
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"{row['val_accuracy_pct']:.1f}",
                (row["epoch"], row["val_accuracy_pct"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color="#4c78a8",
            )
        ax.set_title(display_name(model_name).replace("\n", " "))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(epoch_ticks)
        ax.set_ylim(
            max(0, min(df["train_accuracy_pct"].min(), df["val_accuracy_pct"].min()) - 5),
            min(100, max(df["train_accuracy_pct"].max(), df["val_accuracy_pct"].max()) + 8),
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes.flat[len(model_names):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(csv_path.parent / output_name, dpi=200)
    plt.close(fig)


def maybe_plot_metric_grid(csv_path: Path, output_name: str, title: str) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns:
        return

    metrics = [
        ("train_loss", "Train Loss", "#f58518"),
        ("val_loss", "Test Loss", "#4c78a8"),
        ("train_accuracy_pct", "Train Accuracy (%)", "#e45756"),
        ("val_accuracy_pct", "Test Accuracy (%)", "#54a24b"),
    ]
    model_names = list(df["model_name"].drop_duplicates())
    epochs = sorted(df["epoch"].dropna().unique())
    epoch_ticks = readable_epoch_ticks(epochs)
    label_epochs = readable_label_epochs(epochs)
    fig, axes = plt.subplots(len(model_names), len(metrics), figsize=(4.2 * len(metrics), 2.45 * len(model_names)), squeeze=False)

    for row_index, model_name in enumerate(model_names):
        group = df[df["model_name"] == model_name].sort_values("epoch")
        for col_index, (column, label, color) in enumerate(metrics):
            ax = axes[row_index, col_index]
            ax.plot(group["epoch"], group[column], marker="o", markersize=3.0, linewidth=1.25, color=color, alpha=0.72)
            ax.scatter(
                group["epoch"],
                group[column],
                s=24 if len(epochs) > 40 else 34,
                color=color,
                edgecolors="black",
                linewidths=0.4,
                zorder=4,
            )
            for _, row in group.iterrows():
                if row["epoch"] not in label_epochs:
                    continue
                value = row[column]
                text = f"{value:.1f}" if "accuracy" in column else f"{value:.2f}"
                ax.annotate(text, (row["epoch"], value), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6)
            if row_index == 0:
                ax.set_title(label)
            if col_index == 0:
                ax.set_ylabel(display_name(model_name).replace("\n", " "))
            ax.set_xlabel("Epoch")
            ax.set_xticks(epoch_ticks)
            ax.grid(alpha=0.22)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(csv_path.parent / output_name, dpi=200)
    plt.close(fig)


def maybe_plot_interpolated_trends(csv_path: Path, output_name: str, title: str) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for model_name, group in df.groupby("model_name"):
        group = group.sort_values("epoch")
        dense_epoch = np.linspace(group["epoch"].min(), group["epoch"].max(), 240)
        train_loss = np.interp(dense_epoch, group["epoch"], group["train_loss"])
        test_loss = np.interp(dense_epoch, group["epoch"], group["val_loss"])
        train_acc = np.interp(dense_epoch, group["epoch"], group["train_accuracy_pct"])
        test_acc = np.interp(dense_epoch, group["epoch"], group["val_accuracy_pct"])

        label = display_name(model_name).replace("\n", " ")
        axes[0].plot(dense_epoch, train_loss, marker="o", markersize=2.6, linewidth=1.5, alpha=0.68, label=f"{label} train")
        axes[0].plot(dense_epoch, test_loss, marker="o", markersize=2.6, linewidth=1.5, alpha=0.68, label=f"{label} test")
        axes[0].scatter(group["epoch"], group["train_loss"], s=52, alpha=0.98, edgecolors="black", linewidths=0.5, zorder=4)
        axes[0].scatter(group["epoch"], group["val_loss"], s=52, alpha=0.98, edgecolors="black", linewidths=0.5, zorder=4)

        axes[1].plot(dense_epoch, train_acc, marker="o", markersize=2.6, linewidth=1.5, alpha=0.68, label=f"{label} train")
        axes[1].plot(dense_epoch, test_acc, marker="o", markersize=2.6, linewidth=1.5, alpha=0.68, label=f"{label} test")
        axes[1].scatter(group["epoch"], group["train_accuracy_pct"], s=52, alpha=0.98, edgecolors="black", linewidths=0.5, zorder=4)
        axes[1].scatter(group["epoch"], group["val_accuracy_pct"], s=52, alpha=0.98, edgecolors="black", linewidths=0.5, zorder=4)

    axes[0].set_title("Loss Trends")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    axes[1].set_title("Accuracy Trends")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)

    fig.suptitle(f"{title}\nDense lines interpolate between measured epoch markers", fontsize=13)
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
    maybe_plot_epoch_details(P1 / "problem1_history.csv", "problem1_epoch_detail.png", "Homework 5 Problem 1: Per-Model Accuracy Points")
    maybe_plot_epoch_details(P2 / "problem2_history.csv", "problem2_epoch_detail.png", "Homework 5 Problem 2: Per-Model Accuracy Points")
    maybe_plot_metric_grid(P1 / "problem1_history.csv", "problem1_metric_grid.png", "Homework 5 Problem 1: Per-Epoch Metric Grid")
    maybe_plot_metric_grid(P2 / "problem2_history.csv", "problem2_metric_grid.png", "Homework 5 Problem 2: Per-Epoch Metric Grid")
    maybe_plot_history(P2 / "problem2_extended_history.csv", "problem2_extended_training_curves.png", "Homework 5 Problem 2 Extended Cached-Head Runs")
    maybe_plot_metric_grid(P2 / "problem2_extended_history.csv", "problem2_extended_metric_grid.png", "Homework 5 Problem 2: Extended 100-Epoch Metric Grid")
    maybe_plot_interpolated_trends(P1 / "problem1_history.csv", "problem1_interpolated_trends.png", "Homework 5 Problem 1: Reference-Style Trend Curves")
    maybe_plot_interpolated_trends(P2 / "problem2_history.csv", "problem2_interpolated_trends.png", "Homework 5 Problem 2: Reference-Style Trend Curves")
    maybe_plot_interpolated_trends(P2 / "problem2_extended_history.csv", "problem2_extended_interpolated_trends.png", "Homework 5 Problem 2: Extended 100-Epoch Trend Curves")
    plot_problem1_tradeoffs()
    plot_problem2_tradeoffs()


if __name__ == "__main__":
    main()
