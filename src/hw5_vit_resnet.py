"""
Homework 5 Problem 1 scaffold.

This file provides a clean starting point for:
- Vision Transformer experiments on CIFAR-100
- ResNet-18 baseline training and evaluation
- CSV export for summary/history tables

Fill in the remaining TODO sections with your final implementation details,
hyperparameter choices, and experiment runs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "Results_Problem_1"


@dataclass
class ExperimentConfig:
    model_name: str
    patch_size: int | None
    embed_dim: int | None
    depth: int | None
    heads: int | None
    mlp_dim: int | None
    batch_size: int
    epochs: int
    learning_rate: float


@dataclass
class ExperimentResult:
    model_name: str
    patch_size: int | None
    embed_dim: int | None
    depth: int | None
    heads: int | None
    mlp_dim: int | None
    parameter_count: int | None
    flops_forward: float | None
    train_time_per_epoch_sec: float | None
    final_train_loss: float | None
    final_val_loss: float | None
    test_accuracy_pct: float | None
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homework 5 ViT vs ResNet scaffold")
    parser.add_argument("--export-templates", action="store_true")
    parser.add_argument("--run-vit", action="store_true")
    parser.add_argument("--run-resnet", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def export_summary_template(rows: Iterable[ExperimentResult]) -> None:
    path = RESULTS_DIR / "problem1_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(next(iter(rows))).keys()))
        writer.writeheader()
        handle.seek(0, 2)


def export_default_templates() -> None:
    ensure_dirs()
    summary_rows = [
        ExperimentResult("vit_patch4_dim256_d4_h4", 4, 256, 4, 4, 1024, None, None, None, None, None, None, "TODO"),
        ExperimentResult("vit_patch4_dim512_d8_h8", 4, 512, 8, 8, 2048, None, None, None, None, None, None, "TODO"),
        ExperimentResult("vit_patch8_dim256_d4_h4", 8, 256, 4, 4, 1024, None, None, None, None, None, None, "TODO"),
        ExperimentResult("vit_patch8_dim512_d8_h8", 8, 512, 8, 8, 2048, None, None, None, None, None, None, "TODO"),
        ExperimentResult("resnet18", None, None, None, None, None, None, None, None, None, None, None, "TODO"),
    ]
    summary_path = RESULTS_DIR / "problem1_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summary_rows[0]).keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(asdict(row))

    history_path = RESULTS_DIR / "problem1_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_name", "epoch", "train_loss", "val_loss", "train_accuracy_pct", "val_accuracy_pct"])


def main() -> None:
    args = parse_args()
    if args.export_templates:
        export_default_templates()
        print(f"Exported CSV templates to {RESULTS_DIR}")
        return

    # TODO: add training entry points for ViT and ResNet-18.
    print("Scaffold only. Use --export-templates to create CSV files.")


if __name__ == "__main__":
    main()
