"""
Homework 5 Problem 2 scaffold.

This file provides a starting point for:
- fine-tuning pretrained Swin-Tiny and Swin-Small on CIFAR-100
- training a scratch Swin-style model
- exporting metrics for the report
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "Results_Problem_2"


@dataclass
class SwinResult:
    model_name: str
    pretrained: bool
    frozen_backbone: bool
    parameter_count: int | None
    train_time_per_epoch_sec: float | None
    final_train_loss: float | None
    final_val_loss: float | None
    test_accuracy_pct: float | None
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homework 5 Swin scaffold")
    parser.add_argument("--export-templates", action="store_true")
    parser.add_argument("--run-pretrained", action="store_true")
    parser.add_argument("--run-scratch", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def export_default_templates() -> None:
    ensure_dirs()
    rows = [
        SwinResult("swin_tiny_pretrained", True, True, None, None, None, None, None, "TODO"),
        SwinResult("swin_small_pretrained", True, True, None, None, None, None, None, "TODO"),
        SwinResult("swin_scratch", False, False, None, None, None, None, None, "TODO"),
    ]
    summary_path = RESULTS_DIR / "problem2_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    history_path = RESULTS_DIR / "problem2_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_name", "epoch", "train_loss", "val_loss", "train_accuracy_pct", "val_accuracy_pct"])


def main() -> None:
    args = parse_args()
    if args.export_templates:
        export_default_templates()
        print(f"Exported CSV templates to {RESULTS_DIR}")
        return

    # TODO: add training entry points for pretrained and scratch Swin models.
    print("Scaffold only. Use --export-templates to create CSV files.")


if __name__ == "__main__":
    main()
