from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

try:
    from transformers import AutoImageProcessor, SwinForImageClassification
except ModuleNotFoundError:
    AutoImageProcessor = None
    SwinForImageClassification = None


torch.set_num_threads(2)
SEED = 4106
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "Results_Problem_2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100


@dataclass
class SwinConfig:
    model_name: str
    pretrained: bool
    frozen_backbone: bool
    huggingface_id: str | None
    batch_size: int
    epochs: int
    learning_rate: float
    image_size: int


@dataclass
class SwinResult:
    model_name: str
    pretrained: bool
    frozen_backbone: bool
    total_parameter_count: int | None
    parameter_count: int | None
    train_time_per_epoch_sec: float | None
    final_train_loss: float | None
    final_val_loss: float | None
    test_accuracy_pct: float | None
    notes: str = ""


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def export_default_templates() -> None:
    ensure_dirs()
    rows = [
        SwinResult("swin_tiny_pretrained", True, True, None, None, None, None, None, None, "fill after run"),
        SwinResult("swin_small_pretrained", True, True, None, None, None, None, None, None, "fill after run"),
        SwinResult("swin_scratch", False, False, None, None, None, None, None, None, "fill after run"),
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
        writer.writerow(["model_name", "epoch", "train_loss", "val_loss", "train_accuracy_pct", "val_accuracy_pct", "epoch_seconds"])


def get_loaders(batch_size: int, image_size: int = 224, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_dataset = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, use_hf: bool) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(pixel_values=inputs).logits if use_hf else model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += logits.argmax(dim=1).eq(targets).sum().item()
            total_samples += targets.size(0)
    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy_pct": 100.0 * total_correct / max(1, total_samples),
    }


def train_model(model: nn.Module, config: SwinConfig, train_loader: DataLoader, test_loader: DataLoader, use_hf: bool) -> tuple[list[dict[str, float]], SwinResult]:
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    start_all = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        start_epoch = time.perf_counter()
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=inputs).logits if use_hf else model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_correct += logits.argmax(dim=1).eq(targets).sum().item()
            total_samples += targets.size(0)
        val_metrics = evaluate_model(model, test_loader, criterion, use_hf)
        history.append({
            "model_name": config.model_name,
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_samples),
            "val_loss": val_metrics["loss"],
            "train_accuracy_pct": 100.0 * total_correct / max(1, total_samples),
            "val_accuracy_pct": val_metrics["accuracy_pct"],
            "epoch_seconds": time.perf_counter() - start_epoch,
        })
        latest = history[-1]
        print(
            f"{config.model_name} epoch {epoch}/{config.epochs}: "
            f"train_loss={latest['train_loss']:.4f}, "
            f"val_loss={latest['val_loss']:.4f}, "
            f"val_acc={latest['val_accuracy_pct']:.2f}%, "
            f"seconds={latest['epoch_seconds']:.1f}",
            flush=True,
        )

    average_epoch_seconds = (time.perf_counter() - start_all) / max(1, config.epochs)
    final = history[-1]
    result = SwinResult(
        model_name=config.model_name,
        pretrained=config.pretrained,
        frozen_backbone=config.frozen_backbone,
        total_parameter_count=count_all_parameters(model),
        parameter_count=count_parameters(model),
        train_time_per_epoch_sec=average_epoch_seconds,
        final_train_loss=final["train_loss"],
        final_val_loss=final["val_loss"],
        test_accuracy_pct=final["val_accuracy_pct"],
        notes=f"device={DEVICE.type}",
    )
    return history, result


def append_history(rows: list[dict[str, float]], replace_model: str | None = None) -> None:
    history_path = RESULTS_DIR / "problem2_history.csv"
    existing: list[dict[str, object]] = []
    if replace_model is not None and history_path.exists() and history_path.stat().st_size > 0:
        current = pd.read_csv(history_path)
        if "model_name" in current.columns:
            current = current[current["model_name"] != replace_model]
        existing = current.to_dict(orient="records")

    exists = history_path.exists()
    mode = "w" if replace_model is not None else "a"
    with history_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if mode == "w" or not exists or history_path.stat().st_size == 0:
            writer.writeheader()
        for row in existing:
            writer.writerow(row)
        for row in rows:
            writer.writerow(row)


def write_summary(rows: list[SwinResult]) -> None:
    summary_path = RESULTS_DIR / "problem2_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def default_configs() -> list[SwinConfig]:
    return [
        SwinConfig("swin_tiny_pretrained", True, True, "microsoft/swin-tiny-patch4-window7-224", 32, 5, 2e-5, 224),
        SwinConfig("swin_small_pretrained", True, True, "microsoft/swin-small-patch4-window7-224", 32, 5, 2e-5, 224),
        SwinConfig("swin_scratch", False, False, None, 32, 5, 1e-3, 224),
    ]


def require_transformers() -> None:
    if AutoImageProcessor is None or SwinForImageClassification is None:
        raise RuntimeError(
            "The 'transformers' package is not installed. Install it before running pretrained Swin experiments."
        )


def build_pretrained_model(config: SwinConfig) -> nn.Module:
    require_transformers()
    model = SwinForImageClassification.from_pretrained(
        config.huggingface_id,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    if config.frozen_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    return model


def build_scratch_model() -> nn.Module:
    model = models.swin_t(weights=None)
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homework 5 Problem 2: Swin models")
    parser.add_argument("--export-templates", action="store_true")
    parser.add_argument("--run-pretrained", type=str, default=None, help="swin_tiny_pretrained or swin_small_pretrained")
    parser.add_argument("--run-scratch", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--reset-history", action="store_true")
    return parser.parse_args()


def maybe_reset_history() -> None:
    history_path = RESULTS_DIR / "problem2_history.csv"
    if history_path.exists():
        history_path.unlink()


def override_config(config: SwinConfig, args: argparse.Namespace) -> SwinConfig:
    return SwinConfig(
        model_name=config.model_name,
        pretrained=config.pretrained,
        frozen_backbone=config.frozen_backbone,
        huggingface_id=config.huggingface_id,
        batch_size=args.batch_size or config.batch_size,
        epochs=args.epochs or config.epochs,
        learning_rate=config.learning_rate,
        image_size=args.image_size or config.image_size,
    )


def refresh_summary(results: list[SwinResult]) -> None:
    summary_path = RESULTS_DIR / "problem2_summary.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        by_name = {row.model_name: asdict(row) for row in results}
        merged = []
        for record in existing.to_dict(orient="records"):
            merged.append(by_name.get(record["model_name"], record))
        for row in merged:
            row.setdefault("total_parameter_count", None)
        write_summary([SwinResult(**row) for row in merged])
        return
    write_summary(results)


def run_pretrained(config_name: str, args: argparse.Namespace) -> SwinResult:
    config_map = {config.model_name: config for config in default_configs() if config.pretrained}
    if config_name not in config_map:
        raise ValueError(f"Unknown pretrained config: {config_name}")
    config = override_config(config_map[config_name], args)
    train_loader, test_loader = get_loaders(config.batch_size, image_size=config.image_size, num_workers=args.num_workers)
    history, result = train_model(build_pretrained_model(config), config, train_loader, test_loader, use_hf=True)
    result.notes = f"{result.notes},image_size={config.image_size},batch_size={config.batch_size}"
    append_history(history, replace_model=config.model_name)
    return result


def run_scratch(args: argparse.Namespace) -> SwinResult:
    config = override_config(next(config for config in default_configs() if config.model_name == "swin_scratch"), args)
    train_loader, test_loader = get_loaders(config.batch_size, image_size=config.image_size, num_workers=args.num_workers)
    history, result = train_model(build_scratch_model(), config, train_loader, test_loader, use_hf=False)
    result.notes = f"{result.notes},image_size={config.image_size},batch_size={config.batch_size}"
    append_history(history, replace_model=config.model_name)
    return result


def main() -> None:
    args = parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    else:
        torch.set_num_threads(min(8, os.cpu_count() or 2))
    set_seed()
    ensure_dirs()
    if args.export_templates:
        export_default_templates()
        print(f"Exported CSV templates to {RESULTS_DIR}")
        return
    if args.reset_history:
        maybe_reset_history()

    results: list[SwinResult] = []
    if args.run_pretrained:
        results.append(run_pretrained(args.run_pretrained, args))
    if args.run_scratch:
        results.append(run_scratch(args))
    if not results:
        print("Nothing ran. Use --run-pretrained <name> or --run-scratch.")
        return

    refresh_summary(results)
    print(pd.DataFrame([asdict(row) for row in results]).to_string(index=False))


if __name__ == "__main__":
    main()
