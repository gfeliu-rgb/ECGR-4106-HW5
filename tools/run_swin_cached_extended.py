"""Run extended frozen-backbone Swin head training with cached features.

This is an additional analysis artifact for Homework 5. The required Problem 2
table remains the 5-epoch full-model fine-tuning run. This script creates more
real measured epoch points for the frozen pretrained Swin heads without rerunning
the expensive backbone on every epoch.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from transformers import SwinForImageClassification


SEED = 4106
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "Results_Problem_2"
CACHE_DIR = RESULTS_DIR / "cached_features"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100


@dataclass
class ExtendedResult:
    model_name: str
    source_model: str
    epochs: int
    batch_size: int
    learning_rate: float
    feature_dim: int
    train_samples: int
    test_samples: int
    feature_extract_time_sec: float
    head_train_time_sec: float
    final_train_loss: float
    final_test_loss: float
    final_test_accuracy_pct: float
    notes: str


MODEL_IDS = {
    "swin_tiny_cached_extended": "microsoft/swin-tiny-patch4-window7-224",
    "swin_small_cached_extended": "microsoft/swin-small-patch4-window7-224",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_subset_indices(targets: list[int], total_size: int) -> list[int]:
    if total_size <= 0 or total_size >= len(targets):
        return list(range(len(targets)))
    per_class = max(1, total_size // NUM_CLASSES)
    selected: list[int] = []
    counts = {class_id: 0 for class_id in range(NUM_CLASSES)}
    for index, target in enumerate(targets):
        if counts[target] < per_class:
            selected.append(index)
            counts[target] += 1
        if len(selected) >= per_class * NUM_CLASSES:
            break
    return selected[:total_size]


def get_loader(train: bool, batch_size: int, image_size: int, num_workers: int, sample_limit: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    dataset = datasets.CIFAR100(root=DATA_DIR, train=train, download=True, transform=transform)
    if 0 < sample_limit < len(dataset):
        dataset = Subset(dataset, balanced_subset_indices(dataset.targets, sample_limit))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def cache_path(model_name: str, split: str, sample_count: int) -> Path:
    return CACHE_DIR / f"{model_name}_{split}_{sample_count}_samples.pt"


def extract_features(model_name: str, model_id: str, split: str, loader: DataLoader, force: bool) -> tuple[torch.Tensor, torch.Tensor, float]:
    path = cache_path(model_name, split, len(loader.dataset))
    if path.exists() and not force:
        payload = torch.load(path, map_location="cpu")
        return payload["features"], payload["targets"], 0.0

    model = SwinForImageClassification.from_pretrained(
        model_id,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    model.to(DEVICE)
    model.eval()

    features: list[torch.Tensor] = []
    targets_out: list[torch.Tensor] = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(DEVICE)
            outputs = model.swin(pixel_values=inputs)
            pooled = outputs.pooler_output
            features.append(pooled.cpu())
            targets_out.append(targets.cpu())
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
                print(f"{model_name} {split} feature batch {batch_index}/{len(loader)}", flush=True)

    elapsed = time.perf_counter() - start
    feature_tensor = torch.cat(features, dim=0)
    target_tensor = torch.cat(targets_out, dim=0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"features": feature_tensor, "targets": target_tensor}, path)
    return feature_tensor, target_tensor, elapsed


def evaluate_head(head: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    head.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for features, targets in loader:
            logits = head(features)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += logits.argmax(dim=1).eq(targets).sum().item()
            total_samples += targets.size(0)
    return total_loss / max(1, total_samples), 100.0 * total_correct / max(1, total_samples)


def train_head(
    model_name: str,
    model_id: str,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[list[dict[str, float]], ExtendedResult]:
    feature_dim = train_features.shape[1]
    head = nn.Linear(feature_dim, NUM_CLASSES)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_features, test_targets), batch_size=batch_size, shuffle=False)
    history: list[dict[str, float]] = []
    start_train = time.perf_counter()

    for epoch in range(1, epochs + 1):
        head.train()
        start_epoch = time.perf_counter()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_correct += logits.argmax(dim=1).eq(targets).sum().item()
            total_samples += targets.size(0)

        test_loss, test_acc = evaluate_head(head, test_loader, criterion)
        row = {
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_samples),
            "val_loss": test_loss,
            "train_accuracy_pct": 100.0 * total_correct / max(1, total_samples),
            "val_accuracy_pct": test_acc,
            "epoch_seconds": time.perf_counter() - start_epoch,
        }
        history.append(row)
        print(
            f"{model_name} epoch {epoch}/{epochs}: "
            f"train_loss={row['train_loss']:.4f}, test_loss={row['val_loss']:.4f}, "
            f"test_acc={row['val_accuracy_pct']:.2f}%, seconds={row['epoch_seconds']:.2f}",
            flush=True,
        )

    final = history[-1]
    result = ExtendedResult(
        model_name=model_name,
        source_model=model_id,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        feature_dim=feature_dim,
        train_samples=len(train_targets),
        test_samples=len(test_targets),
        feature_extract_time_sec=0.0,
        head_train_time_sec=time.perf_counter() - start_train,
        final_train_loss=final["train_loss"],
        final_test_loss=final["val_loss"],
        final_test_accuracy_pct=final["val_accuracy_pct"],
        notes=f"cached frozen-backbone head training; device={DEVICE.type}",
    )
    return history, result


def write_history(rows: list[dict[str, float]]) -> None:
    path = RESULTS_DIR / "problem2_extended_history.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[ExtendedResult]) -> None:
    path = RESULTS_DIR / "problem2_extended_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended cached Swin head training for Homework 5")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=10_000)
    parser.add_argument("--test-samples", type=int, default=2_000)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--models", nargs="+", default=list(MODEL_IDS.keys()), choices=list(MODEL_IDS.keys()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_history: list[dict[str, float]] = []
    all_results: list[ExtendedResult] = []

    train_loader = get_loader(True, args.feature_batch_size, args.image_size, args.num_workers, args.train_samples)
    test_loader = get_loader(False, args.feature_batch_size, args.image_size, args.num_workers, args.test_samples)

    for model_name in args.models:
        model_id = MODEL_IDS[model_name]
        print(f"Preparing cached features for {model_name} from {model_id}", flush=True)
        train_features, train_targets, train_extract = extract_features(model_name, model_id, "train", train_loader, args.force_extract)
        test_features, test_targets, test_extract = extract_features(model_name, model_id, "test", test_loader, args.force_extract)
        history, result = train_head(
            model_name,
            model_id,
            train_features,
            train_targets,
            test_features,
            test_targets,
            args.epochs,
            args.batch_size,
            args.learning_rate,
        )
        result.feature_extract_time_sec = train_extract + test_extract
        all_history.extend(history)
        all_results.append(result)

    write_history(all_history)
    write_summary(all_results)
    print("Wrote extended history and summary to Results_Problem_2", flush=True)


if __name__ == "__main__":
    main()
