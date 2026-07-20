from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


torch.set_num_threads(2)
SEED = 4106
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "Results_Problem_1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 32
NUM_CLASSES = 100


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


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return 100.0 * predictions.eq(targets).sum().item() / max(1, targets.numel())


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTClassifier(nn.Module):
    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def build_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_cifar100_loaders(batch_size: int, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_dataset = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += logits.argmax(dim=1).eq(targets).sum().item()
            total_samples += targets.size(0)
    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy_pct": 100.0 * total_correct / max(1, total_samples),
    }


def train_model(model: nn.Module, config: ExperimentConfig, train_loader: DataLoader, test_loader: DataLoader) -> tuple[list[dict[str, float]], ExperimentResult]:
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
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
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_correct += logits.argmax(dim=1).eq(targets).sum().item()
            total_samples += targets.size(0)
        test_metrics = evaluate_model(model, test_loader, criterion)
        history.append({
            "model_name": config.model_name,
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_samples),
            "val_loss": test_metrics["loss"],
            "train_accuracy_pct": 100.0 * total_correct / max(1, total_samples),
            "val_accuracy_pct": test_metrics["accuracy_pct"],
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
    result = ExperimentResult(
        model_name=config.model_name,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        parameter_count=count_parameters(model),
        flops_forward=estimate_flops(config),
        train_time_per_epoch_sec=average_epoch_seconds,
        final_train_loss=final["train_loss"],
        final_val_loss=final["val_loss"],
        test_accuracy_pct=final["val_accuracy_pct"],
        notes=f"device={DEVICE.type}",
    )
    return history, result


def estimate_flops(config: ExperimentConfig) -> float | None:
    if config.model_name == "resnet18":
        return 1.1e9
    if config.patch_size is None or config.embed_dim is None or config.depth is None or config.heads is None or config.mlp_dim is None:
        return None
    tokens = (IMAGE_SIZE // config.patch_size) ** 2 + 1
    d = config.embed_dim
    depth = config.depth
    attn = depth * (4 * tokens * d * d + 2 * tokens * tokens * d)
    mlp = depth * (2 * tokens * d * config.mlp_dim)
    head = d * NUM_CLASSES
    return float(attn + mlp + head)


def append_history_rows(rows: list[dict[str, float]]) -> None:
    history_path = RESULTS_DIR / "problem1_history.csv"
    exists = history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists or history_path.stat().st_size == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(rows: list[ExperimentResult]) -> None:
    summary_path = RESULTS_DIR / "problem1_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def export_default_templates() -> None:
    ensure_dirs()
    summary_rows = [
        ExperimentResult("vit_patch4_dim256_d4_h4", 4, 256, 4, 4, 1024, None, None, None, None, None, None, "fill after run"),
        ExperimentResult("vit_patch4_dim512_d8_h8", 4, 512, 8, 8, 2048, None, None, None, None, None, None, "fill after run"),
        ExperimentResult("vit_patch8_dim256_d4_h4", 8, 256, 4, 4, 1024, None, None, None, None, None, None, "fill after run"),
        ExperimentResult("vit_patch8_dim512_d8_h8", 8, 512, 8, 8, 2048, None, None, None, None, None, None, "fill after run"),
        ExperimentResult("resnet18", None, None, None, None, None, None, None, None, None, None, None, "fill after run"),
    ]
    write_summary(summary_rows)
    history_path = RESULTS_DIR / "problem1_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_name", "epoch", "train_loss", "val_loss", "train_accuracy_pct", "val_accuracy_pct", "epoch_seconds"])


def default_vit_configs() -> list[ExperimentConfig]:
    return [
        ExperimentConfig("vit_patch4_dim256_d4_h4", 4, 256, 4, 4, 1024, 64, 10, 1e-3),
        ExperimentConfig("vit_patch4_dim512_d8_h8", 4, 512, 8, 8, 2048, 64, 10, 1e-3),
        ExperimentConfig("vit_patch8_dim256_d4_h4", 8, 256, 4, 4, 1024, 64, 10, 1e-3),
        ExperimentConfig("vit_patch8_dim512_d8_h8", 8, 512, 8, 8, 2048, 64, 10, 1e-3),
    ]


def build_vit_from_config(config: ExperimentConfig) -> ViTClassifier:
    return ViTClassifier(
        patch_size=config.patch_size or 4,
        embed_dim=config.embed_dim or 256,
        depth=config.depth or 4,
        heads=config.heads or 4,
        mlp_dim=config.mlp_dim or 1024,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homework 5 Problem 1: ViT vs ResNet-18")
    parser.add_argument("--export-templates", action="store_true")
    parser.add_argument("--run-all-vit", action="store_true")
    parser.add_argument("--run-vit", type=str, default=None, help="Specific ViT config name to run")
    parser.add_argument("--run-resnet", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--reset-history", action="store_true")
    return parser.parse_args()


def maybe_reset_history() -> None:
    history_path = RESULTS_DIR / "problem1_history.csv"
    if history_path.exists():
        history_path.unlink()


def select_vit_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    configs = default_vit_configs()
    if args.run_vit:
        configs = [config for config in configs if config.model_name == args.run_vit]
        if not configs:
            raise ValueError(f"Unknown ViT config: {args.run_vit}")
    if args.epochs is not None or args.batch_size is not None:
        updated: list[ExperimentConfig] = []
        for config in configs:
            updated.append(ExperimentConfig(
                model_name=config.model_name,
                patch_size=config.patch_size,
                embed_dim=config.embed_dim,
                depth=config.depth,
                heads=config.heads,
                mlp_dim=config.mlp_dim,
                batch_size=args.batch_size or config.batch_size,
                epochs=args.epochs or config.epochs,
                learning_rate=config.learning_rate,
            ))
        configs = updated
    return configs


def run_vit_experiments(args: argparse.Namespace) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for config in select_vit_configs(args):
        train_loader, test_loader = get_cifar100_loaders(config.batch_size, args.num_workers)
        history, result = train_model(build_vit_from_config(config), config, train_loader, test_loader)
        append_history_rows(history)
        results.append(result)
    return results


def run_resnet_experiment(args: argparse.Namespace) -> ExperimentResult:
    batch_size = args.batch_size or 64
    epochs = args.epochs or 10
    config = ExperimentConfig("resnet18", None, None, None, None, None, batch_size, epochs, 1e-3)
    train_loader, test_loader = get_cifar100_loaders(config.batch_size, args.num_workers)
    history, result = train_model(build_resnet18(), config, train_loader, test_loader)
    append_history_rows(history)
    return result


def refresh_summary_with_results(results: list[ExperimentResult]) -> None:
    summary_path = RESULTS_DIR / "problem1_summary.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        by_name = {result.model_name: asdict(result) for result in results}
        rows = []
        for record in existing.to_dict(orient="records"):
            if record["model_name"] in by_name:
                rows.append(by_name[record["model_name"]])
            else:
                rows.append(record)
        final_rows = [ExperimentResult(**row) for row in rows]
        write_summary(final_rows)
        return
    write_summary(results)


def main() -> None:
    args = parse_args()
    set_seed()
    ensure_dirs()
    if args.export_templates:
        export_default_templates()
        print(f"Exported templates to {RESULTS_DIR}")
        return
    if args.reset_history:
        maybe_reset_history()

    collected: list[ExperimentResult] = []
    if args.run_all_vit or args.run_vit:
        collected.extend(run_vit_experiments(args))
    if args.run_resnet:
        collected.append(run_resnet_experiment(args))

    if not collected:
        print("Nothing ran. Use --run-all-vit, --run-vit <name>, or --run-resnet.")
        return

    refresh_summary_with_results(collected)
    print(pd.DataFrame([asdict(result) for result in collected]).to_string(index=False))


if __name__ == "__main__":
    main()
