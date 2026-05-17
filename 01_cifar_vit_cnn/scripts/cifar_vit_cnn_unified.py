from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (  # noqa: E402
    plot_grouped_history_curves,
    plot_speed_accuracy_tradeoff,
    save_classification_artifacts,
)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, embed_dim: int = 256):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.attn_map: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, tokens, 3, self.heads, channels // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_map = attn.detach()
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(bsz, tokens, channels)
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CIFARViT(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        dim = 256
        self.patch = PatchEmbedding(embed_dim=dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 65, dim))
        self.blocks = nn.Sequential(*[Block(dim, heads=8) for _ in range(8)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x = self.patch(x)
        x = torch.cat([self.cls.expand(bsz, -1, -1), x], dim=1)
        x = x + self.pos
        x = self.blocks(x)
        return self.norm(x[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(256, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.forward_features(x)))


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], float(lam)


def stratified_subset_indices(dataset: torchvision.datasets.CIFAR10, fraction: float, seed: int) -> list[int]:
    if fraction >= 0.999999:
        return list(range(len(dataset)))
    targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    for label in sorted(np.unique(targets).tolist()):
        cls_indices = np.where(targets == label)[0]
        k = max(1, int(round(len(cls_indices) * fraction)))
        kept.append(rng.choice(cls_indices, size=k, replace=False))
    merged = np.concatenate(kept)
    rng.shuffle(merged)
    return merged.astype(int).tolist()


def get_dataloaders(
    data_root: Path,
    batch_size: int,
    fraction: float,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_full = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=train_transform)
    val_source = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=eval_transform)
    test_set = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=eval_transform)

    subset_indices = stratified_subset_indices(train_full, fraction, seed)
    subset_for_split = Subset(train_full, subset_indices)
    val_source_subset = Subset(val_source, subset_indices)
    val_size = max(1, int(round(len(subset_for_split) * 0.1)))
    train_size = len(subset_for_split) - val_size
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(subset_for_split), generator=generator).tolist()
    train_subset = Subset(subset_for_split, perm[:train_size])
    val_subset = Subset(val_source_subset, perm[train_size:])

    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, len(train_subset)


def build_model(model_name: str) -> nn.Module:
    if model_name == "vit":
        return CIFARViT(num_classes=10)
    if model_name == "cnn":
        return SimpleCNN(num_classes=10)
    raise ValueError(f"Unsupported model: {model_name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_mixup: bool,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    start = time.perf_counter()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels)
        amp_ctx = torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda") if device.type == "cuda" else nullcontext()
        with amp_ctx:
            logits = model(images)
            if use_mixup:
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, labels)
        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        preds_all.append(logits.argmax(dim=1).detach().cpu().numpy())
        targets_all.append(labels.detach().cpu().numpy())
    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    return {
        "loss": float(np.mean(losses)),
        "acc": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "time_sec": float(time.perf_counter() - start),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    collect_features: bool = False,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    features_all: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if collect_features and hasattr(model, "forward_features"):
                features = model.forward_features(images)
                logits = model.head(features) if hasattr(model, "head") else model(images)
            else:
                features = None
                logits = model(images)
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            probs_all.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            preds_all.append(logits.argmax(dim=1).detach().cpu().numpy())
            targets_all.append(labels.detach().cpu().numpy())
            if features is not None:
                features_all.append(features.detach().cpu().numpy())
    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    return {
        "loss": float(np.mean(losses)),
        "acc": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "preds": preds,
        "targets": targets,
        "probs": np.concatenate(probs_all),
        "features": np.concatenate(features_all) if features_all else None,
        "time_sec": float(time.perf_counter() - start),
    }


def _unnormalize_cifar(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=image.dtype, device=image.device).view(3, 1, 1)
    image = (image * std + mean).clamp(0, 1)
    return image.detach().cpu().permute(1, 2, 0).numpy()


def save_attention_maps(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    *,
    max_per_class: int = 1,
) -> list[dict[str, Any]]:
    if not isinstance(model, CIFARViT):
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    records: list[dict[str, Any]] = []
    counts = {idx: 0 for idx in range(len(CIFAR10_CLASSES))}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)
            attn = model.blocks[-1].attn.attn_map
            if attn is None:
                continue
            attn = attn.mean(dim=1)
            for batch_idx in range(images.size(0)):
                true_label = int(labels[batch_idx].item())
                if counts[true_label] >= max_per_class:
                    continue
                cls_attn = attn[batch_idx, 0, 1:]
                grid_size = int(np.sqrt(cls_attn.numel()))
                if grid_size * grid_size != cls_attn.numel():
                    continue
                heat = cls_attn.reshape(1, 1, grid_size, grid_size)
                heat = torch.nn.functional.interpolate(heat, size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
                heat_np = heat.detach().cpu().numpy()
                heat_np = (heat_np - heat_np.min()) / max(float(heat_np.max() - heat_np.min()), 1e-8)
                image_np = _unnormalize_cifar(images[batch_idx])

                pred_label = int(preds[batch_idx].item())
                file_name = (
                    f"class_{true_label}_{CIFAR10_CLASSES[true_label]}_"
                    f"pred_{pred_label}_{CIFAR10_CLASSES[pred_label]}_sample_{len(records):03d}.png"
                )
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(image_np)
                axes[0].set_title("Input")
                axes[1].imshow(heat_np, cmap="inferno")
                axes[1].set_title("CLS attention")
                axes[2].imshow(image_np)
                axes[2].imshow(heat_np, cmap="jet", alpha=0.45)
                axes[2].set_title(f"true={CIFAR10_CLASSES[true_label]}\npred={CIFAR10_CLASSES[pred_label]}")
                for ax in axes:
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(output_dir / file_name, dpi=220)
                plt.close(fig)

                records.append(
                    {
                        "file": file_name,
                        "true_label": true_label,
                        "true_name": CIFAR10_CLASSES[true_label],
                        "pred_label": pred_label,
                        "pred_name": CIFAR10_CLASSES[pred_label],
                        "pred_confidence": float(probs[batch_idx, pred_label].item()),
                    }
                )
                counts[true_label] += 1
                if all(value >= max_per_class for value in counts.values()):
                    (output_dir / "attention_map_index.json").write_text(
                        json.dumps(records, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    return records

    (output_dir / "attention_map_index.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_experiment(args: argparse.Namespace, model_name: str, fraction: float) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device)
    fraction_tag = f"{int(round(fraction * 100)):03d}"
    run_dir = Path(args.output_dir) / f"{model_name}_fraction_{fraction_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "best_model.pth"
    if args.skip_existing and summary_path.exists() and checkpoint_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    train_loader, val_loader, test_loader, train_samples = get_dataloaders(
        Path(args.data_root),
        args.batch_size,
        fraction,
        args.seed,
        args.num_workers,
    )
    model = build_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "train_epoch_time_sec": [],
        "val_epoch_time_sec": [],
        "epoch_time_sec": [],
        "lr": [],
    }
    best_val_f1 = -1.0
    best_epoch = 0
    patience = 0
    start_total = time.perf_counter()
    use_mixup = model_name == "vit" and args.mixup_alpha > 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_mixup=use_mixup,
            use_amp=args.use_amp,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["train_epoch_time_sec"].append(train_metrics["time_sec"])
        history["val_epoch_time_sec"].append(val_metrics["time_sec"])
        history["epoch_time_sec"].append(float(epoch_time))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "fraction": fraction,
                    "model_state_dict": model.state_dict(),
                    "best_val_macro_f1": best_val_f1,
                },
                checkpoint_path,
            )
        else:
            patience += 1

        print(
            f"[{model_name} {fraction:.0%}] epoch={epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} time={epoch_time:.2f}s"
        )
        if patience >= args.early_stopping_patience:
            print(f"[{model_name} {fraction:.0%}] Early stopping at epoch {epoch}")
            break

    total_training_time_sec = float(time.perf_counter() - start_total)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate(model, val_loader, criterion, device, collect_features=args.tsne)
    test_metrics = evaluate(model, test_loader, criterion, device, collect_features=args.tsne)

    save_json(run_dir / "history.json", history | {"total_training_time_sec": total_training_time_sec})
    val_pack = save_classification_artifacts(
        run_dir,
        "val",
        val_metrics["targets"],
        val_metrics["preds"],
        CIFAR10_CLASSES,
        y_score=val_metrics["probs"],
        features=val_metrics["features"],
        loss=val_metrics["loss"],
        title_prefix=f"CIFAR-10 {model_name.upper()} {fraction:.0%}",
        tsne_max_samples=args.tsne_max_samples,
        random_state=args.seed,
    )
    test_pack = save_classification_artifacts(
        run_dir,
        "test",
        test_metrics["targets"],
        test_metrics["preds"],
        CIFAR10_CLASSES,
        y_score=test_metrics["probs"],
        features=test_metrics["features"],
        loss=test_metrics["loss"],
        title_prefix=f"CIFAR-10 {model_name.upper()} {fraction:.0%}",
        tsne_max_samples=args.tsne_max_samples,
        random_state=args.seed,
    )
    attention_records = []
    if model_name == "vit" and args.attention_maps:
        attention_records = save_attention_maps(
            model,
            test_loader,
            device,
            run_dir / "attention_maps",
            max_per_class=args.attention_per_class,
        )
    summary = {
        "dataset": "CIFAR-10",
        "model_name": model_name,
        "fraction": float(fraction),
        "fraction_percent": int(round(fraction * 100)),
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "train_samples": int(train_samples),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "avg_epoch_time_sec": float(np.mean(history["epoch_time_sec"])),
        "total_training_time_sec": total_training_time_sec,
        "val_loss": float(val_metrics["loss"]),
        "val_accuracy": float(val_pack["accuracy"]),
        "val_macro_f1": float(val_pack["macro_f1"]),
        "val_weighted_f1": float(val_pack["weighted_f1"]),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_pack["accuracy"]),
        "test_macro_f1": float(test_pack["macro_f1"]),
        "test_weighted_f1": float(test_pack["weighted_f1"]),
        "test_balanced_accuracy": float(test_pack["balanced_accuracy"]),
        "test_top3_accuracy": float(test_pack.get("top3_accuracy", np.nan)),
        "attention_maps": len(attention_records),
    }
    save_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CIFAR-10 ViT/CNN training and evaluation.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "01_cifar_vit_cnn" / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "01_cifar_vit_cnn" / "outputs" / "unified_runs"))
    parser.add_argument("--models", nargs="+", choices=["vit", "cnn"], default=["vit", "cnn"])
    parser.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.5, 0.2, 0.1])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--tsne", action="store_true", help="Save t-SNE feature plots for val/test splits.")
    parser.add_argument("--tsne-max-samples", type=int, default=1000)
    parser.add_argument("--attention-maps", action="store_true", help="Save ViT attention maps for CIFAR test samples.")
    parser.add_argument("--attention-per-class", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name in args.models:
        for fraction in args.fractions:
            results.append(run_experiment(args, model_name, float(fraction)))
    df = pd.DataFrame(results).sort_values(["model_name", "fraction"], ascending=[True, False])
    df.to_csv(output_dir / "cifar_assignment_results.csv", index=False, encoding="utf-8-sig")
    save_json(output_dir / "cifar_assignment_results.json", results)
    plot_speed_accuracy_tradeoff(results, output_dir / "cifar_speed_accuracy_tradeoff.png", title="CIFAR-10 ViT vs CNN")
    for model_name in args.models:
        run_dirs = sorted(output_dir.glob(f"{model_name}_fraction_*"))
        plot_grouped_history_curves(run_dirs, output_dir, model_name=model_name, dataset_name="CIFAR-10")
    print(f"Saved unified CIFAR results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
