from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cifar_vit_cnn_unified import CIFAR10_CLASSES, CIFARViT, evaluate  # noqa: E402
from src.evaluation import save_classification_artifacts  # noqa: E402


CHECKPOINT_FRACTIONS = {
    "vit3_parameters(1).pth": 1.0,
    "vit3_params_50percent.pth": 0.5,
    "vit3_params_20percent.pth": 0.2,
    "vit3_params_10percent.pth": 0.1,
}


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_test_loader(data_root: Path, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_set = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=transform)
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert legacy CIFAR ViT checkpoints to unified evaluation outputs.")
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "01_cifar_vit_cnn" / "checkpoints" / "vit_legacy"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "01_cifar_vit_cnn" / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "01_cifar_vit_cnn" / "outputs" / "legacy_vit_unified"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--tsne-max-samples", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    test_loader = build_test_loader(Path(args.data_root), args.batch_size, args.num_workers)
    criterion = nn.CrossEntropyLoss()
    rows = []

    for checkpoint_name, fraction in CHECKPOINT_FRACTIONS.items():
        checkpoint_path = checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            print(f"Missing checkpoint, skipped: {checkpoint_path}")
            continue
        run_dir = output_dir / f"vit_fraction_{int(round(fraction * 100)):03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model = CIFARViT(num_classes=10).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        metrics = evaluate(model, test_loader, criterion, device, collect_features=args.tsne)
        metric_pack = save_classification_artifacts(
            run_dir,
            "test",
            metrics["targets"],
            metrics["preds"],
            CIFAR10_CLASSES,
            y_score=metrics["probs"],
            features=metrics["features"],
            loss=metrics["loss"],
            title_prefix=f"CIFAR-10 Legacy ViT {fraction:.0%}",
            tsne_max_samples=args.tsne_max_samples,
        )
        summary = {
            "dataset": "CIFAR-10",
            "model_name": "vit",
            "source": "legacy_checkpoint",
            "fraction": float(fraction),
            "fraction_percent": int(round(fraction * 100)),
            "checkpoint_path": str(checkpoint_path),
            "run_dir": str(run_dir),
            "test_loss": float(metrics["loss"]),
            "test_accuracy": float(metric_pack["accuracy"]),
            "test_macro_f1": float(metric_pack["macro_f1"]),
            "test_weighted_f1": float(metric_pack["weighted_f1"]),
            "test_balanced_accuracy": float(metric_pack["balanced_accuracy"]),
            "test_top3_accuracy": float(metric_pack.get("top3_accuracy", np.nan)),
        }
        save_json(run_dir / "summary.json", summary)
        rows.append(summary)
        print(f"Saved unified legacy evaluation: {run_dir}")

    if rows:
        pd.DataFrame(rows).sort_values("fraction", ascending=False).to_csv(
            output_dir / "legacy_vit_results_unified.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_json(output_dir / "legacy_vit_results_unified.json", rows)


if __name__ == "__main__":
    main()
