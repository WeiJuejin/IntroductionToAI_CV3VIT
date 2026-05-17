from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation import plot_grouped_history_curves

PROJECT_ROOT = Path(__file__).resolve().parent
FY_ROOT = PROJECT_ROOT / "02_fy_vit_cnn"
FY_PROCESSED_REQUIRED = FY_ROOT / "data_processed" / "fy4b_vit_month_stratified_scene_split" / "train_x_fy4b.npy"
FY_RAW_DIR = FY_ROOT / "data_raw"


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fy_data_available() -> bool:
    return FY_PROCESSED_REQUIRED.exists()


def fy_results_available() -> bool:
    runs_dir = FY_ROOT / "runs" / "assignment_suite"
    return runs_dir.exists() and any(runs_dir.glob("*/test_predictions.csv"))


def cifar_results_available() -> bool:
    outputs_dir = PROJECT_ROOT / "01_cifar_vit_cnn" / "outputs"
    return outputs_dir.exists() and any(outputs_dir.glob("**/test_predictions.csv"))


def print_data_status() -> None:
    if fy_data_available():
        print("FY processed data found: FY experiments are available.")
    else:
        print("FY processed data not found; FY experiments will be skipped.")
        print("This is expected for the GitHub version because FY raw/processed data is not redistributed.")
        print(f"Expected processed file: {FY_PROCESSED_REQUIRED.relative_to(PROJECT_ROOT)}")
        print(f"Place authorized FY raw files under: {FY_RAW_DIR.relative_to(PROJECT_ROOT)}")

    if cifar_results_available():
        print("Existing CIFAR outputs found.")
    else:
        print("No CIFAR outputs found. CIFAR-10 is public and can be downloaded by the training script.")


def collect_prediction_files(*, include_fy: bool | None = None) -> list[Path]:
    if include_fy is None:
        include_fy = fy_data_available() or fy_results_available()
    patterns = [
        "01_cifar_vit_cnn/outputs/unified_runs/*/test_predictions.csv",
        "01_cifar_vit_cnn/outputs/legacy_vit_unified/*/test_predictions.csv",
    ]
    if include_fy:
        patterns.append("02_fy_vit_cnn/runs/assignment_suite/*/test_predictions.csv")

    files: list[Path] = []
    for pattern in patterns:
        files.extend(PROJECT_ROOT.glob(pattern))
    return sorted(files)


def collect_history_files(*, include_fy: bool | None = None) -> list[Path]:
    if include_fy is None:
        include_fy = fy_data_available() or fy_results_available()
    patterns = ["01_cifar_vit_cnn/outputs/unified_runs/*/history.json"]
    if include_fy:
        patterns.append("02_fy_vit_cnn/runs/assignment_suite/*/history.json")

    files: list[Path] = []
    for pattern in patterns:
        files.extend(PROJECT_ROOT.glob(pattern))
    return sorted(files)


def parse_run_metadata(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    dataset = summary.get("dataset")
    if dataset is None:
        dataset = "FY-4B" if "02_fy_vit_cnn" in str(run_dir) else "CIFAR-10"
    return {
        "dataset": dataset,
        "model_name": summary.get("model_name", run_dir.name.split("_fraction_")[0]),
        "fraction": summary.get("fraction"),
        "fraction_percent": summary.get("fraction_percent"),
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }


def build_predictions_csv(output_path: Path, *, include_fy: bool | None = None) -> int:
    frames: list[pd.DataFrame] = []
    for path in collect_prediction_files(include_fy=include_fy):
        run_dir = path.parent
        metadata = parse_run_metadata(run_dir)
        df = pd.read_csv(path)
        if "sample_index" in df.columns:
            df = df.rename(columns={"sample_index": "id"})
        if "pred_label" in df.columns:
            df = df.rename(columns={"pred_label": "y_pred"})
        if "true_label" in df.columns:
            df = df.rename(columns={"true_label": "y_true"})
        prob_cols = [col for col in df.columns if col.startswith("prob_")]
        keep_cols = [col for col in ["id", "y_true", "y_pred", "true_name", "pred_name", "correct"] if col in df.columns]
        df = df[keep_cols + prob_cols].copy()
        for key, value in metadata.items():
            df.insert(0, key, value)
        frames.append(df)

    if not frames:
        pd.DataFrame(columns=["dataset", "model_name", "fraction_percent", "run_dir", "id", "y_pred"]).to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
        return 0

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(out)


def build_learning_curve_csv(output_path: Path, *, include_fy: bool | None = None) -> int:
    rows: list[dict[str, Any]] = []
    for path in collect_history_files(include_fy=include_fy):
        run_dir = path.parent
        metadata = parse_run_metadata(run_dir)
        history = read_json(path)
        epoch_count = max((len(v) for v in history.values() if isinstance(v, list)), default=0)
        for idx in range(epoch_count):
            row = {"epoch": idx + 1, **metadata}
            for key, value in history.items():
                if isinstance(value, list) and idx < len(value):
                    row[key] = value[idx]
            rows.append(row)

    if not rows:
        pd.DataFrame(columns=["dataset", "model_name", "fraction_percent", "epoch", "train_loss"]).to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
        return 0

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(rows)


def build_grouped_curve_plots(*, include_fy: bool | None = None) -> list[Path]:
    if include_fy is None:
        include_fy = fy_data_available() or fy_results_available()

    root_curve_dir = PROJECT_ROOT / "curve_plots"
    root_curve_dir.mkdir(exist_ok=True)
    generated: list[Path] = []
    cifar_output_dir = PROJECT_ROOT / "01_cifar_vit_cnn" / "outputs" / "unified_runs"
    if cifar_output_dir.exists():
        for model_name in ["vit", "cnn"]:
            run_dirs = sorted(cifar_output_dir.glob(f"{model_name}_fraction_*"))
            plots = plot_grouped_history_curves(run_dirs, cifar_output_dir, model_name=model_name, dataset_name="CIFAR-10")
            for path in plots:
                shutil.copy2(path, root_curve_dir / f"cifar_{path.name}")
            generated.extend(plots)

    if include_fy:
        fy_runs_dir = FY_ROOT / "runs" / "assignment_suite"
        if fy_runs_dir.exists():
            for model_name in ["vit", "cnn"]:
                run_dirs = sorted(fy_runs_dir.glob(f"{model_name}_fraction_*"))
                plots = plot_grouped_history_curves(run_dirs, fy_runs_dir, model_name=model_name, dataset_name="FY-4B")
                for path in plots:
                    shutil.copy2(path, root_curve_dir / f"fy_{path.name}")
                generated.extend(plots)
    return generated


def copy_primary_artifacts(*, include_fy: bool | None = None) -> None:
    if include_fy is None:
        include_fy = fy_data_available() or fy_results_available()

    tradeoff_candidates: list[Path] = []
    if include_fy:
        tradeoff_candidates.append(FY_ROOT / "speed_accuracy_tradeoff.png")
    tradeoff_candidates.append(PROJECT_ROOT / "01_cifar_vit_cnn" / "outputs" / "unified_runs" / "cifar_speed_accuracy_tradeoff.png")
    for path in tradeoff_candidates:
        if path.exists():
            shutil.copy2(path, PROJECT_ROOT / "speed_accuracy_tradeoff.png")
            break

    root_attention = PROJECT_ROOT / "attention_maps"
    root_attention.mkdir(exist_ok=True)
    attention_sources: list[Path] = []
    if include_fy:
        attention_sources.append(FY_ROOT / "attention_maps")
    attention_sources.extend((PROJECT_ROOT / "01_cifar_vit_cnn" / "outputs" / "unified_runs").glob("*/attention_maps"))
    for source in attention_sources:
        if not source.exists():
            continue
        prefix = "fy" if "02_fy_vit_cnn" in str(source) else source.parent.name
        for path in source.glob("*"):
            if path.is_file():
                shutil.copy2(path, root_attention / f"{prefix}_{path.name}")


def write_submission_checklist(pred_rows: int, curve_rows: int, *, include_fy: bool) -> None:
    required = {
        "requirements.txt": (PROJECT_ROOT / "requirements.txt").exists(),
        "main.py": (PROJECT_ROOT / "main.py").exists(),
        "run.bat": (PROJECT_ROOT / "run.bat").exists(),
        "predictions.csv": (PROJECT_ROOT / "predictions.csv").exists(),
        "learning_curve.csv": (PROJECT_ROOT / "learning_curve.csv").exists(),
        "curve_plots/": (PROJECT_ROOT / "curve_plots").exists(),
        "speed_accuracy_tradeoff.png": (PROJECT_ROOT / "speed_accuracy_tradeoff.png").exists(),
        "attention_maps/": (PROJECT_ROOT / "attention_maps").exists(),
        "report.pdf": (PROJECT_ROOT / "report.pdf").exists(),
    }
    lines = [
        "# 提交检查清单",
        "",
        "| 项目 | 状态 | 说明 |",
        "|---|---:|---|",
    ]
    for item, ok in required.items():
        note = ""
        if item == "report.pdf" and not ok:
            note = "报告后续单独撰写。"
        elif item == "predictions.csv":
            note = f"{pred_rows} 行预测结果。"
        elif item == "learning_curve.csv":
            note = f"{curve_rows} 行训练曲线记录。"
        elif item == "speed_accuracy_tradeoff.png" and not ok:
            note = "CIFAR 或 FY 实验完成后生成。"
        lines.append(f"| `{item}` | {'完成' if ok else '待补'} | {note} |")

    lines.extend(
        [
            "",
            "## 数据状态",
            "",
            f"- FY 预处理数据可用：{'是' if fy_data_available() else '否'}",
            f"- 根目录汇总结果是否包含 FY：{'是' if include_fy else '否'}",
            "- FY 原始数据和预处理数据不随 GitHub 版本发布。",
        ]
    )
    (PROJECT_ROOT / "SUBMISSION_CHECKLIST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cifar(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "01_cifar_vit_cnn/scripts/cifar_vit_cnn_unified.py",
        "--models",
        *args.cifar_models,
        "--fractions",
        *[str(v) for v in args.cifar_fractions],
        "--epochs",
        str(args.cifar_epochs),
        "--attention-maps",
    ]
    if args.with_tsne:
        command.append("--tsne")
    run_command(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project entry point and final artifact collector.")
    parser.add_argument(
        "--mode",
        choices=["auto", "collect", "fy-eval", "cifar-train"],
        default="auto",
        help=(
            "auto runs the full public CIFAR experiment if no CIFAR outputs exist; "
            "collect only gathers outputs; fy-eval requires FY processed data; cifar-train runs CIFAR."
        ),
    )
    parser.add_argument("--cifar-epochs", type=int, default=80)
    parser.add_argument("--cifar-models", nargs="+", choices=["vit", "cnn"], default=["vit", "cnn"])
    parser.add_argument("--cifar-fractions", nargs="+", type=float, default=[1.0, 0.2, 0.1])
    parser.add_argument("--with-tsne", action="store_true")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a lightweight demo: ViT+CNN, fractions 0.2 and 0.1, 10 epochs, with attention maps.",
    )
    parser.add_argument("--include-fy-results", action="store_true", help="Include existing FY result files in root CSVs even if FY data is absent.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.demo:
        args.mode = "cifar-train"
        args.cifar_models = ["vit", "cnn"]
        args.cifar_fractions = [0.2, 0.1]
        args.cifar_epochs = 10
    print_data_status()

    include_fy = fy_data_available() or (args.include_fy_results and fy_results_available())

    if args.mode == "auto" and not collect_prediction_files(include_fy=False):
        print("No CIFAR prediction outputs found. Running the full CIFAR public-data experiment.")
        print("Default: models=vit cnn, fractions=1.0 0.2 0.1, epochs=80.")
        run_cifar(args)
    elif args.mode == "fy-eval":
        if not fy_data_available():
            print("Cannot run FY evaluation/training because FY processed data is missing.")
            print("After obtaining authorized FY data, run:")
            print("  python 02_fy_vit_cnn/scripts/build_fy4b_vit_scene_split_month_stratified.py")
            print("  python main.py --mode fy-eval")
            include_fy = False
        else:
            run_command(
                [
                    sys.executable,
                    "02_fy_vit_cnn/scripts/fy4b_assignment_pipeline.py",
                    "full",
                    "--skip-existing",
                    "--no-tsne",
                ]
            )
            include_fy = True
    elif args.mode == "cifar-train":
        run_cifar(args)

    copy_primary_artifacts(include_fy=include_fy)
    pred_rows = build_predictions_csv(PROJECT_ROOT / "predictions.csv", include_fy=include_fy)
    curve_rows = build_learning_curve_csv(PROJECT_ROOT / "learning_curve.csv", include_fy=include_fy)
    curve_plots = build_grouped_curve_plots(include_fy=include_fy)
    write_submission_checklist(pred_rows, curve_rows, include_fy=include_fy)
    print(f"Collected {pred_rows} prediction rows into predictions.csv")
    print(f"Collected {curve_rows} curve rows into learning_curve.csv")
    print(f"Generated {len(curve_plots)} grouped curve plots")
    print("Wrote SUBMISSION_CHECKLIST.md")


if __name__ == "__main__":
    main()
