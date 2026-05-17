from __future__ import annotations

import json
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - plotting fallback
    sns = None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _safe_names(class_names: Sequence[str], labels: Sequence[int]) -> list[str]:
    names = list(class_names)
    if len(names) >= len(labels):
        return [str(names[int(i)]) for i in labels]
    return [str(i) for i in labels]


def _top_k_accuracy(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if y_score.ndim != 2 or y_score.shape[1] < k:
        return float("nan")
    topk = np.argsort(y_score, axis=1)[:, -k:]
    return float(np.mean([target in row for target, row in zip(y_true, topk)]))


def _specificity(cm: np.ndarray) -> np.ndarray:
    total = cm.sum()
    out = []
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        out.append(float(tn / denom) if denom else 0.0)
    return np.asarray(out, dtype=np.float64)


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
    *,
    y_score: np.ndarray | None = None,
    loss: float | None = None,
) -> dict[str, Any]:
    """Compute a broad, consistent metric pack for multi-class classification."""
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sums, 1), dtype=np.float64)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=labels,
        zero_division=0,
    )
    specificity = _specificity(cm)
    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "specificity": float(specificity[i]),
            "support": int(support[i]),
        }
        for i in labels
    }

    metrics: dict[str, Any] = {
        "loss": None if loss is None else float(loss),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "macro_precision": float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "macro_recall": float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true_np, y_pred_np, average="micro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true_np, y_pred_np)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true_np, y_pred_np)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized_true": cm_norm.tolist(),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true_np,
            y_pred_np,
            labels=labels,
            target_names=_safe_names(class_names, labels),
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
    }

    if y_score is not None:
        y_score_np = np.asarray(y_score, dtype=np.float64)
        metrics["top1_accuracy"] = _top_k_accuracy(y_true_np, y_score_np, 1)
        metrics["top2_accuracy"] = _top_k_accuracy(y_true_np, y_score_np, 2)
        metrics["top3_accuracy"] = _top_k_accuracy(y_true_np, y_score_np, min(3, y_score_np.shape[1]))

    return metrics


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    *,
    title: str,
    normalized: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.9), max(5, len(class_names) * 0.75)))
    fmt = ".2f" if normalized else "d"
    if sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            ax=ax,
        )
    else:
        im = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), labels=class_names)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_report_csv(report: dict[str, Any], output_path: Path) -> None:
    rows = []
    for label, value in report.items():
        if isinstance(value, dict):
            rows.append({"label": label, **value})
        else:
            rows.append({"label": label, "value": value})
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_tsne_plot(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    *,
    max_samples: int = 1000,
    random_state: int = 42,
    title: str = "t-SNE Feature Visualization",
) -> pd.DataFrame | None:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or len(features) < 4:
        return None

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(labels))
    if len(indices) > max_samples:
        selected: list[np.ndarray] = []
        per_class = max(1, max_samples // max(1, len(class_names)))
        for label in sorted(np.unique(labels).tolist()):
            cls_idx = indices[labels == label]
            take = min(len(cls_idx), per_class)
            selected.append(rng.choice(cls_idx, size=take, replace=False))
        merged = np.concatenate(selected)
        if len(merged) < max_samples:
            remaining = np.setdiff1d(indices, merged, assume_unique=False)
            extra = rng.choice(remaining, size=min(len(remaining), max_samples - len(merged)), replace=False)
            merged = np.concatenate([merged, extra])
        indices = np.sort(merged[:max_samples])

    x = features[indices]
    y = labels[indices]
    x = StandardScaler().fit_transform(x)
    if x.shape[1] > 50 and len(x) > 50:
        x = PCA(n_components=50, random_state=random_state).fit_transform(x)

    perplexity = min(30, max(2, (len(x) - 1) // 3))
    if perplexity >= len(x):
        perplexity = max(1, len(x) - 1)
    coords = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=random_state,
    ).fit_transform(x)

    df = pd.DataFrame(
        {
            "sample_index": indices,
            "label": y,
            "label_name": [class_names[int(label)] for label in y],
            "tsne_x": coords[:, 0],
            "tsne_y": coords[:, 1],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for label in sorted(np.unique(y).tolist()):
        mask = y == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=16,
            alpha=0.78,
            color=cmap(int(label) % 10),
            label=class_names[int(label)],
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    ax.legend(markerscale=1.5, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    return df


def save_classification_artifacts(
    output_dir: Path,
    split_name: str,
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
    *,
    y_score: np.ndarray | None = None,
    features: np.ndarray | None = None,
    loss: float | None = None,
    title_prefix: str = "",
    tsne_max_samples: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    metrics = compute_classification_metrics(y_true_np, y_pred_np, class_names, y_score=y_score, loss=loss)

    stem = split_name.lower()
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    cm_norm = np.asarray(metrics["confusion_matrix_normalized_true"], dtype=np.float64)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / f"{stem}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(
        output_dir / f"{stem}_confusion_matrix_normalized.csv",
        encoding="utf-8-sig",
    )
    plot_title = f"{title_prefix} {split_name} Confusion Matrix".strip()
    save_confusion_matrix_plot(cm, class_names, output_dir / f"{stem}_confusion_matrix.png", title=plot_title)
    save_confusion_matrix_plot(
        cm_norm,
        class_names,
        output_dir / f"{stem}_confusion_matrix_normalized.png",
        title=f"{plot_title} (row-normalized)",
        normalized=True,
    )

    write_json(output_dir / f"{stem}_metrics.json", metrics)
    _save_report_csv(metrics["classification_report"], output_dir / f"{stem}_classification_report.csv")

    pred_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true_np)),
            "true_label": y_true_np,
            "true_name": [class_names[int(i)] for i in y_true_np],
            "pred_label": y_pred_np,
            "pred_name": [class_names[int(i)] for i in y_pred_np],
            "correct": y_true_np == y_pred_np,
        }
    )
    if y_score is not None:
        score_np = np.asarray(y_score)
        for idx, name in enumerate(class_names):
            if idx < score_np.shape[1]:
                pred_df[f"prob_{name}"] = score_np[:, idx]
    pred_df.to_csv(output_dir / f"{stem}_predictions.csv", index=False, encoding="utf-8-sig")

    if features is not None:
        save_tsne_plot(
            np.asarray(features),
            y_true_np,
            class_names,
            output_dir / f"{stem}_tsne.png",
            max_samples=tsne_max_samples,
            random_state=random_state,
            title=f"{title_prefix} {split_name} t-SNE".strip(),
        )
    return metrics


def plot_history(history: dict[str, Sequence[float]], output_path: Path, *, title: str = "Training History") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    specs = [
        ("Loss", [("train_loss", "train"), ("val_loss", "val"), ("test_loss", "test")]),
        ("Accuracy", [("train_acc", "train"), ("val_acc", "val"), ("test_acc", "test")]),
        ("Macro-F1", [("train_macro_f1", "train"), ("val_macro_f1", "val"), ("test_macro_f1", "test")]),
        ("Timing (seconds)", [("epoch_time_sec", "epoch"), ("train_epoch_time_sec", "train"), ("val_epoch_time_sec", "val")]),
    ]
    for ax, (name, curves) in zip(axes, specs):
        plotted = False
        for key, label in curves:
            values = history.get(key)
            if values:
                ax.plot(list(values), label=label)
                plotted = True
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _history_fraction_percent(run_dir: Path) -> int:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("fraction_percent") is not None:
                return int(round(float(summary["fraction_percent"])))
            if summary.get("fraction") is not None:
                return int(round(float(summary["fraction"]) * 100))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    marker = "_fraction_"
    if marker in run_dir.name:
        suffix = run_dir.name.rsplit(marker, 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    return 0


def _history_series(history: dict[str, Any], keys: Sequence[str]) -> tuple[str | None, list[float]]:
    for key in keys:
        values = history.get(key)
        if not isinstance(values, SequenceABC) or isinstance(values, (str, bytes)):
            continue
        parsed: list[float] = []
        for value in values:
            try:
                parsed.append(float(value))
            except (TypeError, ValueError):
                continue
        if parsed:
            return key, parsed
    return None, []


def plot_history_metric_by_fraction(
    run_dirs: Sequence[Path],
    output_path: Path,
    *,
    model_name: str,
    metric_label: str,
    train_keys: Sequence[str],
    eval_keys: Sequence[str],
    dataset_name: str = "",
) -> bool:
    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        train_key, train_values = _history_series(history, train_keys)
        eval_key, eval_values = _history_series(history, eval_keys)
        if not train_values and not eval_values:
            continue
        records.append(
            {
                "fraction_percent": _history_fraction_percent(run_dir),
                "train_key": train_key,
                "train_values": train_values,
                "eval_key": eval_key,
                "eval_values": eval_values,
            }
        )

    if not records:
        return False

    records.sort(key=lambda item: item["fraction_percent"], reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, record in enumerate(records):
        fraction_label = f"{record['fraction_percent']}%"
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None

        train_values = record["train_values"]
        if train_values:
            markevery = max(1, len(train_values) // 12)
            ax.plot(
                range(1, len(train_values) + 1),
                train_values,
                color=color,
                marker="o",
                markevery=markevery,
                linewidth=1.9,
                markersize=4,
                label=f"{fraction_label} Train {metric_label}",
            )

        eval_values = record["eval_values"]
        if eval_values:
            eval_key = str(record.get("eval_key") or "")
            eval_split = "Test" if eval_key.startswith("test_") else "Val" if eval_key.startswith("val_") else "Eval"
            markevery = max(1, len(eval_values) // 12)
            ax.plot(
                range(1, len(eval_values) + 1),
                eval_values,
                color=color,
                linestyle="--",
                marker="s",
                markevery=markevery,
                linewidth=1.9,
                markersize=4,
                label=f"{fraction_label} {eval_split} {metric_label}",
            )

    prefix = f"{dataset_name} " if dataset_name else ""
    ax.set_title(f"{prefix}{model_name.upper()} {metric_label} Curves Under Different Training Data Ratios")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    return True


def plot_grouped_history_curves(
    run_dirs: Sequence[Path],
    output_dir: Path,
    *,
    model_name: str,
    dataset_name: str = "",
) -> list[Path]:
    """Save one combined curve chart per metric for a model across data ratios."""
    output_dir = Path(output_dir)
    model_key = model_name.lower()
    specs = [
        ("loss", "Loss", ["train_loss"], ["test_loss", "val_loss"]),
        ("accuracy", "Accuracy", ["train_acc", "train_accuracy"], ["test_acc", "test_accuracy", "val_acc", "val_accuracy"]),
        ("macro_f1", "Macro-F1", ["train_macro_f1"], ["test_macro_f1", "val_macro_f1"]),
    ]
    generated: list[Path] = []
    for file_key, metric_label, train_keys, eval_keys in specs:
        output_path = output_dir / f"{model_key}_{file_key}_curves.png"
        ok = plot_history_metric_by_fraction(
            run_dirs,
            output_path,
            model_name=model_name,
            metric_label=metric_label,
            train_keys=train_keys,
            eval_keys=eval_keys,
            dataset_name=dataset_name,
        )
        if ok:
            generated.append(output_path)
    return generated


def plot_speed_accuracy_tradeoff(
    results: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Speed-Accuracy Tradeoff",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    markers = {"vit": "o", "cnn": "s"}
    colors = {"vit": "#1f77b4", "cnn": "#d62728"}
    for row in results:
        model = str(row.get("model_name", row.get("model", "model"))).lower()
        fraction = int(round(float(row.get("fraction", 1.0)) * 100))
        time_sec = float(row.get("total_training_time_sec", row.get("train_time_sec", 0.0)))
        acc = float(row.get("test_accuracy", row.get("test_acc", row.get("accuracy", 0.0))))
        macro_f1 = float(row.get("test_macro_f1", row.get("macro_f1", 0.0)))
        ax.scatter(
            time_sec,
            acc,
            s=130,
            marker=markers.get(model, "o"),
            color=colors.get(model, "#444444"),
            alpha=0.9,
        )
        ax.annotate(
            f"{model.upper()}-{fraction}%\nF1={macro_f1:.3f}",
            (time_sec, acc),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )
    ax.set_xlabel("Total training time (seconds)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
