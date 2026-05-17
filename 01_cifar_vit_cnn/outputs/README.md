# CIFAR-10 输出目录

本目录用于保存 CIFAR-10 的本地训练和评估输出。实际运行结果不随 GitHub 发布。

清理前的主要输出结构：

```text
outputs/
└── unified_runs/
    ├── cifar_assignment_results.csv
    ├── cifar_assignment_results.json
    ├── cifar_speed_accuracy_tradeoff.png
    ├── cnn_loss_curves.png
    ├── cnn_accuracy_curves.png
    ├── cnn_macro_f1_curves.png
    ├── vit_loss_curves.png
    ├── vit_accuracy_curves.png
    ├── vit_macro_f1_curves.png
    ├── cnn_fraction_100/
    ├── cnn_fraction_020/
    ├── cnn_fraction_010/
    ├── vit_fraction_100/
    ├── vit_fraction_020/
    └── vit_fraction_010/
```

每个 `*_fraction_*` 子目录通常包含：

- `best_model.pth`
- `history.json`
- `summary.json`
- `val_metrics.json` / `test_metrics.json`
- `val_predictions.csv` / `test_predictions.csv`
- 混淆矩阵、分类报告和 attention map 等图表文件。

重新运行 `python main.py` 或 CIFAR 子脚本后，会重新生成这些输出。
