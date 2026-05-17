# FY 训练输出目录

本目录用于保存 FY-4B ViT/CNN 实验的本地训练结果、checkpoint 和评估输出。实际运行结果不随 GitHub 发布。

清理前的主要输出结构：

```text
runs/
└── assignment_suite/
    ├── vit_fraction_100/
    ├── vit_fraction_020/
    ├── vit_fraction_010/
    ├── cnn_fraction_100/
    ├── cnn_fraction_020/
    ├── cnn_fraction_010/
    ├── vit_loss_curves.png
    ├── vit_accuracy_curves.png
    ├── vit_macro_f1_curves.png
    ├── cnn_loss_curves.png
    ├── cnn_accuracy_curves.png
    └── cnn_macro_f1_curves.png
```

每个 `*_fraction_*` 子目录通常包含：

- `best_model.pth`
- `history.json`
- `summary.json`
- `val_metrics.json` / `test_metrics.json`
- `val_predictions.csv` / `test_predictions.csv`
- 混淆矩阵、分类报告和训练曲线等文件。

重新运行 FY 训练评估流程后，会重新生成这些输出。
