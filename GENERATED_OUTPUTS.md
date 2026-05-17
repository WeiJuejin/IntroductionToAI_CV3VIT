# 根目录运行产物说明

以下文件和目录是 `main.py` 在项目根目录生成的本地运行产物，不随 GitHub 发布：

```text
FY_CIFAR_ViT_CNN_Project/
├── predictions.csv
├── learning_curve.csv
├── speed_accuracy_tradeoff.png
├── SUBMISSION_CHECKLIST.md
├── attention_maps/
└── curve_plots/
```

清理前这些文件主要用于汇总 CIFAR-10 与 FY 两部分实验结果：

- `predictions.csv`：汇总预测结果。
- `learning_curve.csv`：汇总训练曲线数据。
- `speed_accuracy_tradeoff.png`：速度和精度权衡图。
- `attention_maps/`：汇总 attention map。
- `curve_plots/`：汇总训练曲线图。

整理后可公开展示的参考结果保存在 `reference_results/`，该目录可以随 GitHub 发布。
