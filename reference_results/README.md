# 参考结果说明

本目录用于存放本项目已经完成实验后整理出的参考结果。与默认运行输出目录分开：

- `reference_results/`：随 GitHub 发布，供访问者直接查看本文实验结果。
- 默认输出目录：访问者自己复现时重新生成，继续由 `.gitignore` 忽略，不会和参考结果混在一起。

## 目录结构

```text
reference_results/
├── cifar/
│   ├── summary/          # CIFAR-10 总体指标表和速度-精度图
│   ├── curves/           # CIFAR-10 多比例训练曲线
│   ├── runs/             # CIFAR-10 各模型/比例的指标、预测、混淆矩阵
│   └── attention_maps/   # CIFAR-10 ViT attention map
├── fy/
│   ├── summary/          # FY 总体指标表和速度-精度图
│   ├── curves/           # FY 多比例训练曲线
│   ├── runs/             # FY 各模型/比例的指标、预测、混淆矩阵
│   └── attention_maps/   # FY ViT attention map
└── combined/
    ├── summary/          # main.py 汇总后的根目录结果副本
    ├── curves/           # CIFAR 与 FY 汇总曲线
    └── attention_maps/   # CIFAR 与 FY 汇总 attention map
```

## 复现注意事项

访问者如需直接查看结果，请阅读本目录。

访问者如需自行复现实验，请在项目根目录运行：

```powershell
python main.py
```

复现产生的新结果会输出到默认目录，如：

- `01_cifar_vit_cnn/outputs/unified_runs/`
- `02_fy_vit_cnn/runs/`
- `predictions.csv`
- `learning_curve.csv`
- `curve_plots/`
- `attention_maps/`

这些默认输出目录和文件不会提交到 GitHub，因此不会污染本目录下的参考结果。

## 不包含的内容

本目录只保存可查看的实验结果，不保存：

- CIFAR-10 下载后的数据文件。
- FY 原始数据。
- FY 预处理后的数据。
- 模型 checkpoint，例如 `best_model.pth`。
- Python 缓存或临时文件。
