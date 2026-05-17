# 面向风云卫星云相态分类的 ViT 与 CNN 模型性能比较研究

## 一、项目结构

```text
FY_CIFAR_ViT_CNN_Project/
├── main.py
├── run.bat
├── run.sh
├── requirements.txt
├── reference_results/       # 已整理的参考结果，随 GitHub 发布
├── 01_cifar_vit_cnn/
│   ├── data/                 # CIFAR-10 自动下载目录，GitHub 只保留 README
│   ├── outputs/              # CIFAR 训练输出目录，GitHub 只保留 README
│   └── scripts/
├── 02_fy_vit_cnn/
│   ├── data_raw/             # FY 原始数据目录，不随 GitHub 发布
│   ├── data_processed/       # FY 预处理数据目录，不随 GitHub 发布
│   ├── runs/                 # FY 训练输出和 checkpoint，不随 GitHub 发布
│   └── scripts/
└── src/
```

核心文件夹：

- `01_cifar_vit_cnn/`：使用公开 CIFAR-10 数据集完成 ViT 与 CNN 分类实验。
- `02_fy_vit_cnn/`：使用 FY-4B 卫星数据完成 ViT 与 CNN 云相态分类实验。
- `reference_results/`：保存已经跑好的最终参考结果，访问者可直接查看，不参与重新训练。
- `src/evaluation/`：统一评价指标、混淆矩阵、统一折线图和结果导出工具。


## 二、交付物路径

本项目对应交付物如下。为了避免把本地运行产物和参考结果混在一起，GitHub 中主要查看 `reference_results/` 下已经整理好的结果。

| 课程要求 | 本项目路径 | 说明 |
| --- | --- | --- |
| 代码，可一键运行 | `main.py` | 在项目根目录运行 `python main.py`。 |
| 依赖文件 | `requirements.txt` | 记录 Python 依赖库。 |
| 复现实验脚本 | `run.bat` / `run.sh` | Windows 使用 `run.bat`，Linux/macOS 使用 `run.sh`。 |
| 报告 | `report.pdf` | 最终报告请放在项目根目录；该文件不由代码自动生成。 |
| 分类预测结果 | `reference_results/combined/summary/predictions.csv` | 包含 `id`、`y_pred` 和各类别 `prob_*` 概率列。 |
| 训练曲线数据 | `reference_results/combined/summary/learning_curve.csv` | 包含 epoch、loss、accuracy、macro-F1、训练时间等信息。 |
| speed-accuracy tradeoff 图 | `reference_results/combined/summary/speed_accuracy_tradeoff.png` | 对应课程额外验收物 `speed_accuracy_tradeoff.png`。 |
| attention map | `reference_results/combined/attention_maps/` | 汇总 CIFAR-10 与 FY 的 ViT attention map。 |
| 多比例训练曲线图 | `reference_results/combined/curves/` | 汇总 CIFAR-10 与 FY 的 loss、accuracy、macro-F1 曲线。 |
| CIFAR-10 单独结果 | `reference_results/cifar/` | 包含 CIFAR-10 的 summary、runs、curves 和 attention maps。 |
| FY 单独结果 | `reference_results/fy/` | 包含 FY 云相态分类的 summary、runs、curves 和 attention maps。 |

注：重新运行 `python main.py` 后会在根目录生成：

```text
predictions.csv
learning_curve.csv
speed_accuracy_tradeoff.png
attention_maps/
curve_plots/
```

这些根目录文件属于本地运行产物，当前 GitHub 展示版使用 `reference_results/combined/` 中的整理结果。

## 三、参考结果与复现结果

本项目把“可查看的参考结果”和“访问者自己运行产生的复现结果”分开保存。

参考结果位于：

```text
reference_results/
├── cifar/       # CIFAR-10 的 ViT/CNN 参考结果
├── fy/          # FY 卫星云相态分类参考结果
└── combined/    # main.py 汇总后的参考结果副本
```

访问者可以直接打开 `reference_results/` 查看已有实验结果，包括：

- 指标汇总表；
- 训练曲线；
- 混淆矩阵和归一化混淆矩阵；
- 预测结果表；
- 速度-精度对比图；
- ViT attention map；
- t-SNE 图，如果运行时启用了该功能。

访问者自己复现时，新结果仍会生成在默认输出目录，例如 `01_cifar_vit_cnn/outputs/unified_runs/`、`02_fy_vit_cnn/runs/`、`curve_plots/`、`attention_maps/` 等。这些目录已写入 `.gitignore`，不会和 `reference_results/` 中的参考结果混在一起。

## 四、运行指南

在项目根目录运行：

```powershell
python main.py
```

Windows 下也可以直接运行：

```powershell
.\run.bat
```

注意：

- 如果没有 CIFAR 结果，会自动下载 CIFAR-10，并运行 CIFAR 的 ViT/CNN 全量实验。
- CIFAR 默认设置为 `VIT / CNN`、训练比例 `1.0 0.2 0.1`、`80 epochs`。
- 如果没有 FY 数据，会提示并跳过 FY，不会报错。
- 如果本地有 FY 数据和 FY 结果，会把可用结果汇总到根目录。

轻量级演示流程：

```powershell
python main.py --demo
```

该命令运行 CIFAR-10 的 ViT/CNN，训练比例为 `20% / 10%`，训练 `10 epochs`，并输出 attention map 和统一折线图，适合快速检查项目流程。

## 五、风云卫星数据说明
FY 是我国自主气象卫星，原始 L0/L1 级数据（如 FDI）含高精度辐射、定标、轨道参数，属敏感空间信息。故 FY 原始数据和预处理数据不随 GitHub 发布，若要复现 FY 部分，需要先把有授权的 FY 原始数据放入：

```text
02_fy_vit_cnn/data_raw/
```

然后运行：

```powershell
python .\02_fy_vit_cnn\scripts\build_fy4b_vit_scene_split_month_stratified.py
python main.py --mode fy-eval
```

如果项目文件夹内已经存在 FY 预处理数据，可直接运行：

```powershell
python main.py --mode fy-eval
```

## 六、统一输出

每个实验 run 目录会统一输出：

- `summary.json`
- `history.json`
- `val_metrics.json` / `test_metrics.json`
- `val_classification_report.csv` / `test_classification_report.csv`
- `val_confusion_matrix.png` / `test_confusion_matrix.png`
- `val_confusion_matrix_normalized.png` / `test_confusion_matrix_normalized.png`
- `val_predictions.csv` / `test_predictions.csv`

折线图：

- `cnn_loss_curves.png`
- `cnn_accuracy_curves.png`
- `cnn_macro_f1_curves.png`
- `vit_loss_curves.png`
- `vit_accuracy_curves.png`
- `vit_macro_f1_curves.png`

`main.py` 会在根目录汇总生成：

- `predictions.csv`
- `learning_curve.csv`
- `curve_plots/`
- `speed_accuracy_tradeoff.png`
- `attention_maps/`
- `SUBMISSION_CHECKLIST.md`

这些根目录汇总文件是运行产物，已通过 `.gitignore` 排除，避免影响复现者重新运行。

整理后的参考结果会复制到 `reference_results/`，该目录用于 GitHub 展示，可以提交。

## 七、评价指标

统一评价指标包括：

- Accuracy
- Macro-F1 / Weighted-F1 / Micro-F1
- Precision / Recall
- Balanced Accuracy
- Cohen Kappa
- Matthews Correlation Coefficient
- Top-k Accuracy
- 混淆矩阵和按真实类别归一化的混淆矩阵
- 可选 t-SNE 特征可视化

## GitHub 发布说明

可以提交到 GitHub：

- 代码、脚本、README 和文档；
- `reference_results/` 中整理好的参考结果；
- 数据目录占位说明文件。

以下内容不提交到 GitHub：

（文件夹具体说明以写入相应路径下的README.md）
- `01_cifar_vit_cnn/data/` 中下载的 CIFAR 文件
- `01_cifar_vit_cnn/outputs/unified_runs/`
- `01_cifar_vit_cnn/checkpoints/`
- `02_fy_vit_cnn/data_raw/`
- `02_fy_vit_cnn/data_processed/`
- `02_fy_vit_cnn/runs/`
- `*.pth`、`*.pt`、`*.ckpt` 等模型权重文件
- 根目录生成的 `predictions.csv`、`learning_curve.csv`、`curve_plots/`、`attention_maps/`

这些路径已经写入 `.gitignore`。因此 GitHub 上能看到我们的参考结果，复现者也能重新运行生成自己的结果，两者不会互相覆盖。

