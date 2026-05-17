# FY-4B ViT/CNN 云相分类实验

本目录用于 FY-4B 卫星数据上的云相分类实验，主要比较 ViT 与 CNN baseline 在不同训练数据比例下的表现，并输出统一的评价结果和可视化图表。

## 实验内容

- 使用 15 通道 FY-4B 卫星观测数据。
- 构建基于 patch 的云相分类数据集。
- 训练 ViT 分类模型和 CNN baseline。
- 对比 `100% / 20% / 10%` 训练数据比例下的性能。
- 统计训练时间，生成 speed-accuracy tradeoff 图。
- 输出混淆矩阵、预测结果、评价指标、多比例训练曲线和 ViT attention map。

## 目录结构

```text
02_fy_vit_cnn/
├── data_raw/              # FY-4B 原始 HDF/NC 数据，不随仓库发布
├── data_processed/        # 预处理后的 NPY/CSV/JSON 数据，不随仓库发布
├── notebooks/             # 实验 notebook
├── scripts/               # 数据预处理、训练和评估脚本
├── runs/                  # 训练结果、checkpoint 和评估输出
├── attention_maps/        # ViT 注意力图输出
├── assignment_results.*   # 实验汇总结果
├── speed_accuracy_tradeoff.png
└── README.md
```

## 主要脚本

- `scripts/build_fy4b_vit_scene_split_month_stratified.py`  
  从 FY-4B 原始 HDF/NC 文件构建 patch 数据集，包含场景配对、scene 级划分、月份分层、patch 筛选和通道归一化统计。

- `scripts/fy4b_assignment_pipeline.py`  
  训练和评估主脚本，负责 ViT/CNN 训练、验证集划分、数据量消融实验、结果汇总、图表输出和 attention map 生成。

## 数据准备

FY 原始数据不随仓库发布。复现实验时，需要将授权的 FY-4B 原始数据放入：

```text
02_fy_vit_cnn/data_raw/
```

然后从项目根目录运行预处理脚本：

```powershell
python .\02_fy_vit_cnn\scripts\build_fy4b_vit_scene_split_month_stratified.py
```

预处理结果会生成到：

```text
02_fy_vit_cnn/data_processed/fy4b_vit_month_stratified_scene_split/
```

## 训练与评估

推荐从项目根目录运行完整 FY 实验：

```powershell
python main.py --mode fy-eval
```

也可以直接运行 FY 子脚本：

```powershell
python .\02_fy_vit_cnn\scripts\fy4b_assignment_pipeline.py full --models vit cnn --fractions 1.0 0.2 0.1
```

如果已有 checkpoint，只想补齐评估输出：

```powershell
python .\02_fy_vit_cnn\scripts\fy4b_assignment_pipeline.py full --skip-existing
```

快速检查流程是否能跑通：

```powershell
python .\02_fy_vit_cnn\scripts\fy4b_assignment_pipeline.py full --epochs 2 --batch-size 64 --no-tsne
```

## 输出结果

完整运行后，主要结果保存在：

```text
02_fy_vit_cnn/runs/assignment_suite/
├── vit_fraction_100/
├── vit_fraction_020/
├── vit_fraction_010/
├── cnn_fraction_100/
├── cnn_fraction_020/
└── cnn_fraction_010/
```

每个实验目录通常包含：

- `best_model.pth`
- `history.json`
- `summary.json`
- `val_metrics.json` / `test_metrics.json`
- `val_predictions.csv` / `test_predictions.csv`
- 混淆矩阵图和训练曲线图

汇总结果和主要可视化包括：

- `assignment_results.csv`
- `assignment_results.json`
- `speed_accuracy_tradeoff.png`
- `runs/assignment_suite/*_loss_curves.png`
- `runs/assignment_suite/*_accuracy_curves.png`
- `runs/assignment_suite/*_macro_f1_curves.png`
- `attention_maps/`

## Attention Map

如果已有 ViT checkpoint，可以单独生成 attention map：

```powershell
python .\02_fy_vit_cnn\scripts\fy4b_assignment_pipeline.py attention --checkpoint .\02_fy_vit_cnn\runs\assignment_suite\vit_fraction_100\best_model.pth
```

## 实验说明

- 数据集先按 scene 划分，再切 patch，以减少相邻 patch 泄漏到不同集合的风险。
- 训练集和测试集按月份分层，尽量覆盖不同季节。
- 退化实验只改变训练集采样比例，验证集和测试集保持固定。
- CNN baseline 使用适配 15 通道输入的 ResNet-18 风格模型。
- ViT attention map 使用最后一层 CLS attention 生成。
- 默认数据和训练输出目录通常不提交到 GitHub。
