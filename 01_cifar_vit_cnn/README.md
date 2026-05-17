# CIFAR-10 ViT/CNN 实验

本目录用于公开基础数据集 CIFAR-10 的 ViT 与 CNN 分类对比实验。

## 主要文件

- `scripts/cifar_vit_cnn_unified.py`：CIFAR-10 统一训练与评估入口。
- `scripts/evaluate_legacy_vit_checkpoints.py`：将已有 ViT checkpoint 转换为统一评估输出。
- `scripts/legacy_vit/`：原始 ViT 训练脚本。
- `scripts/legacy_cnn/`：原始 CNN 训练脚本。
- `data/`：CIFAR-10 自动下载目录，GitHub 只保留 README。
- `outputs/`：训练输出目录，GitHub 只保留 README。

## 推荐运行方式

从项目根目录运行：

```powershell
python main.py
```

该命令会在没有 CIFAR 输出时自动下载 CIFAR-10，并运行默认全量实验：

```text
models = vit cnn
fractions = 1.0 0.2 0.1
epochs = 80
```

## 直接运行 CIFAR 子脚本

```powershell
cd C:\Users\38851\Desktop\FY_CIFAR_ViT_CNN_Project
python .\01_cifar_vit_cnn\scripts\cifar_vit_cnn_unified.py --models vit cnn --fractions 1.0 0.2 0.1 --epochs 80 --attention-maps
```

输出默认保存到：

```text
01_cifar_vit_cnn/outputs/unified_runs/
```

## 轻量级流程检查

```powershell
python main.py --demo
```

该命令运行 ViT/CNN、`20% / 10%` 数据比例和 `10 epochs`，用于快速验证流程。
