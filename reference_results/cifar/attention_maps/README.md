# 汇总 Attention Map 输出目录

本目录用于保存 `main.py` 汇总复制出的 CIFAR-10 与 FY attention map。本地运行产物不随 GitHub 发布。

清理前的主要输出：

```text
attention_maps/
├── fy_attention_map_index.json
├── fy_class_*_sample_*.png
├── vit_fraction_010_attention_map_index.json
├── vit_fraction_010_class_*_sample_*.png
├── vit_fraction_020_attention_map_index.json
├── vit_fraction_020_class_*_sample_*.png
├── vit_fraction_100_attention_map_index.json
└── vit_fraction_100_class_*_sample_*.png
```

这些文件会在重新运行 `python main.py` 或相关子脚本后重新生成。
