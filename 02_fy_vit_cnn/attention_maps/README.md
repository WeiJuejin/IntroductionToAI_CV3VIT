# FY Attention Map 输出目录

本目录用于保存 FY-4B ViT attention map。本地生成的 attention map 不随 GitHub 发布。

清理前的主要输出：

```text
attention_maps/
├── attention_map_index.json
├── class_0_clear_sample_*.png
├── class_1_water_sample_*.png
├── class_2_supercooled_sample_*.png
├── class_3_mixed_sample_*.png
└── class_4_ice_sample_*.png
```

重新运行 FY attention map 生成命令后，会重新生成这些文件。
