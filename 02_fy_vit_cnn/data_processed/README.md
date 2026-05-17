# FY 预处理数据目录

本目录用于保存 FY-4B 原始数据预处理后的训练数据。实际预处理数据不随 GitHub 发布。

清理前的主要输出结构：

```text
data_processed/
├── scene_counts_stride64.csv
└── fy4b_vit_month_stratified_scene_split/
    ├── channel_stats_train.npz
    ├── dataset_summary.json
    ├── scene_split.json
    ├── train_x_fy4b.npy
    ├── train_y_fy4b.npy
    ├── test_x_fy4b.npy
    ├── test_y_fy4b.npy
    ├── train_patch_metadata.csv
    ├── test_patch_metadata.csv
    ├── train_scene_manifest.csv
    └── test_scene_manifest.csv
```

其中 `train_x_fy4b.npy` 和 `test_x_fy4b.npy` 是主要的大体积数组文件，本地清理前合计约 2.6 GB。

重新放入授权 FY 原始数据并运行预处理脚本后，会重新生成这些文件。
