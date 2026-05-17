# 汇总曲线图输出目录

本目录用于保存 `main.py` 汇总复制出的 CIFAR-10 与 FY 多比例训练曲线。本地运行产物不随 GitHub 发布。

清理前的主要输出：

```text
curve_plots/
├── cifar_cnn_accuracy_curves.png
├── cifar_cnn_loss_curves.png
├── cifar_cnn_macro_f1_curves.png
├── cifar_vit_accuracy_curves.png
├── cifar_vit_loss_curves.png
├── cifar_vit_macro_f1_curves.png
├── fy_cnn_accuracy_curves.png
├── fy_cnn_loss_curves.png
├── fy_cnn_macro_f1_curves.png
├── fy_vit_accuracy_curves.png
├── fy_vit_loss_curves.png
└── fy_vit_macro_f1_curves.png
```

这些文件会在重新运行 `python main.py` 后重新生成。
