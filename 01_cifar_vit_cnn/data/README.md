# CIFAR-10 数据目录

本目录用于保存 CIFAR-10 自动下载和解压后的数据文件。实际数据不随 GitHub 发布。

清理前的主要内容：

```text
data/
├── cifar-10-python.tar.gz
└── cifar-10-batches-py/
    ├── batches.meta
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── readme.html
    └── test_batch
```

运行项目时，脚本会在需要时重新下载 CIFAR-10，并重新生成上述文件。
