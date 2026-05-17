import torch
import torch.nn as nn
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. 优化后的 CNN 模型定义
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1: 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),#卷积层
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),#池化层，将图片压缩为32×32

            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 在深层特征提取加入轻量 Dropout，进一步防止过拟合
            nn.Dropout2d(0.1) 
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 全局平均池化
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. 带有学习率调度、时间统计与指标记录的训练函数
# ==========================================
def train_and_evaluate_cnn(train_loader, test_loader, device, epochs=30):
    model = SimpleCNN().to(device)
    
    # 使用 AdamW 并增加权重衰减 (weight_decay) 缓解过拟合
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 引入余弦退火学习率调度器 (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = nn.CrossEntropyLoss()

    # 记录训练过程中的各项指标
    stats = {
        "train_time_per_epoch": [],
        "train_loss": [],
        "learning_rate": [],
        "test_acc": 0.0
    }

    print(f"\nStarting CNN Training on {device}...")
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 每个 Epoch 结束后更新学习率
        scheduler.step()
        
        epoch_duration = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        epoch_loss = running_loss / len(train_loader)
        
        # 将指标保存到 stats 字典中
        stats["train_time_per_epoch"].append(epoch_duration)
        stats["train_loss"].append(epoch_loss)
        stats["learning_rate"].append(current_lr)
        
        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_duration:.2f}s")

    # 最终评估精度
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    stats["test_acc"] = 100 * correct / total
    print(f"\nCNN Final Test Accuracy: {stats['test_acc']:.2f}%")
    return stats


# ==========================================
# 3. 主程序执行入口
# ==========================================
if __name__ == '__main__':
    # 1. 环境准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 准备数据：引入数据增强与归一化
    # CIFAR-10 的标准均值和方差
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomRotation(10),     # 小角度随机旋转
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std), # 归一化
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std), # 测试集只做归一化，不做增强
    ])
    
    # 下载/加载数据集
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # 3. 正式调用训练函数 (默认设置为 80 个 Epoch，与你日志中一致)
    results = train_and_evaluate_cnn(train_loader, test_loader, device, epochs=80)