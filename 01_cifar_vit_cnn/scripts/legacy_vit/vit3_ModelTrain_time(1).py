import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time  # 新增：用于计时

# ==========================================
# Mixup
# ==========================================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


# ==========================================
# Model
# ==========================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=256):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.attn_map = None

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_map = attn.detach() 

        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 256
        self.patch = PatchEmbedding(embed_dim=dim)

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 65, dim))

        self.blocks = nn.Sequential(*[
            Block(dim, heads=8) for _ in range(8)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        B = x.size(0)

        x = self.patch(x)

        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos
        x = self.blocks(x)

        x = self.norm(x[:, 0])
        return self.head(x)

def visualize_attention(model, img_tensor, device):
    model.eval()
    with torch.no_grad():
        _ = model(img_tensor.to(device))

    attn_weights = model.blocks[-1].attn.attn_map
    attn_weights = torch.mean(attn_weights, dim=1)
    cls_attn = attn_weights[0, 0, 1:]

    grid_size = int(np.sqrt(cls_attn.size(0)))
    mask = cls_attn.reshape(grid_size, grid_size).cpu().numpy()

    img_show = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    img_show = (img_show * 0.5) + 0.5 
    img_show = np.clip(img_show, 0, 1)

    mask = cv2.resize(mask / mask.max(), (32, 32)) 
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_show)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(img_show)
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.title("Attention Heatmap")
    plt.show()

# ==========================================
# Training
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = 80
    BATCH_SIZE = 128
    LR = 2e-4

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    model = ViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-5)
        ],
        milestones=[5]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- 准备记录时长 ---
    epoch_times = []

    print("开始训练...")
    for epoch in range(EPOCHS):
        start_time = time.time()  # 记录每轮开始时间
        
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)

            optimizer.zero_grad()

            outputs = model(imgs)

            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        
        end_time = time.time()  # 记录每轮结束时间
        duration = end_time - start_time
        epoch_times.append(duration)
        print(f"Epoch {epoch+1} 完成，耗时: {duration:.2f} 秒")

    # --- 保存模型参数 ---
    save_path = "vit3_parameters.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型参数已保存至 {save_path}")

    # --- 生成时长记录文件 ---
    log_path = "training_time100%.txt"
    with open(log_path, "w") as f:
        f.write("Epoch, Duration(seconds)\n")
        for i, t in enumerate(epoch_times):
            f.write(f"{i+1}, {t:.2f}\n")
        f.write(f"\nTotal Training Time: {sum(epoch_times):.2f} seconds")
    print(f"训练时长记录已保存至 {log_path}")

if __name__ == "__main__":
    main()









