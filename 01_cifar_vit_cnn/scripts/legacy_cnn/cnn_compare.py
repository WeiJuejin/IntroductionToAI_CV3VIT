import os
import random

import matplotlib
matplotlib.use('Agg') 


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_dataloaders(data_root, batch_size, img_size, ratio, seed):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    total_size = len(train_set)
    subset_size = max(1, int(total_size * ratio))

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator)[:subset_size].tolist()
    subset_train = Subset(train_set, indices)

    train_loader = DataLoader(
        subset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader, subset_size


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs, tag):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"{tag} Epoch {epoch}/{total_epochs}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def plot_loss_curves(results, save_dir):
    plt.figure(figsize=(10, 6))
    for tag, history in results.items():
        plt.plot(history["train_loss"], marker="o", label=f"{tag} Train Loss")
        plt.plot(history["val_loss"], marker="s", linestyle="--", label=f"{tag} Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss Curves Under Different Training Data Ratios")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cnn_loss_curves.png"), dpi=300)
    plt.show()


def plot_accuracy_bar(results, save_dir):
    labels = list(results.keys())
    accuracies = [results[label]["best_acc"] for label in labels]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, accuracies, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylabel("Accuracy (%)")
    plt.title("CNN Test Accuracy Comparison")
    plt.ylim(0, max(accuracies) + 5)

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cnn_accuracy_comparison.png"), dpi=300)
    plt.show()


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50
    batch_size = 64
    lr = 1e-3
    img_size = 64
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base_dir, "data")
    save_dir = os.path.join(base_dir, "cnn_outputs")
    os.makedirs(save_dir, exist_ok=True)

    ratios = {
        "100%": 1.0,
        "20%": 0.2,
        "10%": 0.1,
    }

    results = {}

    print(f"Using device: {device}")
    print("Dataset: CIFAR-10")

    for tag, ratio in ratios.items():
        print(f"\n{'=' * 60}")
        print(f"Training with {tag} of the training data")
        print(f"{'=' * 60}")

        train_loader, test_loader, subset_size = get_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            img_size=img_size,
            ratio=ratio,
            seed=42,
        )

        print(f"Training samples used: {subset_size}")

        model = SimpleCNN(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []
        val_accuracies = []
        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                epochs,
                tag,
            )
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            best_acc = max(best_acc, val_acc)

            print(
                f"{tag} | Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {val_loss:.4f} | "
                f"Test Acc: {val_acc:.2f}%"
            )

        results[tag] = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_acc": val_accuracies,
            "best_acc": best_acc,
        }

    print("\nFinal Accuracy Comparison:")
    for tag, history in results.items():
        print(f"{tag} training data -> Best Test Accuracy: {history['best_acc']:.2f}%")

    plot_loss_curves(results, save_dir)
    plot_accuracy_bar(results, save_dir)

    print(f"\nPlots saved to: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()
