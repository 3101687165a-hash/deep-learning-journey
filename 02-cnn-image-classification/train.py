"""
CNN 图像分类 — CIFAR-10 训练脚本
=================================

运行方式:
    python train.py --model cnn
    python train.py --model resnet
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNNClassifier, ResNetClassifier


# ═══════════════════════════════════════════════════════════════════════════
#  1. 数据准备 (Data Augmentation + Normalization)
# ═══════════════════════════════════════════════════════════════════════════

def get_dataloaders(batch_size: int = 128, num_workers: int = 2):
    """
    CIFAR-10 数据加载。

    数据增强 (Data Augmentation) — 为什么重要?
      - RandomCrop + RandomHorizontalFlip → 增加训练数据的多样性
      - 相当于在不增加标注成本的情况下扩充数据集
      - 有效缓解过拟合

    Normalization:
      - 使用 CIFAR-10 的统计均值/标准差
      - 将数据归一化到零均值、单位方差 → 加速训练收敛
    """

    # 训练集增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),        # 随机裁剪 + padding
        transforms.RandomHorizontalFlip(),           # 水平翻转
        transforms.ToTensor(),
        transforms.Normalize(                        # 每个通道分别归一化
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])

    # 测试集只做 Normalization（不做增强）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])

    # 下载 CIFAR-10
    print("Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    classes = train_dataset.classes
    print(f"  Train: {len(train_dataset)} images, Test: {len(test_dataset)} images")
    print(f"  Classes: {classes}")

    return train_loader, test_loader, classes


# ═══════════════════════════════════════════════════════════════════════════
#  2. 训练和评估函数
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """单 epoch 训练，返回平均 loss 和准确率"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # 标准训练三步走 + 清零
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if scheduler:
        scheduler.step()

    return running_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估函数 — torch.no_grad() 禁用梯度计算"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100. * correct / total


# ═══════════════════════════════════════════════════════════════════════════
#  3. 学习率调度
# ═══════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler:
    """
    Warmup + Cosine Annealing 学习率调度:
      - 前 warmup_epochs 线性增加到 target_lr
      - 之后按余弦曲线衰减到 min_lr

    为什么需要 Warmup?
      - 训练初期模型参数是随机的，梯度大且不稳定
      - 直接用小 lr 慢慢"热身"，避免模型走偏
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, target_lr, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _get_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            return self.target_lr * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            return self.min_lr + 0.5 * (self.target_lr - self.min_lr) * \
                   (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════════
#  4. 主函数
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据
    train_loader, test_loader, classes = get_dataloaders(args.batch_size)

    # 模型
    if args.model == "cnn":
        model = CNNClassifier(num_classes=10, dropout=args.dropout).to(device)
    else:
        model = ResNetClassifier(num_classes=10).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()} ({n_params:,} params)")

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    print(f"\n{'='*60}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Test Loss':>12} {'Test Acc':>10} {'Time':>8}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        print(f"{epoch:6d} {train_loss:12.4f} {train_acc:9.2f}% {test_loss:12.4f} {test_acc:9.2f}% {elapsed:7.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.model}_best.pth"))

    print(f"{'='*60}")
    print(f"Best Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
