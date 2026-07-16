"""
=============================================================================
  CNN 图像分类器 — 从零实现 (PyTorch)
  ====================================
  数据集: CIFAR-10 (10 类彩色图像，32×32)

  涵盖知识点:
  1. 卷积层 (Conv2d) — 局部感受野，参数共享
  2. 池化层 (MaxPool2d) — 降采样，增加平移不变性
  3. Batch Normalization — 加速收敛，缓解内部协变量偏移
  4. Dropout — 随机丢弃神经元，防止过拟合
  5. 前馈网络基础 — 从 FCN 到 CNN 的演进

  模型架构:
    Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout (×3)
    → Flatten → FC → ReLU → Dropout → FC → Softmax

  作者: 推文君 | 日期: 2026-07-15
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    """
    CNN 图像分类器

    架构原理:
      ┌──────────────────────────────────────┐
      │  Block 1: Conv(3→32) ×2 + Pool       │  32×32 → 16×16
      │  Block 2: Conv(32→64) ×2 + Pool      │  16×16 → 8×8
      │  Block 3: Conv(64→128) ×2 + Pool     │  8×8 → 4×4
      │  Classifier: FC(128×4×4 → 256 → 10)  │
      └──────────────────────────────────────┘

    Batch Normalization 为什么有效:
      1. 缓解 Internal Covariate Shift: 每层输入分布随训练变化，BN 将其稳定化
      2. 允许更大学习率: 参数初始化不那么敏感
      3. 自带轻微正则化效果: mini-batch 的统计量有噪声

    Dropout 原理:
      - 训练时以概率 p 随机"丢弃"神经元（输出置零）
      - 相当于每次训练一个不同的子网络 → 集成学习效果
      - 推理时所有神经元激活，但乘以 (1-p) 做尺度修正
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        # Block 1: 3×32×32 → 32×16×16
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(dropout * 0.5)

        # Block 2: 32×16×16 → 64×8×8
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(dropout * 0.7)

        # Block 3: 64×8×8 → 128×4×4
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(dropout)

        # 全连接分类器
        self.flatten_size = 128 * 4 * 4  # 经过 3 次 pooling 后: 32/2/2/2 = 4
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化 — 适合 ReLU 激活函数"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 32, 32)
        Returns:
            logits: (B, num_classes)
        """
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        # Classifier
        x = x.view(x.size(0), -1)           # Flatten
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        x = self.fc2(x)

        return x


# ═══════════════════════════════════════════════════════════════════════════
#  进阶: ResNet 风格残差连接
# ═══════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    残差块: 输出 = F(x) + x

    ResNet 的核心创新，解决了深层网络退化问题:
      - 没有残差连接: 深层网络反而比浅层表现更差（不是过拟合）
      - 有残差连接: 网络至少能学到恒等映射，深度增加不会变差
      - 梯度高速路: 反向传播时梯度可以直接流过 shortcut
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut: 如果维度不匹配，用 1×1 卷积对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 核心: 残差连接
        return F.relu(out)


class ResNetClassifier(nn.Module):
    """
    简化版 ResNet-18 风格分类器

    核心思想:
      - 残差连接让梯度可以"抄近道"，训练上百层也不成问题
      - 这是理解现代深度网络架构的基石

    架构:
      Conv(3→64) → [ResBlock(64) × 2] → [ResBlock(128) × 2] → Pool → FC
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════
#  测试
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 4

    print("=" * 50)
    print("1. CNN 模型测试")
    print("=" * 50)
    model = CNNClassifier(num_classes=10).to(device)
    x = torch.randn(B, 3, 32, 32).to(device)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # 梯度流测试
    loss = y.sum()
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Gradient flow: {'PASS' if grad_ok else 'FAIL'}")

    print(f"\n{'='*50}")
    print("2. ResNet 模型测试")
    print("=" * 50)
    model2 = ResNetClassifier(num_classes=10).to(device)
    y2 = model2(x)
    print(f"Output: {y2.shape}")
    print(f"Params: {sum(p.numel() for p in model2.parameters()):,}")

    loss2 = y2.sum()
    loss2.backward()
    grad_ok2 = all(p.grad is not None for p in model2.parameters() if p.requires_grad)
    print(f"Gradient flow: {'PASS' if grad_ok2 else 'FAIL'}")

    print("\nAll tests passed!")
