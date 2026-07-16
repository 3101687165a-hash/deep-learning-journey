# 项目二：CNN 图像分类（CIFAR-10）

## 目标

理解卷积神经网络的核心组件：Conv2d、BatchNorm、Dropout、Pooling，以及它们的组合方式。

## 实现了什么

### 模型 (`model.py`)

| 模型 | 架构特点 |
|---|---|
| `CNNClassifier` | 3 个 Conv Block，每 block 包含 Conv×2 → BN → ReLU → Pool → Dropout |
| `ResNetClassifier` | 残差连接版本，包含 `ResidualBlock`，展示 shortcut 连接的实现 |

### 核心知识点

```
Conv2d:
  - 局部感受野（kernel_size=3 → 看 3×3 邻域）
  - 参数共享（同一 kernel 滑过整张图）
  - 参数量: C_in × C_out × K × K
  - 输出尺寸: (W - K + 2P) / S + 1

BatchNorm2d:
  - 对每个通道在 batch 维度上归一化
  - μ, σ² 在 batch 维度计算
  - 训练时用当前 batch 统计量，推理时用 running mean/var

Dropout2d:
  - 以概率 p 随机丢弃整个通道（比普通 Dropout 更适合 CNN）
  - 相当于每次训练不同的子网络 → 集成学习效果
  - 推理时全部激活（PyTorch 自动处理）

MaxPool2d:
  - 降采样，减小计算量
  - 增加平移不变性
  - 2×2 pool + stride=2 → 尺寸减半
```

### 训练 (`train.py`)

- CIFAR-10 数据加载（自动下载）
- 数据增强：RandomCrop + RandomHorizontalFlip
- Cosine Annealing 学习率调度
- 训练/验证/测试完整流程

## 设计取舍

### 为什么用 BN 而非 LN？

CNN 中每个位置是独立的，batch 内的统计量有意义。NLP 中序列长度可变，LN 更合适。

### 为什么用 Kaiming 初始化？

ReLU 在负半轴输出为 0，Xavier 初始化的方差假设在 ReLU 下不成立。Kaiming 初始化针对 ReLU 做了方差修正：W ~ N(0, 2/n_in)。

### Dropout 放哪里？

- 一般在激活函数之后、下一层之前
- CNN 中用 Dropout2d 比 Dropout 更好（丢弃整通道而非单个像素）
- 推理时 PyTorch 自动关闭 dropout

## 运行

```bash
pip install torch torchvision

# CNN 模型
python train.py --model cnn --epochs 50

# ResNet 模型
python train.py --model resnet --epochs 50

# 测试模型
python model.py
```

## 参考

- [Deep Residual Learning (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [Batch Normalization (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167)
- Stanford CS231n — http://cs231n.stanford.edu
