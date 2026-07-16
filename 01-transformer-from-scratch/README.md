# 项目一：从零实现 Transformer

## 目标

**吃透 Transformer 架构的每一个组件**——不是调用 `nn.Transformer`，而是从 `nn.Linear` 和矩阵乘法开始手写。

## 实现了什么

### 核心组件 (`transformer.py`)

| 组件 | 数学原理 | 实现方式 |
|---|---|---|
| `ScaledDotProductAttention` | softmax(QK^T / √d_k)V | `torch.matmul` + `F.softmax` |
| `MultiHeadAttention` | Concat(head_1...head_h)W^O | 拆头/拼头 + 线性投影 |
| `PositionalEncoding` | sin/cos 位置编码 | `register_buffer` (不参与梯度) |
| `PositionwiseFeedForward` | ReLU(xW1+b1)W2+b2 | 双层全连接 |
| `EncoderLayer` | Self-Attn → Add&Norm → FFN → Add&Norm | 残差 + LayerNorm |
| `DecoderLayer` | Masked Self-Attn → Cross-Attn → FFN | 因果掩码 + 交叉注意力 |

### 字符级语言模型 (`train.py`)

- **Decoder-Only 架构**（类 GPT）
- **字符级分词器**（每个字一个 token）
- **Noam 学习率调度器**（warmup + 1/√step 衰减）
- **自回归文本生成**（temperature 控制多样性）

## 关键理解

### 1. 为什么除以 √d_k？

```
Q·K^T 的方差 ≈ d_k
→ softmax 输出趋近 one-hot → 梯度极小
→ 除以 √d_k 将方差控制到 ≈1 → 梯度稳定
```

### 2. Multi-Head 的意义

- 单头：模型只能关注一种 patterns
- 多头：8 个并行的"注意维度"，每个捕捉不同的句法/语义关系
- 类比 CNN 的多个卷积核

### 3. LayerNorm vs BatchNorm

```
BN: 对一个 batch 内所有样本的同一特征归一化 → 依赖 batch size
LN: 对单个样本的所有特征归一化 → 适合变长序列
```

### 4. 残差连接为什么重要？

```
Without shortcut: 梯度逐层连乘 → 指数衰减
With shortcut:   梯度可通过 shortcut 直接回传 → 训练上百层也没问题
```

## 运行

```bash
# 安装依赖
pip install torch

# 训练字符级语言模型
python train.py --mode train --d_model 128 --n_heads 4 --n_layers 3 --epochs 20

# 生成文本
python train.py --mode generate --prompt "江湖"

# 测试模型架构
python transformer.py
```

## 参考

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/annotated-transformer/)
- [李沐《动手学深度学习》第11章](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
