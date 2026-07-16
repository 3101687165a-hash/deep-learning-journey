# 深度学习实战仓库 — Deep Learning Journey

> 不仅会用框架，更能独立调试模型、理解常见架构。

## 仓库概览

| 项目 | 架构 | 领域 | 知识点 |
|---|---|---|---|
| [01-transformer-from-scratch](./01-transformer-from-scratch/) | Transformer | NLP / 序列建模 | Self-Attention, Multi-Head, PE, LN, 残差连接 |
| [02-cnn-image-classification](./02-cnn-image-classification/) | CNN / ResNet | 计算机视觉 | Conv2d, BN, Dropout, 残差连接 |
| [03-lstm-sentiment-analysis](./03-lstm-sentiment-analysis/) | LSTM / GRU | NLP | 门控机制, Embedding, Pack/Pad, 双向RNN |

## 学习路线

```
理论学习 (3-4周)                    框架精通 (2-3周)                实战产出 (2-3周)
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│ 前馈网络 & 反向传播  │   ──▶   │ PyTorch 基础操作    │   ──▶   │ 项目1: CNN 图像分类  │
│ 正则化, BN, Dropout │          │ nn.Module        │          │ 项目2: LSTM 情感分析 │
│ CNN / RNN / LSTM  │          │ DataLoader       │          │ 项目3: Transformer │
│ Transformer ★★★   │          │ 自动求导 & 训练循环  │          │ README & 代码整理   │
│                    │          │ 分布式训练 (DDP)   │          │ GitHub 仓库        │
└──────────────────┘          └──────────────────┘          └──────────────────┘
```

## 快速开始

### 环境安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy tqdm
```

### 运行项目

```bash
# 项目1: CNN 图像分类
cd 02-cnn-image-classification
python train.py --model cnn --epochs 50

# 项目2: LSTM 情感分析
cd 03-lstm-sentiment-analysis
python train.py --rnn_type lstm --epochs 15

# 项目3: Transformer 字符级语言模型
cd 01-transformer-from-scratch
python train.py --mode train --epochs 20
python train.py --mode generate --prompt "江湖"
```

## 推荐学习资源

### 理论
- 李沐《动手学深度学习》(PyTorch 版) — https://d2l.ai
- Stanford CS231n (计算机视觉) — http://cs231n.stanford.edu
- Stanford CS224n (自然语言处理) — http://web.stanford.edu/class/cs224n
- 吴恩达 Deep Learning Specialization (Coursera)

### 论文必读
1. **"Attention Is All You Need"** (Vaswani et al., 2017) — Transformer 原论文
2. **"Deep Residual Learning for Image Recognition"** (He et al., 2015) — ResNet
3. **"Batch Normalization"** (Ioffe & Szegedy, 2015)
4. **"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"** (Srivastava et al., 2014)

### 框架
- PyTorch 官方文档 — https://pytorch.org/docs/stable
- PyTorch 官方教程 — https://pytorch.org/tutorials

## 进阶目标

- [ ] 引入 DDP 分布式训练
- [ ] 项目1: 在 ImageNet 子集上训练 ResNet
- [ ] 项目2: 用真实 IMDB 数据训练 LSTM
- [ ] 项目3: 实现论文中的完整 Encoder-Decoder Transformer
- [ ] 对比不同优化器 (SGD/Adam/AdamW) 的性能差异
- [ ] 添加 TensorBoard 可视化
- [ ] 模型量化和导出 ONNX

## 作者

@推文君 — 专注 AI 小说推文赛道，用深度学习提升生产效率。

---

*这个仓库是学习过程的见证。每个项目从零写起，理解每一行代码背后的数学。*
