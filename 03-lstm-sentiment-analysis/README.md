# 项目三：LSTM 文本情感分析

## 目标

理解循环神经网络的时序建模能力，掌握 LSTM/GRU 的门控机制，学会处理变长序列的 Pack/Pad 技术。

## 实现了什么

### 模型 (`model.py`)

| 组件 | 用途 |
|---|---|
| `SimpleRNNCell` | 基础 RNN — 理解 h_t = tanh(W·[h_{t-1}, x_t]) |
| `LSTMCell` | 手写 LSTM — 理解遗忘门/输入门/输出门的数学 |
| `SentimentClassifier` | 生产级分类器 — 支持 LSTM/GRU，双向，多种池化 |

### 核心理解

```
RNN 的问题:
  h_t = tanh(W_h·h_{t-1} + W_x·x_t)
  梯度: ∂L/∂h_0 = ∂L/∂h_T · Π(∂h_i/∂h_{i-1})
  连乘导致梯度指数级衰减 → 长距离依赖丢失

LSTM 的解决方案:
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t     (细胞状态 — 信息高速公路)
  h_t = o_t ⊙ tanh(c_t)                (隐藏状态 — 对外可见)
  
  关键: c_t 的更新是加法操作，加法不衰减梯度！
  梯度可以通过 c_t 这条"高速公路"无损回传。

遗忘门 f_t: σ(W_f·[h_{t-1}, x_t]) → [0,1]
  - 0: 完全遗忘 → 擦除无用信息
  - 1: 完全保留 → 保持长期记忆

输入门 i_t: σ(W_i·[h_{t-1}, x_t]) → [0,1]
  - 控制新信息写入的幅度

输出门 o_t: σ(W_o·[h_{t-1}, x_t]) → [0,1]
  - 控制多少记忆对外可见

GRU (简化版):
  合并遗忘门和输入门为"更新门" z_t
  合并细胞状态和隐藏状态
  参数量更少，效果接近 LSTM
```

### Pack/Pad 技术

```python
# 问题: batch 中序列长度不同，padding 产生大量无效计算
# 解决: pack_padded_sequence 跳过 padding 部分
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

packed = pack_padded_sequence(embedded, lengths, batch_first=True)
output, _ = rnn(packed)
unpacked, _ = pad_packed_sequence(output, batch_first=True)
```

### 双向 RNN

```
单向: h_t = RNN(h_{t-1}, x_t)              → 只看过去
双向: h_t^f = RNN(h_{t-1}^f, x_t)          → 前向
      h_t^b = RNN(h_{t+1}^b, x_t)          → 后向
      h_t = [h_t^f; h_t^b]                  → 拼接 → 看过全文
```

## 运行

```bash
pip install torch

# LSTM 模型
python train.py --rnn_type lstm --epochs 15

# GRU 模型
python train.py --rnn_type gru --epochs 15

# 测试模型
python model.py
```

## 参考

- [Understanding LSTM Networks (Christopher Olah)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Empirical Evaluation of GRU (Chung et al., 2014)](https://arxiv.org/abs/1412.3555)
- Stanford CS224n — http://web.stanford.edu/class/cs224n
