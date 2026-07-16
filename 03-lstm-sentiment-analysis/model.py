"""
=============================================================================
  LSTM/GRU 文本情感分析器 — 从零实现 (PyTorch)
  ==============================================
  数据集: IMDB 电影评论 (二分类: 正面/负面)

  涵盖知识点:
  1. RNN 基础 — 循环网络的时序依赖与 BPTT
  2. LSTM — 门控机制解决长距离梯度消失
  3. GRU — LSTM 的简化变体
  4. 词嵌入 (Embedding) — 离散 token → 稠密向量
  5. 序列模型的 Pack/Pad 技术

  为什么 RNN 适合序列?
    - 天然处理变长输入
    - 隐藏状态 h_t 携带历史信息: h_t = f(h_{t-1}, x_t)
    - 参数共享: 每个时间步使用相同的权重

  为什么 LSTM 比普通 RNN 好?
    普通 RNN: 梯度随序列长度指数级衰减/爆炸（长期依赖问题）
    LSTM:   门控机制 (遗忘门、输入门、输出门) 让梯度选择性流动，
            解决了长距离依赖问题

  作者: 推文君 | 日期: 2026-07-15
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ═══════════════════════════════════════════════════════════════════════════
#  1. 基础 RNN Cell (纯手工，用于理解原理)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleRNNCell(nn.Module):
    """
    基础 RNN 单元 — 手动实现以理解数学原理

    公式: h_t = tanh(W_ih·x_t + b_ih + W_hh·h_{t-1} + b_hh)

    这是理解 LSTM 的前置知识。
    注意: 这个简化版只能学习短期依赖，长序列会梯度消失。
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x_t, h_prev):
        """
        Args:
            x_t: (B, input_size)   当前时刻输入
            h_prev: (B, hidden_size) 上一时刻隐藏状态
        Returns:
            h_t: (B, hidden_size)
        """
        return torch.tanh(self.W_ih(x_t) + self.W_hh(h_prev))


# ═══════════════════════════════════════════════════════════════════════════
#  2. LSTM 手工实现（教育用途，帮助理解门控）
# ═══════════════════════════════════════════════════════════════════════════

class LSTMCell(nn.Module):
    """
    LSTM 单元 — 手工实现

    四个门（本质都是带 sigmoid/tanh 的全连接层）:

    遗忘门: f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
      决定保留多少旧记忆 → 值越接近 0，忘得越多

    输入门: i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
      决定写入多少新信息

    候选记忆: g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)
      新信息的候选值

    记忆更新: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
      遗忘旧信息 + 写入新信息

    输出门: o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
      决定输出多少记忆

    隐藏状态: h_t = o_t ⊙ tanh(c_t)
      控制对外可见的信息

    关键洞察:
      - 遗忘门允许错误信息被选择性擦除 → 梯度可以无损传播
      - 细胞状态 c_t 是"信息高速公路"，加法操作不衰减梯度
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # 合并四个门的线性变换，提高效率
        self.gate = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x_t, state):
        """
        Args:
            x_t: (B, input_size)
            state: (h_{t-1}, c_{t-1})  各 (B, hidden_size)
        Returns:
            h_t: (B, hidden_size)
            c_t: (B, hidden_size)
        """
        h_prev, c_prev = state

        # 拼接 [h_{t-1}, x_t]
        combined = torch.cat([h_prev, x_t], dim=-1)  # (B, hidden + input)

        # 一次矩阵乘法算出四个门
        gates = self.gate(combined)  # (B, 4 * hidden)
        f, i, g, o = gates.chunk(4, dim=-1)

        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # 细胞状态更新
        c_t = f * c_prev + i * g

        # 隐藏状态
        h_t = o * torch.tanh(c_t)

        return h_t, c_t


# ═══════════════════════════════════════════════════════════════════════════
#  3. 情感分类模型 (生产级 — 使用 PyTorch 内置 LSTM/GRU)
# ═══════════════════════════════════════════════════════════════════════════

class SentimentClassifier(nn.Module):
    """
    基于 LSTM/GRU 的文本情感分类器。

    架构:
      Input tokens → Embedding → LSTM/GRU → Pool/Last → FC → Softmax

    三种池化策略:
      - last: 取最后一个时间步的隐藏状态（最常用）
      - mean: 对所有时间步取平均
      - max:  对所有时间步取最大值
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 256, num_layers: int = 2,
                 num_classes: int = 2, dropout: float = 0.3,
                 rnn_type: str = "lstm", bidirectional: bool = True,
                 pooling: str = "last"):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.pooling = pooling

        # Embedding 层: 将 token ID 映射为稠密向量
        # 训练好的 embedding 向量会捕捉语义关系：
        #   例如: vec("king") - vec("man") + vec("woman") ≈ vec("queen")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # RNN 层
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,         # 输入格式: (B, L, D)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 分类头
        self.dropout = nn.Dropout(dropout)
        if pooling == "last":
            self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)
        else:
            self.fc = nn.Linear(hidden_dim * self.num_directions * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor, lengths=None):
        """
        Args:
            x: (B, L)  token IDs
            lengths: (B,) 每个序列的实际长度（用于 pack_padded）
        Returns:
            logits: (B, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)  # (B, L, embed_dim)

        # Pack 技术: 跳过 padding 部分的计算，加速训练
        if lengths is not None:
            embedded = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # RNN
        rnn_out, hidden = self.rnn(embedded)

        # Unpack
        if lengths is not None:
            rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        # 池化提取最终表示
        if self.pooling == "last":
            if lengths is not None:
                # 取每个序列最后一个有效 token 的输出
                idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, rnn_out.size(-1)).to(rnn_out.device)
                rep = rnn_out.gather(1, idx).squeeze(1)
            else:
                rep = rnn_out[:, -1, :]
        elif self.pooling == "mean":
            rep = rnn_out.mean(dim=1)
        elif self.pooling == "max":
            rep = rnn_out.max(dim=1)[0]
        else:
            # attention: 加权池化
            attn_weights = F.softmax(
                self.attn(rnn_out).squeeze(-1), dim=-1
            )  # (B, L)
            rep = torch.bmm(attn_weights.unsqueeze(1), rnn_out).squeeze(1)

        # 分类
        rep = self.dropout(rep)
        return self.fc(rep)


# ═══════════════════════════════════════════════════════════════════════════
#  测试
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, L, VOCAB = 4, 50, 1000

    print("=" * 50)
    print("1. 手工 LSTM Cell 测试")
    print("=" * 50)
    cell = LSTMCell(128, 256).to(device)
    x_t = torch.randn(B, 128).to(device)
    h_prev = torch.zeros(B, 256).to(device)
    c_prev = torch.zeros(B, 256).to(device)
    h_t, c_t = cell(x_t, (h_prev, c_prev))
    print(f"h_t: {h_t.shape}, c_t: {c_t.shape}")

    print(f"\n{'='*50}")
    print("2. SentimentClassifier (LSTM) 测试")
    print("=" * 50)
    model = SentimentClassifier(
        vocab_size=VOCAB, embed_dim=128, hidden_dim=128,
        num_layers=2, rnn_type="lstm", bidirectional=True
    ).to(device)

    x = torch.randint(1, VOCAB, (B, L)).to(device)
    lengths = torch.tensor([L, L-5, L-10, L-15]).to(device)
    y = model(x, lengths)
    print(f"Input: {x.shape}, lengths: {lengths}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # 梯度流
    loss = y.sum()
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Gradient flow: {'PASS' if grad_ok else 'FAIL'}")

    print(f"\n{'='*50}")
    print("3. SentimentClassifier (GRU) 测试")
    print("=" * 50)
    model2 = SentimentClassifier(
        vocab_size=VOCAB, embed_dim=128, hidden_dim=128,
        num_layers=2, rnn_type="gru", bidirectional=True
    ).to(device)
    y2 = model2(x, lengths)
    print(f"Output: {y2.shape}")
    print(f"Params: {sum(p.numel() for p in model2.parameters()):,}")

    print("\nAll tests passed!")
