"""
=============================================================================
  从零实现 Transformer 架构 (PyTorch 版)
  ======================================
  参考论文: "Attention Is All You Need" (Vaswani et al., 2017)

  架构总览:
  ┌─────────────────────────────────────────┐
  │  Encoder                                 │
  │  Input → Embedding + PE → N×EncoderLayer │
  │                                          │
  │  Decoder                                 │
  │  Target → Embedding + PE → N×DecoderLayer│
  │                              ↓           │
  │                         Linear + Softmax │
  │                              ↓           │
  │                        Output Probs      │
  └─────────────────────────────────────────┘

  学习重点:
  1. Scaled Dot-Product Attention → 为什么除 sqrt(d_k)?
  2. Multi-Head Attention → 多子空间并行注意力的意义
  3. Positional Encoding → 为何用 sin/cos 而非可学习参数
  4. Layer Normalization vs Batch Normalization
  5. 残差连接 → 缓解深层网络梯度消失
  6. Mask 机制 → Padding Mask & Look-ahead Mask

  作者: 推文君 | 日期: 2026-07-15
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#  1. Scaled Dot-Product Attention (缩放点积注意力)
# ═══════════════════════════════════════════════════════════════════════════

class ScaledDotProductAttention(nn.Module):
    """
    核心公式:
      Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

    为什么除以 √d_k?
      - 当 d_k 较大时，点积 Q·K^T 的方差会变大（~d_k），
        导致 softmax 输出趋近于 one-hot（梯度极小）。
      - 除以 √d_k 将方差控制在 1 附近，保证梯度稳定。

    Mask 机制:
      - Padding Mask: 忽略 <pad> token，将对应位置设为 -inf
      - Look-ahead Mask (Decoder): 防止看到未来 token，上三角设为 -inf
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query:  (batch, n_heads, seq_len_q, d_k)
            key:    (batch, n_heads, seq_len_k, d_k)
            value:  (batch, n_heads, seq_len_k, d_v)
            mask:   (batch, 1, seq_len_q, seq_len_k) or broadcastable
        Returns:
            output: (batch, n_heads, seq_len_q, d_v)
            attn:   (batch, n_heads, seq_len_q, seq_len_k) — 注意力权重
        """
        d_k = query.size(-1)

        # Step 1: 计算注意力分数 scores = Q·K^T / √d_k
        # (B, H, Lq, d_k) × (B, H, d_k, Lk) → (B, H, Lq, Lk)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Step 2: 应用 Mask （将不可见位置设为 -inf）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 3: Softmax 得到注意力权重
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Step 4: 加权求和 (B, H, Lq, Lk) × (B, H, Lk, d_v) → (B, H, Lq, d_v)
        output = torch.matmul(attn, value)

        return output, attn


# ═══════════════════════════════════════════════════════════════════════════
#  2. Multi-Head Attention (多头注意力)
# ═══════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    核心思想:
      将 Q、K、V 线性投影到 h 个不同的子空间（每个维度 d_k = d_model / h），
      在每个子空间独立做 Attention，最后拼接。

    为什么用 Multi-Head?
      - 单头注意力只能关注一种关系模式；
      - 多头允许模型同时关注不同位置的不同表示子空间。
      - 类比：CNN 中的多个卷积核，每个捕捉不同的特征。

    公式:
      MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
      head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # 每个头的维度

        # 线性投影层（合并 Q、K、V 的投影以提高效率）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 (B, L, d_model) 变换为 (B, n_heads, L, d_k)
        """
        B, L, _ = x.size()
        x = x.view(B, L, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (B, n_heads, L, d_k)

    def _concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 (B, n_heads, L, d_k) 合并回 (B, L, d_model)
        """
        B, _, L, _ = x.size()
        x = x.transpose(1, 2).contiguous()  # (B, L, n_heads, d_k)
        return x.view(B, L, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (B, L, d_model)
            mask: (B, 1, Lq, Lk) or None
        Returns:
            output: (B, Lq, d_model)
            attn:   (B, n_heads, Lq, Lk)
        """
        # 线性投影 + 分头
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        # Scaled Dot-Product Attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 拼回头 + 输出投影
        output = self._concat_heads(attn_output)
        output = self.W_o(output)
        output = self.dropout(output)

        return output, attn_weights


# ═══════════════════════════════════════════════════════════════════════════
#  3. Positional Encoding (位置编码)
# ═══════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Transformer 本身没有递归或卷积结构，对序列顺序无感知。
    位置编码将位置信息注入输入表示。

    使用正弦/余弦函数:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    为什么用 sin/cos 而非可学习参数?
      1. 可以泛化到训练时未见过的序列长度
      2. 相对位置信息可通过线性变换获得：
         PE(pos+k) 可由 PE(pos) 线性表示 → sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
      3. 不同频率的正弦波让模型能区分不同尺度的位置关系
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # div_term: 10000^(2i/d_model) 的对数形式，避免数值不稳定
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        # 偶数位用 sin，奇数位用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (max_len, d_model) → (1, max_len, d_model) 方便 broadcast
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 不参与梯度更新，但会随模型 save/load

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        Returns:
            x + pe: (B, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════════════════════
#  4. Position-wise Feed-Forward Network (逐位置前馈网络)
# ═══════════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    对每个位置独立应用相同的两层全连接网络:

    FFN(x) = ReLU(x·W1 + b1)·W2 + b2
    内部维度 d_ff 通常是 d_model 的 4 倍

    作用:
      - 引入非线性变换，增强模型的表达能力
      - 两层结构提供了"记忆-处理-输出"的模式
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        Returns:
            output: (B, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ═══════════════════════════════════════════════════════════════════════════
#  5. Encoder Layer (编码器层)
# ═══════════════════════════════════════════════════════════════════════════
#  架构:  x → MultiHeadAttention → Add&Norm → FFN → Add&Norm

class EncoderLayer(nn.Module):
    """
    单层 Encoder 的数据流:
      ┌─────────────────────────┐
      │  Input: x               │
      │    ↓                    │
      │  Self-Attention (Q=K=V=x)
      │    ↓                    │
      │  x + Dropout(Attn(x))   │  ← 残差连接 (Residual Connection)
      │    ↓                    │
      │  LayerNorm              │  ← 层归一化
      │    ↓                    │
      │  FFN                    │
      │    ↓                    │
      │  x + Dropout(FFN(x))    │  ← 残差连接
      │    ↓                    │
      │  LayerNorm → Output     │
      └─────────────────────────┘

    残差连接 (Residual Connection) 的意义:
      - 提供梯度高速公路，缓解深层网络的梯度消失
      - 允许模型学习恒等映射，训练更稳定

    Layer Normalization vs Batch Normalization:
      - BN: 对 batch 维度归一化，依赖 batch size，不适用于可变长度序列
      - LN: 对 feature 维度归一化，独立于 batch，更适合 NLP
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: (B, seq_len, d_model)
            mask: (B, 1, seq_len, seq_len)  自注意力 mask (通常为 padding mask)
        Returns:
            output: (B, seq_len, d_model)
            attn:   (B, n_heads, seq_len, seq_len)
        """
        # Sub-layer 1: Multi-Head Self-Attention + Add & Norm
        attn_output, attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差 + LN

        # Sub-layer 2: Feed-Forward + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))   # 残差 + LN

        return x, attn


# ═══════════════════════════════════════════════════════════════════════════
#  6. Decoder Layer (解码器层)
# ═══════════════════════════════════════════════════════════════════════════
#  架构:  x → Masked MHA → Add&Norm → Cross MHA → Add&Norm → FFN → Add&Norm

class DecoderLayer(nn.Module):
    """
    单层 Decoder 的数据流:
      ┌──────────────────────────────────┐
      │  Input: x (target sequence)       │
      │    ↓                              │
      │  Masked Self-Attention           │  ← Look-ahead mask: 不能看未来
      │    ↓                              │
      │  Add & LayerNorm                  │
      │    ↓                              │
      │  Cross-Attention                 │  ← Q 来自 decoder，K/V 来自 encoder
      │    ↓                              │
      │  Add & LayerNorm                  │
      │    ↓                              │
      │  FFN                             │
      │    ↓                              │
      │  Add & LayerNorm → Output        │
      └──────────────────────────────────┘

    Look-ahead Mask (因果掩码):
      - 训练时：防止 decoder 看到未来 token
      - 上三角矩阵设为 -inf，确保位置 i 只能看到位置 1..i

    Cross-Attention:
      - Q 来自 decoder 的当前表示
      - K、V 来自 encoder 的最终输出
      - 让 decoder 能够"查阅"源序列信息
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                src_mask=None, tgt_mask=None):
        """
        Args:
            x:           (B, tgt_len, d_model)  — decoder input
            enc_output:  (B, src_len, d_model)  — encoder output
            src_mask:    用于 cross-attention 的 padding mask
            tgt_mask:    look-ahead mask (因果掩码)
        Returns:
            output:  (B, tgt_len, d_model)
            self_attn, cross_attn: 注意力权重
        """
        # Sub-layer 1: Masked Self-Attention + Add & Norm
        self_attn_output, self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Sub-layer 2: Cross-Attention (Q=dec, K=enc, V=enc) + Add & Norm
        cross_attn_output, cross_attn = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Sub-layer 3: Feed-Forward + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x, self_attn, cross_attn


# ═══════════════════════════════════════════════════════════════════════════
#  7. Encoder (编码器)
# ═══════════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """
    N 层 EncoderLayer 堆叠。
    输入 → Embedding + PE → N × EncoderLayer → 输出
    """

    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 d_ff: int = 2048, n_layers: int = 6, dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x:     (B, src_len)  源序列 token IDs
            mask:  (B, 1, src_len, src_len)  padding mask
        Returns:
            output: (B, src_len, d_model)
        """
        # Embedding 缩放: 嵌入向量乘以 √d_model，防止嵌入值相对于位置编码过小
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # 逐层传递
        for layer in self.layers:
            x, _ = layer(x, mask)

        return x


# ═══════════════════════════════════════════════════════════════════════════
#  8. Decoder (解码器)
# ═══════════════════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """
    N 层 DecoderLayer 堆叠。
    输入 → Embedding + PE → N × DecoderLayer → 输出
    """

    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 d_ff: int = 2048, n_layers: int = 6, dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                src_mask=None, tgt_mask=None):
        """
        Args:
            x:           (B, tgt_len)  目标序列 token IDs
            enc_output:  (B, src_len, d_model)
            src_mask:    padding mask for cross-attention
            tgt_mask:    look-ahead mask
        Returns:
            output: (B, tgt_len, d_model)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x, _, _ = layer(x, enc_output, src_mask, tgt_mask)

        return x


# ═══════════════════════════════════════════════════════════════════════════
#  9. 完整 Transformer 模型
# ═══════════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    完整的 Transformer 模型:
      Encoder × N → Decoder × N → Linear + Softmax

    用途（本项目的两种使用方式）:
      1. Seq2Seq 模式: 机器翻译等 (有独立的 Encoder 输入和 Decoder 输入)
      2. 字符级语言模型: 仅用 Decoder 做自回归生成 (见 train.py)
    """

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, n_heads: int = 8, d_ff: int = 2048,
                 n_layers: int = 6, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_heads,
                               d_ff, n_layers, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads,
                               d_ff, n_layers, dropout, max_len)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # 参数初始化: Xavier/Glorot
        self._init_parameters()

    def _init_parameters(self):
        """Xavier 初始化：有助于稳定训练初期的梯度流"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask=None, tgt_mask=None):
        """
        Args:
            src: (B, src_len)  源序列
            tgt: (B, tgt_len)  目标序列
        Returns:
            logits: (B, tgt_len, tgt_vocab_size)  每个位置在词表上的 logits
        """
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        logits = self.output_proj(dec_output)
        return logits

    def encode(self, src: torch.Tensor, src_mask=None):
        """仅编码，返回 encoder 输出（用于推理时复用）"""
        return self.encoder(src, src_mask)

    def decode(self, tgt: torch.Tensor, enc_output: torch.Tensor,
               src_mask=None, tgt_mask=None):
        """仅解码"""
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return self.output_proj(dec_output)


# ═══════════════════════════════════════════════════════════════════════════
#  10. Mask 生成工具函数
# ═══════════════════════════════════════════════════════════════════════════

def generate_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    生成 Padding Mask: 标记真实 token (1) 和 padding token (0)

    Args:
        seq: (B, L)  token IDs
        pad_idx:    padding token 的索引
    Returns:
        mask: (B, 1, 1, L)  在 attention 中用于 key 维度
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)


def generate_lookahead_mask(seq_len: int) -> torch.Tensor:
    """
    生成 Look-ahead Mask (因果掩码):
      位置 i 只能关注位置 0..i，不能看到位置 i+1 及之后。

    Example (seq_len=4):
      [[1, 0, 0, 0],
       [1, 1, 0, 0],
       [1, 1, 1, 0],
       [1, 1, 1, 1]]

    Returns:
        mask: (1, 1, seq_len, seq_len)  下三角矩阵
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


# ═══════════════════════════════════════════════════════════════════════════
#  11. 测试代码
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Transformer 架构测试")
    print("=" * 60)

    # --- 配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, SRC_LEN, TGT_LEN = 2, 10, 8
    SRC_VOCAB, TGT_VOCAB = 100, 100
    D_MODEL, N_HEADS, N_LAYERS = 256, 8, 2

    print(f"Device: {device}")
    print(f"Config: d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}")

    # --- 创建模型 ---
    model = Transformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=0.1
    ).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # --- 构造假数据 ---
    src = torch.randint(1, SRC_VOCAB, (B, SRC_LEN)).to(device)
    tgt = torch.randint(1, TGT_VOCAB, (B, TGT_LEN)).to(device)

    # Masks
    src_pad_mask = generate_padding_mask(src, pad_idx=0).to(device)
    tgt_lookahead_mask = generate_lookahead_mask(TGT_LEN).to(device)

    # --- 前向传播测试 ---
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt, src_pad_mask, tgt_lookahead_mask)

    print(f"\nInput shapes: src={src.shape}, tgt={tgt.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected output: ({B}, {TGT_LEN}, {TGT_VOCAB})")
    assert logits.shape == (B, TGT_LEN, TGT_VOCAB), "Output shape mismatch!"

    # --- 梯度流测试 ---
    model.train()
    logits = model(src, tgt, src_pad_mask, tgt_lookahead_mask)
    loss = logits.sum()
    loss.backward()

    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"\nGradient flow: {'PASS' if grad_ok else 'FAIL'}")
    print(f"Loss value: {loss.item():.4f}")

    print("\nAll tests passed!")
