"""
字符级 Transformer 语言模型 — 训练脚本
=========================================

这个脚本展示了:
1. 从零构建 DataLoader，理解数据加载流程
2. 自定义训练循环，理解 loss 计算和反向传播
3. 学习率调度 (Noam Scheduler)
4. 文本生成 (自回归解码)
5. 模型 checkpoint 保存与加载

运行方式:
    python train.py --mode train
    python train.py --mode generate  # 生成文本
"""

import os
import sys
import math
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 导入我们的 Transformer 实现
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer import Transformer, generate_padding_mask, generate_lookahead_mask


# ═══════════════════════════════════════════════════════════════════════════
#  1. 字符级 Tokenizer
# ═══════════════════════════════════════════════════════════════════════════

class CharTokenizer:
    """
    字符级分词器：每个字符就是一个 token。
    简单的映射 char <-> idx，外加 <pad>、<sos>、<eos>、<unk> 特殊 token。

    这是理解 tokenization 最直观的方式——
    不需要 BPE、WordPiece 等复杂算法。
    """

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"   # Start of Sequence
    EOS_TOKEN = "<eos>"   # End of Sequence
    UNK_TOKEN = "<unk>"   # Unknown

    def __init__(self):
        # 特殊 token 始终在前
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        self.idx2char = {}
        self.char2idx = {}
        self.vocab_size = 0

    def build_vocab(self, texts):
        """从文本列表中构建词表"""
        chars = set()
        for text in texts:
            chars.update(text)

        # 合并特殊 token 和文本字符
        all_chars = self.special_tokens + sorted(list(chars))
        self.idx2char = {i: ch for i, ch in enumerate(all_chars)}
        self.char2idx = {ch: i for i, ch in enumerate(all_chars)}
        self.vocab_size = len(all_chars)

        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = True):
        """将文本转为 token ID 列表"""
        ids = [self.char2idx.get(c, self.char2idx[self.UNK_TOKEN]) for c in text]
        if add_special_tokens:
            ids = [self.char2idx[self.SOS_TOKEN]] + ids + [self.char2idx[self.EOS_TOKEN]]
        return ids

    def decode(self, ids, skip_special: bool = True):
        """将 token ID 列表转回文本"""
        chars = []
        for idx in ids:
            ch = self.idx2char.get(idx, self.UNK_TOKEN)
            if skip_special and ch in self.special_tokens:
                continue
            chars.append(ch)
        return ''.join(chars)

    def get_pad_idx(self):
        return self.char2idx[self.PAD_TOKEN]

    def get_sos_idx(self):
        return self.char2idx[self.SOS_TOKEN]

    def get_eos_idx(self):
        return self.char2idx[self.EOS_TOKEN]


# ═══════════════════════════════════════════════════════════════════════════
#  2. 字符级语言模型 Dataset
# ═══════════════════════════════════════════════════════════════════════════

class CharLMDataset(Dataset):
    """
    字符级语言模型数据集。

    将文本切分为固定长度的片段，每个样本的输入是 tokens[0:seq_len]，
    目标是 tokens[1:seq_len+1] —— 即预测下一个字符。
    """

    def __init__(self, text: str, seq_len: int, tokenizer: CharTokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # 将整个文本编码为 token IDs
        self.data = tokenizer.encode(text, add_special_tokens=False)
        self.num_samples = max(0, len(self.data) - seq_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 输入: tokens[idx : idx+seq_len]
        # 目标: tokens[idx+1 : idx+seq_len+1] (右移一位)
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════
#  3. Noam 学习率调度器
# ═══════════════════════════════════════════════════════════════════════════

class NoamScheduler:
    """
    Transformer 论文提出的学习率调度策略:
      lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    原理: 前 warmup 步线性增长（避免训练初期不稳定），之后按 1/√step 衰减。

    这是理解学习率调度的重要实践——不是所有模型都适合固定的 lr。
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        step = self.step_num
        factor = self.d_model ** (-0.5)
        return factor * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))

    def get_current_lr(self):
        return self._get_lr()


# ═══════════════════════════════════════════════════════════════════════════
#  4. Decoder-Only Transformer (用于字符级语言模型)
# ═══════════════════════════════════════════════════════════════════════════

class DecoderOnlyTransformer(nn.Module):
    """
    将完整 Transformer 的 Decoder 部分用作自回归语言模型。

    架构: Embedding + PE → N × DecoderLayer → Linear → Softmax

    注意: 这里复用了 transformer.py 中的 DecoderLayer，
    实际上就是一个简化版的 GPT 架构。
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8,
                 d_ff: int = 1024, n_layers: int = 4, dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding (复用)
        from transformer import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 堆叠 DecoderLayer (实际上是带 causal mask 的 Self-Attention + FFN)
        from transformer import DecoderLayer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # 输出投影
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: (B, seq_len)
            mask: (B, 1, seq_len, seq_len)  causal mask
        Returns:
            logits: (B, seq_len, vocab_size)
        """
        # Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # 通过 Decoder Layers (cross-attention 部分传入 dummy)
        # 对于 Decoder-Only 模型，encoder output 不存在，传 None 处理
        dummy_enc = torch.zeros(x.size(0), 1, self.d_model, device=x.device)
        dummy_mask = None

        for layer in self.layers:
            # DecoderLayer 需要 enc_output，这里传入 dummy
            # self-attention (Q=K=V=x, mask=causal) + cross-attn (skipped via dummy) + FFN
            x, _, _ = layer(x, dummy_enc, dummy_mask, mask)

        return self.output_proj(x)


# ═══════════════════════════════════════════════════════════════════════════
#  5. 训练循环
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, pad_idx):
    """
    单个 epoch 的训练循环。

    重点理解:
    - model.train() vs model.eval() → 影响 Dropout 和 BN 的行为
    - optimizer.zero_grad() → 清零梯度，防止累积
    - loss.backward() → 自动反向传播计算梯度
    - optimizer.step() → 用梯度更新参数
    """
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        B, L = src.size()

        # 生成 causal mask (下三角)
        causal_mask = generate_lookahead_mask(L).to(device)

        # 前向传播
        logits = model(src, causal_mask)  # (B, L, vocab_size)

        # 计算损失: 交叉熵 (CrossEntropyLoss 自动做 softmax)
        # 将 logits 和 targets 展平为 2D
        loss = criterion(
            logits.view(-1, logits.size(-1)),  # (B*L, vocab_size)
            tgt.view(-1)                        # (B*L)
        )

        # 反向传播三步走
        optimizer.zero_grad()  # Step 1: 清零梯度
        loss.backward()        # Step 2: 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()       # Step 3: 更新参数

        if scheduler:
            scheduler.step()

        total_loss += loss.item()

        # 每 100 个 batch 打印一次进度
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            lr = scheduler.get_current_lr() if scheduler else optimizer.param_groups[0]['lr']
            print(f"  Batch {batch_idx:4d}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {lr:.6f} | "
                  f"Time: {elapsed:.1f}s")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """
    评估循环。
    torch.no_grad() 禁止梯度计算，节省显存、加速推理。
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            L = src.size(1)
            causal_mask = generate_lookahead_mask(L).to(device)

            logits = model(src, causal_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ═══════════════════════════════════════════════════════════════════════════
#  6. 文本生成 (自回归解码)
# ═══════════════════════════════════════════════════════════════════════════

def generate_text(model, tokenizer, prompt: str, max_len: int = 200,
                  temperature: float = 0.8, device=None):
    """
    自回归文本生成: 每次预测下一个字符，追加到序列末尾，循环。

    temperature 控制生成多样性:
      - temperature → 0: 确定性输出（总是选概率最高的）
      - temperature = 1: 按原始分布采样
      - temperature > 1: 增加多样性（概率分布变平）
    """
    model.eval()
    generated = []

    # 编码 prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            # 截取最后 max_len 个 token（防止序列过长）
            if input_tensor.size(1) > 512:
                input_tensor = input_tensor[:, -512:]

            L = input_tensor.size(1)
            causal_mask = generate_lookahead_mask(L).to(device)

            # 前向传播获取 logits
            logits = model(input_tensor, causal_mask)
            next_logits = logits[:, -1, :] / temperature  # 取最后一个位置的 logits

            # 采样或 greedy
            if temperature > 0:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            token_id = next_token.item()

            # 遇到 EOS 就停止
            if token_id == tokenizer.get_eos_idx():
                break

            generated.append(token_id)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

    return tokenizer.decode(generated, skip_special=True)


# ═══════════════════════════════════════════════════════════════════════════
#  7. 数据准备
# ═══════════════════════════════════════════════════════════════════════════

def get_sample_text():
    """
    示例训练数据: 一段中文小说文本。
    实际使用时替换为更大的语料库（如网络小说、维基百科 dump）。
    """
    return """
    第一章 初入江湖

    秋风萧瑟，落叶纷飞。长安城外的古道上，一个青衣少年背着长剑，缓缓而行。

    他叫林风，今年刚满十八岁。从小在山中跟随师父习武，如今师父说他已学有所成，
    该下山历练了。

    "江湖险恶，人心叵测。"师父临别时的话犹在耳边回响。

    林风摸了摸怀中的玉佩——那是师父交给他的信物，据说与他的身世有关。

    远处传来马蹄声，一队黑衣人疾驰而来。

    "站住！把玉佩交出来！"

    林风握紧了剑柄。他的江湖之旅，从这一刻真正开始了。

    第二章 剑出鞘

    黑衣人共有七个，个个手持长刀，面露凶光。

    "你们是什么人？"林风沉声问道。

    "少废话！"为首的黑衣人一挥手，"上！"

    刀光闪烁，杀气逼人。林风深吸一口气，长剑出鞘。

    叮叮当当——刀剑相击，火花四溅。

    林风的剑法飘逸灵动，一招"清风拂柳"挡开了三把刀，随即身形一转，
    "回风落雁"反手刺出，一名黑衣人闷哼倒地。

    但这些黑衣人显然训练有素，剩下的六人迅速变阵，将林风围在中间。

    "小兄弟剑法不错，"为首黑衣人冷笑道，"可惜今天就是你的死期！"

    就在这时，一道白影从天而降。

    "这么多人欺负一个少年，不觉得丢人吗？"

    那是一个白衣女子，手持玉笛，面若冰霜。
    """ * 10  # 复制 10 倍增加数据量


# ═══════════════════════════════════════════════════════════════════════════
#  8. 主函数
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="字符级 Transformer 语言模型")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "generate"],
                        help="运行模式: train 或 generate")
    parser.add_argument("--d_model", type=int, default=128,
                        help="模型隐藏维度")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="注意力头数")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="Decoder 层数")
    parser.add_argument("--d_ff", type=int, default=512,
                        help="FFN 内部维度")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="序列长度")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率 (Noam 模式下忽略)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 比例")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="模型保存路径")
    parser.add_argument("--prompt", type=str, default="江湖",
                        help="生成文本的起始 prompt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- (A) 构建 Tokenizer 和词表 ---
    print("\n[1/4] 构建词表...")
    text = get_sample_text()
    tokenizer = CharTokenizer()
    vocab_size = tokenizer.build_vocab([text])
    print(f"  Vocab size: {vocab_size}")

    # 保存词表信息供生成时使用
    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "train":
        # --- (B) 准备数据 ---
        print("[2/4] 准备数据...")
        # 训练集: 前 90%
        split_idx = int(len(text) * 0.9)
        train_text = text[:split_idx]
        val_text = text[split_idx:]

        train_dataset = CharLMDataset(train_text, args.seq_len, tokenizer)
        val_dataset = CharLMDataset(val_text, args.seq_len, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

        # --- (C) 创建模型 ---
        print(f"\n[3/4] 创建模型 (d_model={args.d_model}, "
              f"heads={args.n_heads}, layers={args.n_layers})...")
        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            dropout=args.dropout,
            max_len=args.seq_len
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        # --- (D) 损失函数 & 优化器 ---
        pad_idx = tokenizer.get_pad_idx()
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=4000)

        # --- (E) 训练循环 ---
        print(f"\n[4/4] 开始训练 ({args.epochs} epochs)...\n")
        best_val_loss = float('inf')

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            # 训练
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, pad_idx
            )
            # 验证
            val_loss = evaluate(model, val_loader, criterion, device)

            elapsed = time.time() - epoch_start
            print(f"\nEpoch {epoch:2d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {elapsed:.1f}s")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'vocab_size': vocab_size,
                    'config': {
                        'd_model': args.d_model,
                        'n_heads': args.n_heads,
                        'n_layers': args.n_layers,
                        'd_ff': args.d_ff,
                        'dropout': args.dropout,
                    }
                }
                torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
                print(f"  Saved best model (val_loss={val_loss:.4f})")

    elif args.mode == "generate":
        # 加载模型并生成文本
        checkpoint_path = os.path.join(args.save_dir, 'best_model.pt')
        if not os.path.exists(checkpoint_path):
            print(f"Error: 未找到模型文件 {checkpoint_path}，请先运行训练模式。")
            sys.exit(1)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']

        model = DecoderOnlyTransformer(
            vocab_size=checkpoint['vocab_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            n_layers=config['n_layers'],
            dropout=0.0,  # 生成时关闭 dropout
            max_len=args.seq_len
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})\n")
        print("=" * 50)
        print(f"Prompt: {args.prompt}")
        print("-" * 50)

        generated = generate_text(model, tokenizer, args.prompt, max_len=300, temperature=0.7, device=device)
        print(f"生成结果:\n{args.prompt}{generated}")
        print("=" * 50)


if __name__ == "__main__":
    main()
