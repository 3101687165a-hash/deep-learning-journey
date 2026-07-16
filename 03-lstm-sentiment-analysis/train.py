"""
LSTM 情感分析 — IMDB 电影评论训练脚本
=======================================

运行方式:
    python train.py --rnn_type lstm
    python train.py --rnn_type gru
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from model import SentimentClassifier


# ═══════════════════════════════════════════════════════════════════════════
#  1. 简易分词器 & 词表构建
# ═══════════════════════════════════════════════════════════════════════════

class SimpleTokenizer:
    """简易英文分词器（按空格分词 + 基础清洗）"""

    def __init__(self, min_freq: int = 2, max_vocab: int = 25000):
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.PAD, self.UNK = "<pad>", "<unk>"
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = {0: self.PAD, 1: self.UNK}

    def build_vocab(self, texts):
        """从文本列表构建词表（按词频排序）"""
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        # 取 top max_vocab 个高频词
        for word, freq in counter.most_common(self.max_vocab):
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        return len(self.word2idx)

    def _tokenize(self, text: str):
        """简单清洗 + 小写 + 空格分词"""
        text = text.lower()
        # 保留字母和基本标点
        text = ''.join(c if c.isalpha() or c in " '.,!?" else ' ' for c in text)
        return text.split()

    def encode(self, text: str, max_len: int = None):
        """将文本转换为 token ID 列表"""
        tokens = self._tokenize(text)
        ids = [self.word2idx.get(w, self.word2idx[self.UNK]) for w in tokens]
        if max_len:
            ids = ids[:max_len]  # 截断
        return ids

    @property
    def vocab_size(self):
        return len(self.word2idx)

    @property
    def pad_idx(self):
        return 0


# ═══════════════════════════════════════════════════════════════════════════
#  2. Dataset
# ═══════════════════════════════════════════════════════════════════════════

class SentimentDataset(Dataset):
    """
    情感分析数据集。

    内部使用文本数据，可以换成 torchtext 的 IMDB 或自定义数据。
    这里提供了一个内置的小样本数据集用于演示。
    """

    def __init__(self, texts, labels, tokenizer, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.texts[idx], self.max_len)
        length = len(ids)
        # Padding 到 max_len
        padded = ids + [self.tokenizer.pad_idx] * (self.max_len - len(ids))
        return torch.tensor(padded), torch.tensor(length), torch.tensor(self.labels[idx])


def collate_fn(batch):
    """自定义 collate: 按长度排序（pack_padded 要求降序）"""
    batch.sort(key=lambda x: x[1], reverse=True)
    texts, lengths, labels = zip(*batch)
    return torch.stack(texts), torch.tensor(lengths), torch.tensor(labels)


# ═══════════════════════════════════════════════════════════════════════════
#  3. 示例数据（实际使用时替换为 IMDB）
# ═══════════════════════════════════════════════════════════════════════════

def get_sample_data():
    """用于快速验证的小样本数据。实际训练请用 torchtext 的 IMDB。"""
    positive = [
        "This movie was absolutely wonderful and fantastic great acting",
        "I loved every minute of this film it was brilliant and amazing",
        "An incredible masterpiece with outstanding performances all around",
        "The best movie I have seen this year highly recommended",
        "Brilliant storytelling and fantastic cinematography truly a gem",
        "A heartwarming and beautiful film that touched my soul deeply",
        "Exceptional direction and powerful acting made this unforgettable",
        "A delightful and charming movie that made me smile throughout",
    ]
    negative = [
        "This was a terrible waste of time boring and completely awful",
        "I hated this movie it was so bad I wanted to leave early",
        "The worst film I have ever seen absolutely dreadful acting",
        "A complete disaster of a movie nothing redeeming about it",
        "Poorly written badly directed and a total waste of money",
        "An absolute mess of a film confusing and painfully boring",
        "The acting was wooden and the plot made no sense at all",
        "One of the most disappointing and terrible movies ever made",
    ]

    texts = positive * 100 + negative * 100  # 扩充到 800 + 800
    labels = [1] * len(positive) * 100 + [0] * len(negative) * 100
    return texts, labels


# ═══════════════════════════════════════════════════════════════════════════
#  4. 训练 & 评估
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for texts, lengths, labels in loader:
        texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(texts, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * texts.size(0)
        _, preds = logits.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for texts, lengths, labels in loader:
        texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)
        logits = model(texts, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * texts.size(0)
        _, preds = logits.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return total_loss / total, 100. * correct / total


# ═══════════════════════════════════════════════════════════════════════════
#  5. 主函数
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_type", default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 数据 ---
    print("Preparing data...")
    texts, labels = get_sample_data()

    # 简单划分
    split = int(len(texts) * 0.8)
    train_texts, val_texts = texts[:split], texts[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    # 构建词表
    tokenizer = SimpleTokenizer(min_freq=1, max_vocab=5000)
    vocab_size = tokenizer.build_vocab(train_texts)
    print(f"  Vocab size: {vocab_size}")

    # 创建 Dataset / DataLoader
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, args.max_len)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # --- 模型 ---
    model = SentimentClassifier(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
        bidirectional=args.bidirectional
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.rnn_type.upper()} ({n_params:,} params)")

    # --- 训练 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>12} {'Val Acc':>10} {'Time':>8}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        print(f"{epoch:6d} {train_loss:12.4f} {train_acc:9.2f}% {val_loss:12.4f} {val_acc:9.2f}% {elapsed:7.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.rnn_type}_best.pth"))

    print(f"{'='*60}")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
