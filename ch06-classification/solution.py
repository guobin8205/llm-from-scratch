"""第 6 章整理版：文本分类微调。

把 GPT 续写模型改造成情感分类器：
1. 替换 out_head 为分类头（emb_dim → num_classes）
2. 冻结 backbone，只训练最后块 + final_norm + 分类头
3. 用分类交叉熵微调 + 评估准确率

主线用自造中文情感 demo 数据验证流程；真实场景用 SST-2/IMDb（见 bonus 02）。

运行：python ch06-classification/solution.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402
import tiktoken  # noqa: E402

from src.gpt import GPTModel, GPT_CONFIG_124M  # noqa: E402


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
POSITIVE = [
    "这部电影非常精彩 我很喜欢", "太好看了 剧情感人至深",
    "画面优美 值得推荐", "演技出色 故事动人",
    "完美之作 强烈推荐", "音乐动听 视觉震撼",
    "节奏紧凑 引人入胜", "结局温暖 回味无穷",
]
NEGATIVE = [
    "太糟糕了 浪费时间", "剧情无聊 让人失望",
    "画面粗糙 毫无诚意", "演技尴尬 故事混乱",
    "简直烂片 不忍直视", "噪音刺耳 看不下去",
    "节奏拖沓 昏昏欲睡", "结局糟糕 一无是处",
]


class SentimentDataset(Dataset):
    """情感分类数据集：文本 → 定长 token 序列 + 标签。"""

    def __init__(self, texts, labels, tokenizer, max_len=32, pad_id=50256):
        self.max_len = max_len
        self.pad_id = pad_id
        self.data = [(tokenizer.encode(t)[:max_len], l)
                     for t, l in zip(texts, labels)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids, label = self.data[i]
        ids = ids + [self.pad_id] * (self.max_len - len(ids))  # 右侧 pad
        return torch.tensor(ids), torch.tensor(label)


# ---------------------------------------------------------------------------
# 模型改造
# ---------------------------------------------------------------------------
def build_classifier(cfg, num_classes=2, seed=123):
    """构造 GPT 分类器：替换输出头 + 冻结 backbone。"""
    torch.manual_seed(seed)
    model = GPTModel(cfg)
    # 核心改造：续写头 → 分类头
    model.out_head = nn.Linear(cfg["emb_dim"], num_classes)
    # 冻结除「最后块 + final_norm + 分类头」外的所有参数
    for p in model.parameters():
        p.requires_grad = False
    for p in model.trf_blocks[-1].parameters():
        p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True
    for p in model.out_head.parameters():
        p.requires_grad = True
    return model


def classify(model, x):
    """取最后一个 token 的输出做分类。"""
    return model(x)[:, -1, :]   # [b, num_classes]


# ---------------------------------------------------------------------------
# 训练与评估
# ---------------------------------------------------------------------------
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = classify(model, input_batch)
    return F.cross_entropy(logits, target_batch)


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total = 0.0
    n = num_batches if num_batches is not None else len(data_loader)
    for i, (x, y) in enumerate(data_loader):
        if i >= n:
            break
        total += calc_loss_batch(x, y, model, device).item()
    return total / n


def calc_accuracy_loader(data_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = classify(model, x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def evaluate_model(model, train_loader, val_loader, test_loader, device):
    """返回 train/val/test 的 (loss, accuracy)。"""
    model.eval()
    result = {}
    for name, loader in [("train", train_loader), ("val", val_loader),
                         ("test", test_loader)]:
        if loader is None:
            continue
        loss = calc_loss_loader(loader, model, device)
        acc = calc_accuracy_loader(loader, model, device)
        result[name] = (loss, acc)
        print(f"  {name:5s}: loss {loss:.4f} | acc {100*acc:.1f}%")
    return result


def train_classifier(model, train_loader, optimizer, device, num_epochs,
                     val_loader=None):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total = 0
        n = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        line = f"Epoch {epoch+1:2d}/{num_epochs} | train loss {total/n:.4f}"
        if val_loader is not None and (epoch % 3 == 0 or epoch == num_epochs - 1):
            val_acc = calc_accuracy_loader(val_loader, model, device)
            line += f" | val acc {100*val_acc:.1f}%"
        print(line)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = tiktoken.get_encoding("gpt2")

    # 1. 数据（自造 demo，真实场景用 SST-2/IMDb）
    texts = POSITIVE + NEGATIVE
    labels = [1] * len(POSITIVE) + [0] * len(NEGATIVE)
    print(f"[1] 样本: {len(texts)} (正{len(POSITIVE)}/负{len(NEGATIVE)})")

    # 划分 train/val/test（小数据简单切分）
    import random
    random.seed(42)
    idx = list(range(len(texts)))
    random.shuffle(idx)
    n_train, n_val = 10, 3
    splits = {
        "train": (idx[:n_train], True),
        "val": (idx[n_train:n_train+n_val], False),
        "test": (idx[n_train+n_val:], False),
    }
    loaders = {}
    for name, (ids, shuffle) in splits.items():
        ds = SentimentDataset([texts[i] for i in ids], [labels[i] for i in ids], tok)
        loaders[name] = DataLoader(ds, batch_size=4, shuffle=shuffle)

    # 2. 模型
    cfg = dict(GPT_CONFIG_124M)
    cfg.update({"emb_dim": 128, "n_layers": 2, "n_heads": 4, "context_length": 32})
    model = build_classifier(cfg, num_classes=2)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[2] 参数: {total:,} (可训练 {trainable:,}, {100*trainable/total:.1f}%)")

    # 3. 训练
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4, weight_decay=0.1,
    )
    print("[3] 训练...")
    train_classifier(model, loaders["train"], optimizer, device,
                     num_epochs=15, val_loader=loaders.get("val"))

    # 4. 评估
    print("[4] 评估:")
    evaluate_model(model, loaders["train"], loaders.get("val"),
                   loaders.get("test"), device)


if __name__ == "__main__":
    main()
