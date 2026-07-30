"""第 5 章整理版：无监督预训练（训练循环 + 生成）。

完整复现主线流程：数据加载 → 训练循环（loss 下降）→ 贪婪生成。
OpenAI 权重加载见原书 5.5 节（需先下载权重文件到 models/）。

运行：python ch05-pretraining/solution.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import tiktoken  # noqa: E402

from src.gpt import (  # noqa: E402
    GPTModel,
    GPT_CONFIG_124M,
    create_dataloader_v1,
    generate_text_simple,
)


# ---------------------------------------------------------------------------
# 训练工具
# ---------------------------------------------------------------------------
def calc_loss_batch(input_batch, target_batch, model, device):
    """单个批次的下一步预测交叉熵损失。"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    # flatten 成 [batch*seq, vocab] 与 [batch*seq]，对齐做交叉熵
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """整个 loader 的平均损失。"""
    total = 0.0
    n = num_batches if num_batches is not None else len(data_loader)
    for i, (x, y) in enumerate(data_loader):
        if i >= n:
            break
        total += calc_loss_batch(x, y, model, device).item()
    return total / n


def train_model_simple(model, train_loader, optimizer, device, num_epochs,
                       eval_freq=5, eval_loader=None):
    """极简训练循环：每个 epoch 记录训练（和可选的验证）损失。"""
    model.to(device)
    losses = []
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()

        train_loss = calc_loss_loader(train_loader, model, device)
        line = f"Epoch {epoch + 1:2d}/{num_epochs} | train {train_loss:.4f}"
        if eval_loader is not None:
            model.eval()
            val_loss = calc_loss_loader(eval_loader, model, device)
            line += f" | val {val_loss:.4f}"
            losses.append((train_loss, val_loss))
        else:
            losses.append(train_loss)
        print(line)
    return losses


# ---------------------------------------------------------------------------
# 文本生成工具
# ---------------------------------------------------------------------------
def text_to_token_ids(text, tokenizer):
    return torch.tensor([tokenizer.encode(text)])


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None):
    """支持贪婪 / 温度采样 / top-k 的生成。"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = tiktoken.get_encoding("gpt2")

    # 1. 数据
    data_path = Path(__file__).resolve().parent.parent / "data" / "the-verdict.txt"
    with open(data_path, encoding="utf-8") as f:
        text = f.read()
    print(f"[1] 语料 {len(text):,} 字符")

    # 用小配置快速 demo（完整 124M 太慢）
    cfg = dict(GPT_CONFIG_124M)
    cfg.update({"emb_dim": 128, "n_layers": 2, "n_heads": 4, "context_length": 256})

    dl = create_dataloader_v1(text, batch_size=2, max_length=cfg["context_length"],
                              stride=cfg["context_length"], shuffle=True, drop_last=True)
    print(f"[2] {len(dl)} 个批次")

    # 2. 训练
    torch.manual_seed(123)
    model = GPTModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    print("[3] 开始训练（10 epoch）...")
    losses = train_model_simple(model, dl, optimizer, device, num_epochs=10)

    # 3. 生成
    start = "I had a little"
    start_ids = text_to_token_ids(start, tok).to(device)
    model.eval()
    with torch.no_grad():
        out = generate(model, start_ids, max_new_tokens=30,
                       context_size=cfg["context_length"], temperature=0.8, top_k=10)
    print(f"\n[4] 生成（temperature=0.8, top_k=10）：")
    print("   ", repr(token_ids_to_text(out, tok)))

    print("\n[done] 注：demo 模型仅验证流程；生成高质量文本需加载 OpenAI 预训练权重（见原书 5.5 节）。")


if __name__ == "__main__":
    main()
