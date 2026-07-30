"""第 7 章整理版：指令微调（Instruction Fine-tuning）。

让 GPT 学会遵循指令：
1. Alpaca 模板格式化指令数据（Instruction / Input / Response）
2. loss masking：只对 Response 算 loss，Instruction 部分置 -100
3. 冻结 backbone，在指令数据上微调

主线用自造中文指令 demo 数据验证流程；真实场景用 Alpaca/OASST 等数据集。

运行：python ch07-instruction/solution.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
import tiktoken  # noqa: E402

from src.gpt import GPTModel, GPT_CONFIG_124M  # noqa: E402


# ---------------------------------------------------------------------------
# 指令数据（自造 demo）
# ---------------------------------------------------------------------------
INSTRUCTION_DATA = [
    {"instruction": "识别情感", "input": "今天天气真好", "output": " 正面"},
    {"instruction": "识别情感", "input": "太让人失望了", "output": " 负面"},
    {"instruction": "识别情感", "input": "电影非常精彩", "output": " 正面"},
    {"instruction": "识别情感", "input": "服务态度很差", "output": " 负面"},
    {"instruction": "翻译成英文", "input": "你好", "output": " Hello"},
    {"instruction": "翻译成英文", "input": "谢谢", "output": " Thank you"},
    {"instruction": "翻译成英文", "input": "再见", "output": " Goodbye"},
    {"instruction": "回答问题", "input": "法国首都是哪", "output": " 巴黎"},
    {"instruction": "回答问题", "input": "地球绕着什么转", "output": " 太阳"},
    {"instruction": "回答问题", "input": "一年有几个月", "output": " 十二"},
]


# ---------------------------------------------------------------------------
# Alpaca 格式化与 loss masking
# ---------------------------------------------------------------------------
def format_prompt(entry):
    """Alpaca 模板：把指令和输入拼成 prompt。"""
    return (
        f"### Instruction:\n{entry['instruction']}\n"
        f"### Input:\n{entry['input']}\n"
        f"### Response:\n"
    )


def custom_collate(batch_entries, tokenizer, max_len=64, pad_id=50256,
                   ignore_index=-100):
    """整理批次：拼 prompt+response，构造 loss mask。

    关键：targets 中 prompt 对应位置置 ignore_index，只对 response 算 loss。
    """
    batch_inputs, batch_targets = [], []
    for entry in batch_entries:
        prompt_ids = tokenizer.encode(format_prompt(entry))
        resp_ids = tokenizer.encode(entry["output"])
        input_ids = prompt_ids + resp_ids
        # targets = input_ids 右移一位（预测下一个 token）
        targets = input_ids[1:] + [tokenizer.eot_token]
        # 截断 + mask prompt 部分
        input_ids = input_ids[:max_len]
        targets = targets[:max_len]
        for i in range(min(len(prompt_ids), len(targets))):
            targets[i] = ignore_index
        # pad
        input_ids = input_ids + [pad_id] * (max_len - len(input_ids))
        targets = targets + [ignore_index] * (max_len - len(targets))
        batch_inputs.append(input_ids)
        batch_targets.append(targets)
    return torch.tensor(batch_inputs), torch.tensor(batch_targets)


# ---------------------------------------------------------------------------
# 训练与生成
# ---------------------------------------------------------------------------
def build_model(cfg, seed=123):
    """构造模型并冻结 backbone（只训最后块 + norm + 输出头）。"""
    torch.manual_seed(seed)
    model = GPTModel(cfg)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.trf_blocks[-1].parameters():
        p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True
    for p in model.out_head.parameters():
        p.requires_grad = True
    return model


def train(model, dataloader, optimizer, device, num_epochs):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total = 0
        n = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                   ignore_index=-100)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:2d}/{num_epochs} | loss {total/n:.4f}")


def generate_response(model, entry, tokenizer, context_size, max_new_tokens=10,
                      device="cpu"):
    """给定指令，让模型生成 Response。"""
    model.eval()
    prompt = format_prompt(entry)
    idx = torch.tensor([tokenizer.encode(prompt)]).to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == tokenizer.eot_token:
                break
    generated = tokenizer.decode(idx[0].tolist())
    return generated.split("### Response:\n")[-1].strip()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = tiktoken.get_encoding("gpt2")

    cfg = dict(GPT_CONFIG_124M)
    cfg.update({"emb_dim": 128, "n_layers": 2, "n_heads": 4, "context_length": 64})
    print(f"[1] 指令数据: {len(INSTRUCTION_DATA)} 条")

    model = build_model(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[2] 参数: {total:,} (可训练 {trainable:,}, {100*trainable/total:.1f}%)")

    # 数据乘 3 倍增加样本量
    data = INSTRUCTION_DATA * 3
    dataloader = DataLoader(
        data, batch_size=4, shuffle=True,
        collate_fn=lambda b: custom_collate(b, tok, max_len=cfg["context_length"]),
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4, weight_decay=0.1,
    )
    print("[3] 训练...")
    train(model, dataloader, optimizer, device, num_epochs=10)

    print("[4] 生成测试:")
    test_cases = [
        {"instruction": "识别情感", "input": "今天天气真好"},
        {"instruction": "翻译成英文", "input": "你好"},
        {"instruction": "回答问题", "input": "法国首都是哪"},
    ]
    for entry in test_cases:
        resp = generate_response(model, entry, tok, cfg["context_length"], device=device)
        print(f"  [{entry['instruction']}] {entry['input']!r} → {resp!r}")
    print("\n[done] demo 数据量小、未加载预训练权重，输出仅供参考验证流程。")


if __name__ == "__main__":
    main()
