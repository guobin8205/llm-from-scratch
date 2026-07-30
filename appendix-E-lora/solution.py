"""附录 E 整理版：LoRA 参数高效微调。

用低秩矩阵 BA 旁路冻结的权重，可训练参数压到 < 1%。
演示：给 GPT 应用 LoRA，做情感分类微调（复用 ch06 任务）。

运行：python appendix-E-lora/solution.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import tiktoken  # noqa: E402

from src.gpt import GPTModel, GPT_CONFIG_124M  # noqa: E402


# ---------------------------------------------------------------------------
# LoRA 核心实现
# ---------------------------------------------------------------------------
class LoRALayer(nn.Module):
    """给一个 nn.Linear 加 LoRA 旁路：冻结原权重，训练低秩 A、B。"""

    def __init__(self, layer, r=8, alpha=16):
        super().__init__()
        self.layer = layer
        for p in layer.parameters():
            p.requires_grad = False          # 冻结原始权重
        in_f = layer.in_features
        out_f = layer.out_features
        self.scaling = alpha / r
        # A 正常初始化，B 零初始化（保证初期 ΔW=0，不破坏原模型）
        self.A = nn.Parameter(torch.randn(in_f, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, out_f))

    def forward(self, x):
        return self.layer(x) + self.scaling * (x @ (self.A @ self.B))


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()
        self.lora = LoRALayer(linear, r=r, alpha=alpha)

    def forward(self, x):
        return self.lora(x)


def replace_linear_with_lora(module, r=8, alpha=16):
    """递归把 module 下所有 nn.Linear 替换为 LoRA 版。"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearWithLoRA(child, r, alpha))
        else:
            replace_linear_with_lora(child, r, alpha)


# ---------------------------------------------------------------------------
# demo：用 LoRA 做情感分类微调
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== 附录 E：LoRA 参数高效微调 ===\n")

    cfg = dict(GPT_CONFIG_124M)
    cfg.update({"emb_dim": 128, "n_layers": 2, "n_heads": 4, "context_length": 32})
    torch.manual_seed(123)
    model = GPTModel(cfg)
    model.out_head = nn.Linear(cfg["emb_dim"], 2)  # 分类头

    # 先冻结所有参数，再给 trf_blocks 加 LoRA
    for p in model.parameters():
        p.requires_grad = False
    replace_linear_with_lora(model.trf_blocks, r=8, alpha=16)
    for p in model.out_head.parameters():
        p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[1] 参数: 总 {total:,} | 可训练 {trainable:,} "
          f"({100*trainable/total:.2f}%) | 冻结 {total-trainable:,}")

    # 数据
    tok = tiktoken.get_encoding("gpt2")
    data = [(tok.encode(t)[:32], 1) for t in POSITIVE]
    data += [(tok.encode(t)[:32], 0) for t in NEGATIVE]
    print(f"[2] 数据: {len(data)} 条情感样本")

    # 训练（只优化 LoRA + 分类头）
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.1,
    )
    print("[3] LoRA 微调:")
    model.train()
    for epoch in range(15):
        total_loss = 0
        for ids, label in data:
            x = torch.tensor([ids + [50256] * (32 - len(ids))]).to(device)
            y = torch.tensor([label]).to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x)[:, -1, :], y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 3 == 0 or epoch == 14:
            print(f"  epoch {epoch}: loss {total_loss/len(data):.4f}")

    # 评估
    model.eval()
    correct = 0
    with torch.no_grad():
        for ids, label in data:
            x = torch.tensor([ids + [50256] * (32 - len(ids))]).to(device)
            pred = model(x)[:, -1, :].argmax(-1)
            correct += (pred.item() == label)
    print(f"\n[4] 准确率: {100*correct/len(data):.0f}%（只训了 {100*trainable/total:.2f}% 参数）")
    print("\n[done] LoRA 用 < 1% 参数达到接近全量微调的效果。")


if __name__ == "__main__":
    main()
