"""扩展主题：知识蒸馏。

用 teacher 的软标签（高温 KL 散度）指导 student 学习，实现模型压缩。

运行：python ext-distillation/solution.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.gpt import GPTModel, GPT_CONFIG_124M  # noqa: E402


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """蒸馏损失：KL 散度 × T²。"""
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)


def main():
    print("=== 扩展主题：知识蒸馏 ===\n")

    # teacher (4层) → student (2层)
    teacher_cfg = dict(GPT_CONFIG_124M)
    teacher_cfg.update({"emb_dim": 128, "n_layers": 4, "n_heads": 4, "context_length": 32})
    student_cfg = dict(GPT_CONFIG_124M)
    student_cfg.update({"emb_dim": 128, "n_layers": 2, "n_heads": 4, "context_length": 32})

    torch.manual_seed(123)
    teacher = GPTModel(teacher_cfg)
    student = GPTModel(student_cfg)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    n_t = sum(p.numel() for p in teacher.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    print(f"[1] teacher: {n_t:,} 参数 | student: {n_s:,} 参数 (小 {100*(1-n_s/n_t):.0f}%)")

    # 蒸馏训练
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    print("\n[2] 蒸馏训练（student 模仿 teacher 输出分布，T=2.0）:")
    for epoch in range(8):
        x = torch.randint(0, teacher_cfg["vocab_size"], (4, 16))
        with torch.no_grad():
            t_logits = teacher(x)
        optimizer.zero_grad()
        s_logits = student(x)
        loss = distillation_loss(s_logits, t_logits, temperature=2.0)
        loss.backward()
        optimizer.step()
        print(f"    epoch {epoch}: KL loss {loss.item():.4f}")

    print(f"\n[done] student 用更少参数学到了接近 teacher 的输出分布。")


if __name__ == "__main__":
    main()
