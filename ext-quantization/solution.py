"""扩展主题：模型量化。

int8/int4 对称量化 + 量化 GPT + QLoRA 概念演示。

运行：python ext-quantization/solution.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import tiktoken  # noqa: E402

from src.gpt import GPTModel, GPT_CONFIG_124M  # noqa: E402


def quantize_int8(weight):
    """对称量化：fp32 → int8。"""
    max_val = weight.abs().max()
    scale = max_val / 127.0
    q_weight = torch.round(weight / scale).to(torch.int8)
    return q_weight, scale


def quantize_int4(weight):
    """对称量化：fp32 → int4（范围 [-8,7]，用 int8 存储）。"""
    max_val = weight.abs().max()
    scale = max_val / 7.0
    q_weight = torch.round(weight / scale).clamp(-8, 7).to(torch.int8)
    return q_weight, scale


def dequantize(q_weight, scale):
    return q_weight.float() * scale


def quantize_model_int8(model):
    """量化模型所有 Linear 权重。"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            q_w, scale = quantize_int8(module.weight.data)
            module.weight.data = dequantize(q_w, scale)


def main():
    print("=== 扩展主题：模型量化 ===\n")

    # 1. 量化误差对比
    torch.manual_seed(0)
    W = torch.randn(128, 128) * 0.1
    q8, s8 = quantize_int8(W)
    q4, s4 = quantize_int4(W)
    err8 = (W - dequantize(q8, s8)).abs().mean()
    err4 = (W - dequantize(q4, s4)).abs().mean()
    print("[1] 量化误差对比:")
    print(f"    int8: {err8.item():.6f} (省 75% 显存)")
    print(f"    int4: {err4.item():.6f} (省 87.5% 显存)")

    # 2. 量化 GPT 并对比输出
    cfg = dict(GPT_CONFIG_124M)
    cfg.update({"emb_dim": 128, "n_layers": 2, "n_heads": 4, "context_length": 32})
    torch.manual_seed(123)
    model_fp32 = GPTModel(cfg)
    model_int8 = GPTModel(cfg)
    quantize_model_int8(model_int8)

    tok = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([tok.encode("Hello")])
    model_fp32.eval(); model_int8.eval()
    with torch.no_grad():
        diff = (model_fp32(idx) - model_int8(idx)).abs().mean().item()
    print(f"\n[2] 量化前后 GPT 输出差异(MAE): {diff:.6f}（应很小）")

    # 3. QLoRA 概念
    print("\n[3] QLoRA = 4bit量化基座 + LoRA旁路")
    print("    量化省基座显存，LoRA 只训极小参数 → 单卡微调大模型")
    print("    (详见附录 E 的 LoRA 实现)")
    print("\n[done] 真实工程用 bitsandbytes/GPTQ/AWQ 做生产级量化。")


if __name__ == "__main__":
    main()
