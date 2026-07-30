"""第 4 章整理版：GPT-2 124M。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.gpt import GPTModel, GPT_CONFIG_124M, generate_text_simple


def main():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    print(f"[1] 参数量: {sum(p.numel() for p in model.parameters()):,}")
    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([tok.encode("Hello, I am")])
    print("[2] 未训练:", tok.decode(generate_text_simple(model, idx, 10, 1024)[0].tolist()))


if __name__ == "__main__":
    main()
