"""第 2 章整理版代码：端到端演示「文本 → 训练批次」。

运行：python ch02-text-data/solution.py
"""

from pathlib import Path
import sys

# 让本脚本能从仓库根目录导入 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.gpt.data import create_dataloader_v1  # noqa: E402


def main():
    # 1. 读取语料
    data_path = Path(__file__).resolve().parent.parent / "data" / "the-verdict.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"[1/3] 已读取语料，共 {len(raw_text)} 字符")

    # 2. 构建 DataLoader
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    print(f"[2/3] 数据集共 {len(dataloader.dataset)} 个样本，{len(dataloader)} 个批次")

    # 3. 取出第一个批次展示
    inputs, targets = next(iter(dataloader))
    print(f"[3/3] 第一个批次:")
    print(f"  inputs  形状: {inputs.shape}, dtype: {inputs.dtype}")
    print(f"  targets 形状: {targets.shape}, dtype: {targets.dtype}")
    print(f"  inputs[0]:  {inputs[0].tolist()}")
    print(f"  targets[0]: {targets[0].tolist()}")


if __name__ == "__main__":
    main()
