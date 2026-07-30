"""《从零构建大模型》核心模块包。

随着主线章节推进，本包逐步累积可复用的核心组件：
- ch02: DataLoader（滑动窗口数据加载）
- ch03: MultiHeadAttention（多头注意力）
- ch04: GPT（完整模型）、LayerNorm、GELU、TransformerBlock
- ch05: 训练循环、文本生成
- ch06: 分类头
- ch07: 指令微调

组件成熟后从各章 solution.py 提取至此，供后续章节复用，避免重复代码。
"""

from src.gpt.data import GPTDatasetV1, create_dataloader_v1

__all__ = ["GPTDatasetV1", "create_dataloader_v1"]
