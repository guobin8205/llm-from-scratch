"""第 2 章核心数据模块：把原始文本转化为 GPT 训练用的 token 张量批次。

流水线：原始文本 → BPE 编码 → 滑动窗口生成 (input, target) 对 → PyTorch Dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """自回归语言模型的训练数据集。

    对一段已编码的 token 序列施加滑动窗口，每个样本是一对
    (input_chunk, target_chunk)，target 比 input 右移一位。
    """

    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        """初始化数据集。

        Args:
            txt: 原始文本。
            tokenizer: 带 encode 方法的分词器（如 tiktoken）。
            max_length: 每个样本的 token 数（上下文长度）。
            stride: 滑动步长。通常等于 max_length 以避免样本重叠；
                    设为 1 可最大化样本数但会高度重叠。
        """
        self.input_ids = []
        self.target_ids = []

        # 整篇文本一次性编码成 token ID
        token_ids = tokenizer.encode(txt)

        # 滑动窗口切分（确保最后一段不满 max_length 的被丢弃）
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
):
    """从原始文本创建一个 Yield (input_batch, target_batch) 的 DataLoader。

    Args:
        txt: 原始文本。
        batch_size: 每个批次的样本数。
        max_length: 每个样本的 token 数。
        stride: 滑动步长。
        shuffle: 是否打乱。
        drop_last: 是否丢弃最后不满一个 batch 的样本。
        num_workers: DataLoader 工作进程数。

    Returns:
        配置好的 torch.utils.data.DataLoader。
    """
    # 延迟导入，避免在没装 tiktoken 的环境里 import 本模块就报错
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
