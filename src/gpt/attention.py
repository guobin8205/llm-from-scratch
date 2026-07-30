"""第 3 章核心注意力模块：多头因果注意力。

供 ch04 GPT 模型复用。标准权重分割实现，支持批处理与因果掩码。
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """多头因果注意力（权重分割实现）。

    一个大矩阵切分成 num_heads 份，一次矩阵乘算完所有头，效率高于堆叠版。
    带因果掩码（上三角置 -∞）保证只关注左侧 token，适配自回归生成。

    Args:
        d_in: 输入 embedding 维度。
        d_out: 输出维度（必须能被 num_heads 整除）。
        context_length: 最大上下文长度（用于预分配掩码 buffer）。
        dropout: dropout 比例。
        num_heads: 注意力头数。
        qkv_bias: Q/K/V 投影是否带偏置。
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out 必须能被 num_heads 整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 输出投影，融合多头
        self.dropout = nn.Dropout(dropout)

        # 因果掩码：上三角（不含对角线）为 True，用于把未来 token 的注意力分数置 -∞
        # register_buffer：随模型搬设备（.to('cuda')），但不参与训练（不在 parameters() 里）
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
        )

    def forward(self, x):
        """前向传播。

        Args:
            x: [batch, seq, d_in]

        Returns:
            [batch, seq, d_out]
        """
        b, num_tokens, _ = x.shape

        # 投影后重塑成 [b, num_heads, seq, head_dim]，便于并行算每个头
        keys = (
            self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        )
        queries = (
            self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        )
        values = (
            self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        )

        # [b, heads, seq, seq]
        attn_scores = queries @ keys.transpose(2, 3)
        # 用掩码把上三角（未来 token）置 -∞
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 缩放点积注意力
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [b, heads, seq, head_dim] → [b, seq, d_out]
        context_vec = (
            (attn_weights @ values)
            .transpose(1, 2)
            .contiguous()
            .view(b, num_tokens, self.d_out)
        )
        return self.out_proj(context_vec)
