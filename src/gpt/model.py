"""第 4 章核心模型模块：GPT-2 124M 完整实现。

供 ch05（预训练）+ ch06/ch07（微调）复用。
组件：LayerNorm / GELU / FeedForward / TransformerBlock / GPTModel / generate_text_simple。
"""

import math
import torch
import torch.nn as nn

from src.gpt.attention import MultiHeadAttention


# GPT-2 124M 配置（与 OpenAI 原版一致）
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词表大小（GPT-2 的 BPE）
    "context_length": 1024,  # 最大上下文长度
    "emb_dim": 768,          # embedding 维度
    "n_heads": 12,           # 注意力头数
    "n_layers": 12,          # Transformer 层数
    "drop_rate": 0.1,        # dropout 比例
    "qkv_bias": False,       # Q/K/V 是否带偏置
}


class LayerNorm(nn.Module):
    """层归一化：把每个样本的特征归一化到均值0/方差1，再用可训练 scale/shift 还原。

    作用：稳定训练，缓解梯度爆炸/消失，让深层网络可训练。
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可训练缩放
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可训练偏移

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU 激活函数（tanh 近似版，GPT-2 使用）。

    相比 ReLU，GELU 处处可微、负半轴有小幅非零值，梯度更平滑。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """前馈网络：Linear(扩4倍) → GELU → Linear(压回)。

    隐藏维扩展为 4×emb_dim，让模型有更大容量做非线性变换。
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 扩展 4 倍
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 压回原维
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """单个 Transformer 块（pre-LN 结构）。

    注意力子层：LayerNorm → MultiHeadAttention → Dropout → 残差
    前馈子层：  LayerNorm → FeedForward → Dropout → 残差
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力子层（pre-LN：先归一化再进子层，残差用归一化前的 x）
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈子层
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    """GPT-2 风格的 decoder-only Transformer。

    结构：token嵌入 + 位置嵌入 → dropout → n_layers×TransformerBlock → 最终LayerNorm → 线性输出头
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)   # [batch, seq, vocab_size]


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """贪婪自回归文本生成。

    Args:
        model: GPT 模型
        idx: 当前上下文 [batch, seq]
        max_new_tokens: 要生成的新 token 数
        context_size: 模型上下文长度（超过则裁剪）

    Returns:
        生成后的 [batch, seq + max_new_tokens]
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]   # 裁剪到上下文长度
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]           # 只取最后一个 token 的 logits
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # 贪婪：取最大概率
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
