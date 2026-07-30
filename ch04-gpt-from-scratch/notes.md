# 第 04 章 · 从零搭建 GPT

> **状态**：`[x]` 已完成 ｜ `2026-07-31`
>
> **对应原书**：Chapter 04 — Implementing a GPT Model from Scratch

---

## 本章目标

把 ch03 的注意力模块，配上 LayerNorm、前馈网络、残差连接，拼装成完整的 **GPT-2 124M** 模型。完成后能初始化模型、做一次前向、生成（未训练的）文本。

## 核心内容

### GPT 的积木块（自底向上）

```
GPTModel
  ├── token embedding + position embedding
  ├── dropout
  ├── n_layers × TransformerBlock
  │     ├── 注意力子层：LayerNorm → MultiHeadAttention → Dropout → 残差
  │     └── 前馈子层：  LayerNorm → FeedForward        → Dropout → 残差
  ├── final LayerNorm
  └── Linear 输出头（→ vocab_size）
```

### 各组件的作用

| 组件 | 解决什么问题 | 要点 |
|------|------------|------|
| **LayerNorm** | 训练不稳定 | 归一化特征到均值0/方差1，再学 scale/shift |
| **GELU** | ReLU 的平滑替代 | 处处可微，负半轴有小幅非零值，梯度更平滑 |
| **FeedForward** | 非线性变换容量 | Linear(扩4倍)→GELU→Linear(压回) |
| **残差连接** | 深层梯度消失 | `x + sublayer(x)`，让梯度直通 |
| **pre-LN** | 训练稳定性 | 先归一化再进子层，残差用归一化前的 x |
| **位置嵌入** | 注入顺序信息 | GPT 用学习的绝对位置嵌入（Llama 改用 RoPE） |

### GPT-2 124M 配置

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # GPT-2 BPE 词表
    "context_length": 1024,  # 最大上下文
    "emb_dim": 768,          # embedding 维度
    "n_heads": 12,           # 注意力头数
    "n_layers": 12,          # Transformer 层数
    "drop_rate": 0.1,        # dropout
    "qkv_bias": False,       # Q/K/V 无偏置
}
```

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 层归一化 | LayerNorm | 特征维度归一化 |
| GELU | GELU | GPT-2 用的激活函数（tanh 近似） |
| 前馈网络 | FeedForward (FFN) | 两层 Linear + 激活 |
| 残差连接 | Residual connection | x + sublayer(x) |
| Transformer 块 | TransformerBlock | 注意力+FFN 两个子层 |
| pre-LN | Pre-LayerNorm | 先归一化再进子层 |
| 词嵌入 | Token embedding | token ID → 向量 |
| 位置嵌入 | Position embedding | 注入位置信息 |
| 输出头 | Output head | 映射回 vocab 的 Linear |
| 自回归生成 | Autoregressive generation | 逐 token 预测下一个 |

## 代码走读

- `ch04.ipynb` — 初始化 GPT 124M，打印参数量
- `solution.py` — 整理版：初始化 + 未训练生成 demo
- `src/gpt/model.py` — 完整模型实现（LayerNorm/GELU/FeedForward/TransformerBlock/GPTModel/generate_text_simple）
- `bonus/` — ⭐8 种注意力变体（KV缓存/GQA/MLA/SWA/MoE/DeltaNet/DSA/KV共享）

### 核心实现（src/gpt/model.py）

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(n_layers)])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, in_idx):
        x = self.tok_emb(in_idx) + self.pos_emb(arange(seq_len))  # token + 位置
        return self.out_head(self.final_norm(self.trf_blocks(x)))  # → [b, seq, vocab]
```

### `generate_text_simple`（贪婪生成）

每次取最后一个 token 的 logits → softmax → argmax → 拼接到末尾 → 循环。

## 运行结果

`solution.py` 输出：
```
[1] 参数量: 163,009,536        # 注意：未做权重绑定为 163M，官方 124M 是因 weight tying
[2] 未训练: "Hello, I am ..."   # 未训练模型输出乱码（预期）
```

> **参数量说明**：我们实现的 163M 而非 124M，是因为没用 **weight tying**（输出层与 token embedding 共享权重）。OpenAI 官方 GPT-2 用了 tying，省掉一大部分参数。ch05 会讲。

## 踩坑记录

- **163M vs 124M**：未做 weight tying 导致参数偏大，notebook 里已注明。
- **dropout 位置**：嵌入后、残差路径、注意力权重后都要 dropout，防过拟合。
- **未训练输出乱码**：正常。ch05 预训练后才有意义。

## 思考题 / 扩展

1. 为什么用 pre-LN 而非 post-LN？（pre-LN 训练更稳定，梯度流更顺）
2. GELU 相比 ReLU 的优势？（处处可微、负半轴不"死"）
3. 残差连接为什么能缓解梯度消失？（提供梯度直通路径）
4. 📎 **bonus**：`ch04/bonus/` 8 种现代注意力变体——GQA(分组)、MLA(潜变量)、SWA(滑窗)、MoE(专家混合) 等，是全书最密集的架构宝藏。

---

> 📌 **下一步**：进入 [第 5 章：预训练](../ch05-pretraining/notes.md)，给这个模型喂数据、跑训练循环，让它学会生成。
