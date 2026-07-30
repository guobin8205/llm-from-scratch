# 第 03 章 · 注意力机制

> **状态**：`[x]` 已完成 ｜ `2026-07-31`
>
> **对应原书**：Chapter 03 — Understanding Large Language Models → 注意力机制

---

## 本章目标

理解并实现 Transformer 的核心：**注意力机制**。从最简单的自注意力出发，逐步加上因果掩码、多头并行，最终得到 GPT 真正用的 `MultiHeadAttention`。这是 ch04 拼装 GPT 的关键零件。

## 核心内容

### 注意力的三步演进

```
1. 自注意力（Self-Attention）   → 每个 token 看所有 token，算相关性
2. 因果注意力（Causal Attention）→ 加掩码，只看左侧（不能偷看未来）
3. 多头注意力（Multi-Head）     → 切成多组并行算，捕捉不同维度的关系
```

### 1. 自注意力：Q/K/V 三元组

每个 token 生成三个向量：
- **Query（查询）**：我想找什么？
- **Key（键）**：我有什么可被找到？
- **Value（值）**：我的实际内容

注意力分数 = `softmax(Q·Kᵀ / √d)`，再乘 V 得到加权和。直觉：用 Query 去和所有 Key 匹配，匹配度高的 Value 权重大。

> **为什么除以 √d**：点积会随维度增大而变大，导致 softmax 进入饱和区（梯度消失）。除以 √d 稳定数值。

### 2. 因果注意力：上三角掩码

自回归生成时，当前位置**不能看未来的 token**。用上三角掩码把未来位置的注意力分数置 `-∞`，softmax 后权重变 0：

```python
mask = torch.triu(torch.ones(L, L), diagonal=1).bool()  # 上三角为 True
attn_scores.masked_fill_(mask, -torch.inf)
```

> `diagonal=1` 保留对角线（自己能看自己），只屏蔽严格上三角（未来）。

### 3. 多头注意力：并行多组 Q/K/V

把 `d_out` 维度切成 `num_heads` 份，每份 `head_dim = d_out // num_heads`，各组独立算注意力。好处：不同头关注不同子空间的关系（有的头看语法、有的看语义）。

**高效实现**（本项目用）：一个大矩阵投影后 reshape 成 `[b, heads, seq, head_dim]`，一次矩阵乘算完所有头，效率远高于堆叠多个单头。

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 自注意力 | Self-Attention | token 间相互关注 |
| 查询/键/值 | Query/Key/Value | 注意力的三元组 |
| 注意力分数 | Attention scores | Q·Kᵀ 的点积 |
| 注意力权重 | Attention weights | softmax 后的归一化权重 |
| 缩放点积 | Scaled dot-product | 点积除以 √d |
| 因果掩码 | Causal mask | 上三角屏蔽未来 |
| 多头注意力 | Multi-Head Attention | 多组并行注意力 |
| 头维度 | Head dim | 每个头的维度 = d_out/heads |
| 缓冲区 | Buffer (register_buffer) | 随模型移动但不训练的张量 |

## 代码走读

- `ch03.ipynb` — 可运行 notebook，从 self-attention 演进到 multi-head
- `solution.py` — `SelfAttention_v1` + `CausalAttention` 整理版
- `src/gpt/attention.py` — 最终的 `MultiHeadAttention`（供 ch04 复用）
- `bonus/efficient-mha.ipynb` — 权重分割 vs 堆叠两种 MHA 实现的效率对比
- `bonus/understanding-buffers.ipynb` — 深入讲解 `register_buffer` 的作用

### `MultiHeadAttention` 核心结构（src/gpt/attention.py）

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 三个投影
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)               # 输出投影融合多头
        self.register_buffer("mask", torch.triu(...).bool())  # 因果掩码 buffer
```

## 运行结果

`solution.py` 输出（注意力演进验证）：
```
v1:     torch.Size([6, 2])      # SelfAttention_v1
Causal: torch.Size([2, 6, 2])   # 批处理因果注意力
MHA:    torch.Size([2, 6, 4])   # 多头注意力（2头）
```

## 踩坑记录

- **`register_buffer` vs `nn.Parameter`**：mask 不参与训练但需随模型搬到 GPU。用 `register_buffer` 注册，`.to('cuda')` 会自动带上它，且不在 `parameters()` 里。详见 bonus。
- **`masked_fill_` 的下划线**：原地操作，省内存。配合 `-torch.inf` 让 softmax 后为 0。
- **view 前要 contiguous**：transpose 后的张量内存不连续，view 会报错，需 `.contiguous()`。

## 思考题 / 扩展

1. 为什么除以 √d 而不是 d？（√d 让点积方差稳定在 1 附近）
2. 多头注意力为什么比单头更强？（不同头关注不同子空间，类似集成）
3. 如果去掉因果掩码会怎样？（模型能"偷看"未来 token，自回归生成就失效了）
4. 📎 **bonus**：`ch04/bonus-gqa` 等变体——现代 LLM 如何用 GQA/MLA 改进多头注意力？

---

> 📌 **下一步**：进入 [第 4 章：从零搭建 GPT](../ch04-gpt-from-scratch/notes.md)，把注意力模块配上 LayerNorm/FFN/残差，拼成完整 GPT。
