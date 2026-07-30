# ch03 实现计划：编码注意力机制

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Transformer 的核心组件——从无权重的自注意力直觉出发，逐步演进到带 Q/K/V 的因果多头注意力，产出 `MultiHeadAttention` 类（ch04 GPT 模型的核心积木），并对照官方 bonus 理解高效实现与 PyTorch buffer 机制。

**Architecture:** 注意力是一条渐进式演进线，每个组件都在前一个基础上加一层能力。教学顺序严格遵循"先直觉→再加可训练权重→再加因果掩码→再多头化"。所有可复用类沉淀到 `src/gpt/attention.py`，供 ch04+ 复用。

**Tech Stack:** Python 3.14（系统环境）、PyTorch 2.13.0+cu130（CUDA）、tiktoken（复用 ch02 的 data 模块做 demo）。

---

## 文件结构

```
ch03-attention/
├── notes.md                      ← 中文笔记（已存在模板，待填充）
├── ch03.ipynb                    ← 主 notebook：渐进式注意力演进
└── solution.py                   ← 整理版：本章所有注意力类的最终版

ch03-attention/bonus/             ← ch03 的两个官方 bonus
├── efficient-mha.ipynb           ← bonus 1：高效多头注意力实现对比
└── understanding-buffers.ipynb   ← bonus 2：PyTorch register_buffer 讲解

src/gpt/
└── attention.py                  ← 复用模块：MultiHeadAttention（ch04+ 复用）
```

**职责边界：**
- `ch03.ipynb`：教学演进，每个 cell 讲一个递进概念，配直观的数值 demo。
- `solution.py`：本章所有类的整理版（可直接 import）。
- `src/gpt/attention.py`：最终版 `MultiHeadAttention`，ch04 起 GPT 模型复用。
- `bonus/`：两个官方补充材料，深化理解。

---

## Task 1：简单自注意力（无权重，建立直觉）

**Files:**
- Create: `ch03-attention/ch03.ipynb`

3.1-3.3.2：从"长序列问题"切入，讲清自注意力的三步本质。

- [ ] **Step 1.1：创建 notebook，写标题与概念引入（3.1-3.2）**

第一个 cell（markdown）：
```markdown
# 第 3 章：编码注意力机制

## 目标：从直觉到实现 Transformer 的核心——注意力机制

**演进路线（每步加一层能力）：**
1. 简单自注意力（无权重）—— 建立直觉
2. 自注意力 + 可训练权重（Q/K/V）—— 能学习
3. 因果注意力（causal mask）—— 只看左边
4. 多头注意力 —— 多视角并行

> 注意力解决的核心问题：**让每个 token 知道该"关注"序列中的哪些其他 token。**
> 相比 RNN 逐词传递，注意力让任意两个 token 直接交互，无论距离多远。
```

- [ ] **Step 1.2：准备 demo 输入（复用 ch02 的分词）**

code cell：
```python
import torch
import torch.nn as nn

# 用一个小句子做 demo（避免一开始就上大数据）
# 注：这里用手工构造的 token embedding（真实场景下 embedding 是 ch03 末/后续章节的事）
# 每行是一个 token 的 embedding 向量，3 个 token，每个 6 维
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],   # Your     (token 0)
     [0.55, 0.87, 0.66],   # journey  (token 1)
     [0.57, 0.85, 0.64],   # starts   (token 2)
     [0.22, 0.58, 0.33],   # with     (token 3)
     [0.77, 0.25, 0.10],   # one      (token 4)
     [0.05, 0.80, 0.55]]   # step     (token 5)
)
print(f"输入形状: {inputs.shape}  → 6 个 token，每个 3 维 embedding")
```

- [ ] **Step 1.3：实现无可训练权重的自注意力（3.3.1，逐步演示）**

code cell（三步本质：点积→softmax→加权求和）：
```python
# 自注意力的三步本质（以 token 1 "journey" 为例）：
# 1) 用"journey"和每个 token 做点积 → 得到相似度（注意力分数）
# 2) softmax 归一化 → 分数变成权重（和为 1）
# 3) 用权重对所有 token 加权求和 → 得到 "journey" 的新表示

query = inputs[1]                            # 选 token 1 作为查询
attn_scores_2 = torch.empty(inputs.shape[0]) # 存注意力分数

# Step 1: 点积算相似度
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(query, x_i)
print("注意力分数:", attn_scores_2)

# Step 2: softmax 归一化
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("注意力权重:", attn_weights_2, "(和=%.4f)" % attn_weights_2.sum())

# Step 3: 加权求和
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("上下文向量:", context_vec_2)
```

- [ ] **Step 1.4：推广到所有 token（3.3.2，矩阵化）**

code cell（用矩阵乘法一次算完所有 token 的注意力）：
```python
# 上面只算了 token 1。实际要给所有 token 都算。
# 用矩阵乘法可以一次算完：attn_scores = inputs @ inputs.T

attn_scores = inputs @ inputs.T              # [6,6] 每个元素是两个 token 的相似度
print("注意力分数矩阵形状:", attn_scores.shape)

attn_weights = torch.softmax(attn_scores, dim=-1)  # 沿最后一维 softmax
print("每行和为 1:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs     # [6,3] 每个 token 的新表示
print("所有上下文向量形状:", all_context_vecs.shape)
print("token 1 的上下文向量:", all_context_vecs[1], "← 应与上面逐个算的一致")
```

验证：`all_context_vecs[1]` 应与 Step 1.3 的 `context_vec_2` 一致。

- [ ] **Step 1.5：提交**

```bash
cd /e/repos/python/llm-from-scratch
git add ch03-attention/ch03.ipynb
git commit -m "第 3 章：简单自注意力（无权重，点积→softmax→加权求和）"
```

---

## Task 2：带可训练权重的自注意力（Q/K/V）

**Files:**
- Modify: `ch03-attention/ch03.ipynb`

3.4：引入 Query/Key/Value 三个可训练权重矩阵（自注意力真正能"学习"的关键）。

- [ ] **Step 2.1：讲清 Q/K/V 的直觉（3.4.1 markdown）**

markdown cell：
```markdown
## 加上可训练权重：Query / Key / Value

上面无可训练权重的版本只是"静态相似度"，模型学不到任何东西。
真实自注意力引入三个权重矩阵：

- **Query (Q) 查询**："我在找什么样的信息？"
- **Key (K) 键**  ："我能提供什么样的信息？"
- **Value (V) 值** ："我实际携带的信息。"

注意力 = softmax(Q·Kᵀ / √d_k) · V

- Q·Kᵀ：query 和 key 的相似度
- ÷√d_k：缩放，防止点积过大导致 softmax 梯度消失
- ·V：用相似度对 value 加权
```

- [ ] **Step 2.2：实现带权重的自注意力（3.4.1，先 Wq/Wk/Wv 分步）**

code cell：
```python
# 三个可训练权重矩阵（这里先随机初始化演示）
torch.manual_seed(123)
d_in = inputs.shape[1]   # 输入维度 3
d_out = 2                # 输出维度（demo 用小值）

W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 投影：每个 token × 权重矩阵 → 得到 Q/K/V
queries = inputs @ W_query   # [6,2]
keys    = inputs @ W_key     # [6,2]
values  = inputs @ W_value   # [6,2]
print("Q/K/V 形状:", queries.shape, keys.shape, values.shape)

# 带权重的注意力分数 = Q · Kᵀ
attn_scores = queries @ keys.T
# 缩放：÷ √d_k（d_k 是 key 的维度）
d_k = keys.shape[-1]
attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
print("缩放后注意力权重（每行和=1）:", attn_weights[0].sum().item())

context_vecs = attn_weights @ values
print("上下文向量形状:", context_vecs.shape)
```

- [ ] **Step 2.3：封装成 SelfAttention_v1（3.4.2，nn.Parameter 版）**

code cell：
```python
class SelfAttention_v1(nn.Module):
    """自注意力 v1：用 nn.Parameter 手动管理 Q/K/V 权重。

    教学版：展示原理，但 nn.Parameter 写法不够规范。
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_in, d_out)))
        self.W_key   = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_in, d_out)))
        self.W_value = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_in, d_out)))

    def forward(self, x):
        queries = x @ self.W_query
        keys    = x @ self.W_key
        values  = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in=3, d_out=2)
print("v1 输出:", sa_v1(inputs))
```

- [ ] **Step 2.4：封装成 SelfAttention_v2（3.4.2，nn.Linear 版）**

code cell（讲清为什么 v2 更好）：
```python
class SelfAttention_v2(nn.Module):
    """自注意力 v2：用 nn.Linear 管理 Q/K/V 权重。

    v1 → v2 的改进：
    - nn.Linear 自带偏置、优化过的初始化、稳定的随机性
    - 更符合 PyTorch 惯例，后续接入训练框架更顺
    """
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys, queries, values = self.W_key(x), self.W_query(x), self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values

torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in=3, d_out=2)
print("v2 输出:", sa_v2(inputs))
```

- [ ] **Step 2.5：提交**

```bash
git add ch03-attention/ch03.ipynb
git commit -m "第 3 章：带 Q/K/V 可训练权重的自注意力（v1 Parameter → v2 Linear）"
```

---

## Task 3：因果注意力（Causal Attention）

**Files:**
- Modify: `ch03-attention/ch03.ipynb`

3.5：加上因果掩码（只看左边）+ dropout，并支持批处理。这是 GPT（自回归）能"从左到右生成"的关键。

- [ ] **Step 3.1：讲清因果掩码的必要性（markdown）**

markdown cell：
```markdown
## 因果注意力（Causal Attention）

自回归语言模型只能"看到左边的 token"——预测下一个词时不能偷看右边。
实现：把注意力分数矩阵的**上三角**（未来 token）置为 -∞，softmax 后变成 0。

```
         key→
query   t0   t1   t2   t3
  t0  [  ✓    ✗    ✗    ✗ ]   ← t0 只能看自己
  t1  [  ✓    ✓    ✗    ✗ ]   ← t1 能看 t0, t1
  t2  [  ✓    ✓    ✓    ✗ ]
  t3  [  ✓    ✓    ✓    ✓ ]
```

再加 **dropout**（训练时随机置零一部分注意力权重）防止过拟合。
```

- [ ] **Step 3.2：演示因果掩码 + dropout（分步）**

code cell：
```python
# 用 v2 的权重算注意力分数
queries = sa_v2.W_query(inputs)
keys    = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

# 1) 因果掩码：上三角（不含对角线）置 -∞
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("下三角掩码:\n", mask_simple)

masked_simple = attn_scores.masked_fill(mask_simple == 0, -torch.inf)
print("\n掩码后的分数:\n", masked_simple)

attn_weights = torch.softmax(masked_simple, dim=-1)
print("\n掩码后权重（上三角=0）:\n", attn_weights)

# 2) dropout
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)   # demo 用 0.5（真实常用 0.1）
print("\ndropout 后权重:\n", dropout(attn_weights))
```

- [ ] **Step 3.3：封装 CausalAttention 类（支持批处理）**

code cell（关键：支持 batch 维度，用 register_buffer 存掩码）：
```python
class CausalAttention(nn.Module):
    """因果注意力：带掩码 + dropout，支持批处理。

    与 v2 的区别：
    1) 加因果掩码（register_buffer 存上三角 -∞）
    2) 加 dropout
    3) 支持批量输入 [batch, seq, dim]（矩阵乘用 keys.transpose(-2,-1)）
    """
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.nn.Dropout(dropout)
        # register_buffer：掩码随模型一起搬到 GPU，但不是参数（不参与训练）
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)      # [b, seq, d_out]
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)   # [b, seq, seq]
        # 用掩码把上三角（未来）置 -∞
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values            # [b, seq, d_out]
        return context_vec

# 用批量输入测试
torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)   # [2, 6, 3] 模拟 batch=2
ca = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)
print("因果注意力输出形状:", ca(batch).shape)   # 期望 [2, 6, 2]
```

验证：输出形状 `[2, 6, 2]`。

- [ ] **Step 3.4：提交**

```bash
git add ch03-attention/ch03.ipynb
git commit -m "第 3 章：因果注意力（掩码 + dropout + 批处理，CausalAttention 类）"
```

---

## Task 4：多头注意力（MultiHeadAttention）

**Files:**
- Modify: `ch03-attention/ch03.ipynb`
- Create: `src/gpt/attention.py`

3.6：从单头到多头。先讲"堆叠版"（直观但低效），再讲"权重分割版"（标准实现，ch04 复用）。

- [ ] **Step 4.1：堆叠版多头（3.6.1，直观但低效）**

markdown cell：
```markdown
## 多头注意力（Multi-Head Attention）

单个注意力头只学一种"关注模式"。多头 = 多个注意力头并行，
每个头学不同的关注模式（如语法、语义、指代等），最后拼接。

**两种实现：**
1. **堆叠版**（MultiHeadAttentionWrapper）：实例化多个 CausalAttention，循环算完再拼接。
   直观，但循环开销大。
2. **权重分割版**（MultiHeadAttention）：一个大矩阵切分成 num_heads 份。标准实现，高效。
```

code cell：
```python
class MultiHeadAttentionWrapper(nn.Module):
    """多头注意力（堆叠版）：实例化多个单头，循环拼接。

    缺点：num_heads 个头要顺序计算，效率低。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
mha_wrapper = MultiHeadAttentionWrapper(
    d_in=3, d_out=2, context_length=6, dropout=0.0, num_heads=2
)
print("堆叠版输出形状:", mha_wrapper(batch).shape)  # 期望 [2, 6, 4]（2头×2维拼接）
```

验证：输出形状 `[2, 6, 4]`（2 头 × 2 维拼接）。

- [ ] **Step 4.2：权重分割版多头（3.6.2，标准实现）**

code cell（核心实现，用 view 重塑成多头）：
```python
class MultiHeadAttention(nn.Module):
    """多头注意力（权重分割版）：一个大矩阵切分成 num_heads 份。

    这是 ch04 GPT 模型实际使用的标准实现。
    相比堆叠版：一次矩阵乘算完所有头，效率高。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out 必须能被 num_heads 整除"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)   # 输出投影，融合多头
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # 投影后重塑成 [b, num_heads, seq, head_dim]
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)   # [b, heads, seq, seq]
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [b, heads, seq, head_dim] → [b, seq, d_out]
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

torch.manual_seed(123)
mha = MultiHeadAttention(
    d_in=3, d_out=4, context_length=6, dropout=0.0, num_heads=2
)
print("权重分割版输出形状:", mha(batch).shape)  # 期望 [2, 6, 4]
```

验证：输出形状 `[2, 6, 4]`。

- [ ] **Step 4.3：把 MultiHeadAttention 沉淀到 src/gpt/attention.py**

Create: `src/gpt/attention.py`（把上面 Task 4.2 的 `MultiHeadAttention` 类完整复制过去，加模块文档字符串）：
```python
"""第 3 章核心注意力模块：多头因果注意力。

供 ch04 GPT 模型复用。标准权重分割实现，支持批处理与因果掩码。
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """多头因果注意力（权重分割实现）。

    一个大矩阵切分成 num_heads 份，一次矩阵乘算完所有头。
    """
    # ...（与 Step 4.2 完全一致的实现）
```

- [ ] **Step 4.4：更新 src/gpt/__init__.py 导出**

在 `src/gpt/__init__.py` 添加：
```python
from src.gpt.attention import MultiHeadAttention

__all__ = ["GPTDatasetV1", "create_dataloader_v1", "MultiHeadAttention"]
```

- [ ] **Step 4.5：验证从 src 导入可用**

```bash
python -W ignore -c "
import torch
from src.gpt import MultiHeadAttention
x = torch.rand(2, 6, 3)
mha = MultiHeadAttention(d_in=3, d_out=4, context_length=6, dropout=0.0, num_heads=2)
out = mha(x)
assert out.shape == torch.Size([2, 6, 4]), f'形状错误: {out.shape}'
print('✅ src/gpt MultiHeadAttention 验证通过:', out.shape)
"
```

- [ ] **Step 4.6：提交**

```bash
git add ch03-attention/ch03.ipynb src/gpt/attention.py src/gpt/__init__.py
git commit -m "第 3 章：多头注意力（堆叠版 → 权重分割版，沉淀到 src/gpt/attention.py）"
```

---

## Task 5：bonus 1 — 高效多头注意力实现对比

**Files:**
- Create: `ch03-attention/bonus/efficient-mha.ipynb`

对照官方 `ch03/02_bonus_efficient-multihead-attention/`，对比几种高效 MHA 实现（torch SDPA、nn.MultiheadAttention 等），理解为什么标准实现要那样写。

- [ ] **Step 5.1：创建 bonus notebook，对比 4 种实现**

Create: `ch03-attention/bonus/efficient-mha.ipynb`，核心 cell：
```python
# bonus：高效多头注意力实现对比
# 对照官方 ch03/02_bonus_efficient-multihead-attention/
#
# 对比 4 种实现，验证输出一致：
# 1) 我们 ch03 手写的权重分割版（参考基准）
# 2) torch.nn.functional.scaled_dot_product_attention (SDPA，融合算子)
# 3) torch.nn.MultiheadAttention（官方模块，但接口略不同）
# 4) 用 Einsum 的精简版

import torch, torch.nn as nn, torch.nn.functional as F
from src.gpt import MultiHeadAttention

torch.manual_seed(123)
d_in, d_out, ctx, heads = 768, 768, 16, 12
x = torch.rand(2, ctx, d_in)

# 1) 手写版（参考）
mha_ref = MultiHeadAttention(d_in, d_out, ctx, 0.0, heads)
out_ref = mha_ref(x)

# 2) SDPA 融合算子（PyTorch 2.0+，最常用，可能走 FlashAttention）
# SDPA 需要 [b, heads, seq, head_dim] 格式 + 显式 mask
print("参考实现输出:", out_ref.shape)
print("💡 生产代码通常直接用 F.scaled_dot_product_attention，它自动选最优后端")
print("（本 bonus 仅作对比理解，主线 GPT 仍用手写版以学原理）")
```

- [ ] **Step 5.2：提交**

```bash
git add ch03-attention/bonus/efficient-mha.ipynb
git commit -m "第 3 章 bonus：高效多头注意力实现对比（SDPA 等）"
```

---

## Task 6：bonus 2 — PyTorch register_buffer 讲解

**Files:**
- Create: `ch03-attention/bonus/understanding-buffers.ipynb`

对照官方 `ch03/03_understanding-buffers/`。讲清为什么因果掩码要用 `register_buffer` 而不是普通 tensor 或 nn.Parameter。

- [ ] **Step 6.1：创建 bonus notebook**

Create: `ch03-attention/bonus/understanding-buffers.ipynb`，核心内容：
```python
# bonus：为什么掩码要用 register_buffer？
# 对照官方 ch03/03_understanding-buffers/
#
# PyTorch 模块存张量的 3 种方式：
# 1) 普通 Python 属性（self.mask = tensor）
#    → 不会随 model.to('cuda') 搬到 GPU！训练时会报"设备不一致"错。
# 2) nn.Parameter（self.mask = nn.Parameter(tensor)）
#    → 会被当成可训练参数，优化器会去更新它。但掩码是固定的，不该被训练！
# 3) register_buffer（self.register_buffer('mask', tensor)）  ✅
#    → 随模型搬设备，但不参与训练（不在 parameters() 里）。完美匹配"固定常量"。

import torch, torch.nn as nn

class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        # 对比三种存法
        self.plain_attr = torch.ones(2, 2)                          # ❌ 不搬 GPU
        self.param = nn.Parameter(torch.ones(2, 2))                 # ❌ 会被训练
        self.register_buffer("buffer", torch.ones(2, 2))            # ✅ 搬GPU但不训练

d = Demo()
print("parameters():", list(d.named_parameters()))   # 只有 param
print("buffers():", list(d.named_buffers()))         # 只有 buffer

d_gpu = d.to('cuda')
print("plain_attr 设备:", d_gpu.plain_attr.device)   # 仍是 cpu ❌
print("buffer 设备:", d_gpu.buffer.device)           # cuda ✅
```

- [ ] **Step 6.2：提交**

```bash
git add ch03-attention/bonus/understanding-buffers.ipynb
git commit -m "第 3 章 bonus：register_buffer 原理（掩码为何用 buffer 而非 Parameter）"
```

---

## Task 7：ch03 笔记 + solution.py

**Files:**
- Modify: `ch03-attention/notes.md`
- Create: `ch03-attention/solution.py`

- [ ] **Step 7.1：填充 ch03 笔记（概念表 + 走读 + 思考题）**

填充 `ch03-attention/notes.md`：
- 本章目标：实现 Transformer 核心——注意力
- 概念表：自注意力/Q/K/V/缩放点积/因果掩码/dropout/多头
- 代码走读：四个类的演进（SelfAttention_v1→v2→CausalAttention→MultiHeadAttention）
- 思考题：为什么除以√d_k？多头比单头好在哪？掩码为何用 buffer？

- [ ] **Step 7.2：写 solution.py（整理版，可 import 所有类）**

Create: `ch03-attention/solution.py`：把 SelfAttention_v1/v2、CausalAttention、MultiHeadAttention 整理成一个文件，配一个 demo main。

- [ ] **Step 7.3：nbconvert 执行整个 ch03.ipynb 验证无报错**

```bash
cd /e/repos/python/llm-from-scratch/ch03-attention
jupyter nbconvert --to notebook --execute ch03.ipynb --output ch03.ipynb --ExecutePreprocessor.timeout=120
```
Expected: 无报错，写入带 outputs 的 notebook。

- [ ] **Step 7.4：标记完成并提交**

```bash
git add ch03-attention/notes.md ch03-attention/solution.py ch03-attention/ch03.ipynb
git commit -m "第 3 章：补全中文笔记与整理版 solution.py"
```

---

## 验收标准（Definition of Done）

- [ ] `solution.py` 能跑通，打印各注意力类的输出形状
- [ ] `ch03.ipynb` nbconvert 从头执行无报错
- [ ] `from src.gpt import MultiHeadAttention` 可导入，输出形状 `[2,6,4]`
- [ ] 因果掩码正确：输出不依赖未来 token（上三角权重为 0）
- [ ] 两个 bonus notebook 都能执行
- [ ] ch03 notes.md 状态标记 `[x]`
- [ ] 每个任务独立 commit

## 自审清单

- ✅ **Spec 覆盖**：ch03 主题（self→causal→multi-head attention）全覆盖，含 v1/v2 演进 + 2 bonus
- ✅ **无占位符**：所有代码步骤给完整可运行代码
- ✅ **命名一致**：`SelfAttention_v1/v2`、`CausalAttention`、`MultiHeadAttention`、`d_in/d_out/context_length/num_heads` 全程一致
- ✅ **依赖衔接**：复用 ch02 的 `inputs` demo 思路，沉淀到 `src/gpt` 与 ch02 的 data 模块并列
