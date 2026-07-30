# ch05 实现计划：无监督预训练

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 ch04 搭建的 GPT-124M 真正"学会"语言——实现训练循环（交叉熵损失 + 困惑度），在 "The Verdict" 上 demo 训练，加载 OpenAI GPT-2 权重让它立刻生成通顺文本，并掌握温度采样/top-k 解码策略。18 个 bonus 覆盖训练工具与现代 LLM zoo（Llama/Qwen/Gemma 等）。

**Architecture:** 预训练的标准三件套：**损失函数**（交叉熵，预测下一个 token）+ **优化训练循环**（前向→损失→反向→更新）+ **解码策略**（控制生成随机性）。加载 OpenAI 权重是"捷径"——跳过昂贵的从头训练。训练/生成组件沉淀到 `src/gpt/training.py` + `src/gpt/generate.py`，ch06/ch07 复用。

**Tech Stack:** Python 3.14、PyTorch 2.13.0+cu130（CUDA）、tiktoken、复用 ch02-04 的 `src/gpt` 包。

**执行策略：主线（Task 1-8）优先做完并验证，拿到能训练/能生成通顺文本的 GPT；18 个 bonus（Task 9-12）分批提交，每个讲清原理+参考模型+改造点。**

---

## 文件结构

```
ch05-pretraining/
├── notes.md                      ← 中文笔记（主线 + bonus 总览）
├── ch05.ipynb                    ← 主 notebook：训练循环 + 损失 + 解码 + 加载权重
├── solution.py                   ← 整理版：训练/生成/损失函数
└── bonus/                        ← 18 个 bonus（分 3 类）
    ├── 【训练工具】
    │   ├── 01-gutenberg.ipynb            ← 03 在更大语料预训练
    │   ├── 02-lr-schedulers.ipynb        ← 04 学习率调度
    │   ├── 03-hparam-tuning.ipynb        ← 05 超参搜索
    │   ├── 04-training-speed.ipynb       ← 10 训练加速（单卡/多卡DDP）
    │   ├── 05-muon.ipynb                 ← 18 Muon 优化器
    │   ├── 06-alt-weight-loading.ipynb   ← 02 替代权重加载
    │   ├── 07-mem-efficient-loading.ipynb← 08 内存高效加载
    │   ├── 08-extending-tokenizers.ipynb ← 09 扩展词表
    │   └── 09-user-interface.ipynb       ← 06 Gradio 界面
    ├── 【GPT→Llama】
    │   └── 10-gpt-to-llama.ipynb         ← 07 GPT→Llama2/3（RoPE/RMSNorm/SwiGLU）
    └── 【LLM zoo】
        ├── 11-qwen3.ipynb                ← 11/16 Qwen3/Qwen3.5
        ├── 12-gemma3.ipynb               ← 12 Gemma3
        ├── 13-gemma4.ipynb               ← 17 Gemma4
        ├── 14-olmo3.ipynb                ← 13 OLMo3
        ├── 15-tiny-aya.ipynb             ← 15 TinyAya
        ├── 16-ch05-with-llms.ipynb       ← 14 用 Llama/Qwen 跑 ch05 训练
        ├── 17-llama3-standalone.ipynb    ← 07 standalone-llama32
        └── 18-llm-zoo-overview.ipynb     ← LLM zoo 总览对比

src/gpt/
├── training.py                   ← 复用：损失/训练循环（ch06/ch07 复用）
└── generate.py                   ← 复用：温度/top-k 生成
```

---

## Task 1：回顾生成 + 评估数据准备（5.1.1）

**Files:**
- Create: `ch05-pretraining/ch05.ipynb`

- [ ] **Step 1.1：标题 + 预训练全景（markdown）**

```markdown
# 第 5 章：无监督预训练

## 目标：让 ch04 的 GPT-124M 真正"学会"语言

**预训练三件套：**
1. 损失函数（交叉熵：预测下一个 token）
2. 训练循环（前向→损失→反向→更新）
3. 解码策略（温度/top-k 控制生成）

**关键捷径**：加载 OpenAI 已训练好的 GPT-2 权重 → 立刻能生成通顺文本，
省去从头训练的高昂成本。
```

- [ ] **Step 1.2：准备训练/验证数据（复用 ch02 dataloader）**

```python
import torch, sys; sys.path.insert(0, "..")
from src.gpt.data import create_dataloader_v1
from src.gpt import GPTModel, GPT_CONFIG_124M, generate_text_simple

GPT_CONFIG_124M["context_length"] = 256  # demo 用小上下文，训练快

with open("../data/the-verdict.txt", encoding="utf-8") as f:
    text_data = f.read()

# 划分训练/验证集（90/10）
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
train_loader = create_dataloader_v1(train_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"], drop_last=True, shuffle=True)
val_loader = create_dataloader_v1(val_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"], drop_last=False, shuffle=False)
print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
```

- [ ] **Step 1.3：提交**

```bash
git add ch05-pretraining/ch05.ipynb
git commit -m "第 5 章：训练/验证数据准备（复用 ch02 dataloader）"
```

---

## Task 2：交叉熵损失 + 困惑度（5.1.2）

**Files:** Modify: `ch05-pretraining/ch05.ipynb`

- [ ] **Step 2.1：理解损失（从 logits 到交叉熵）**

```python
# 模型输出是 logits（未归一化分数），要变成"预测下一个 token 的概率"
# 交叉熵损失 = -log(正确 token 的预测概率)
# 目标：让正确 token 的预测概率尽量大 → 损失尽量小

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # cross_entropy 期望展平的 [N, vocab] 和 [N] 目标
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
```

- [ ] **Step 2.2：困惑度 perplexity**

```python
# 困惑度 = exp(交叉熵损失)，更直观的指标
# 含义：模型在每个位置"平均在多少个词之间犹豫"
# 困惑度=1 完美；困惑度=vocab_size 等同随机猜
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0: return float("nan")
    num_batches = num_batches or len(data_loader)
    for i, (x, y) in enumerate(data_loader):
        if i >= num_batches: break
        total_loss += calc_loss_batch(x, y, model, device)
    return total_loss / num_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M).to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print(f"未训练 - 训练损失: {train_loss:.3f}, 验证损失: {val_loss:.3f}")
print(f"未训练 - 困惑度: {torch.exp(torch.tensor(train_loss)):.1f}（接近 vocab_size 说明随机）")
```

- [ ] **Step 2.3：提交**

```bash
git add ch05-pretraining/ch05.ipynb
git commit -m "第 5 章：交叉熵损失 + 困惑度（评估模型质量）"
```

---

## Task 3：训练循环 train_model_simple（5.2）⭐核心

**Files:** Modify: `ch05-pretraining/ch05.ipynb` + Create: `src/gpt/training.py`

- [ ] **Step 3.1：训练循环（标准 PyTorch 训练流程）**

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, tokens_seen = [], [], 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss); val_losses.append(val_loss)
                print(f"Ep {epoch+1} step {global_step:06d}: train {train_loss:.3f}, val {val_loss:.3f}")

        # 每个 epoch 末生成一段文本看效果
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, tokens_seen
```

- [ ] **Step 3.2：辅助函数 evaluate_model + generate_and_print_sample**

```python
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0).to(device)
    with torch.no_grad():
        out_ids = generate_text_simple(model, encoded, 50, context_size)
    decoded = tokenizer.decode(out_ids.squeeze().tolist())
    print(decoded.replace("\n", " "))
    model.train()
```

- [ ] **Step 3.3：实际训练（在 the-verdict 上 demo）**

```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)

num_epochs = 10
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer)
# 观察损失下降 + 生成文本逐渐"像样"
```

- [ ] **Step 3.4：沉淀训练函数到 src/gpt/training.py**

Create: `src/gpt/training.py`：把 calc_loss_batch/calc_loss_loader/evaluate_model/train_model_simple/generate_and_print_sample 完整复制，加文档字符串。

- [ ] **Step 3.5：提交**

```bash
git add ch05-pretraining/ch05.ipynb src/gpt/training.py src/gpt/__init__.py
git commit -m "第 5 章：训练循环 train_model_simple（核心，沉淀 src/gpt/training.py）"
```

---

## Task 4：解码策略——温度采样 + top-k（5.3.1）⭐

**Files:** Modify: `ch05-pretraining/ch05.ipynb` + Create: `src/gpt/generate.py`

- [ ] **Step 4.1：温度缩放（控制随机性）**

```python
# generate_text_simple 是贪婪的（总取最大概率），导致输出重复/无聊
# 温度采样：把 logits 除以温度 T，再 softmax 采样
# T<1 更确定（趋近贪婪）；T>1 更随机（多样但可能乱）；T=1 原始
def generate_with_temperature(model, idx, max_new_tokens, context_size, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :] / temperature  # 温度缩放
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probas, num_samples=1)  # 按概率采样（非贪婪）
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

- [ ] **Step 4.2：top-k 采样（只在 top-k 候选里采样）**

```python
# top-k：只保留概率最高的 k 个 token，其余置 0（屏蔽低质量候选），再采样
def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[:, [-1]], torch.full_like(logits, -float("inf")), logits)
        if temperature > 0:
            probas = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and idx_next.item() == eos_id: break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

- [ ] **Step 4.3：沉淀到 src/gpt/generate.py 并对比效果**

Create: `src/gpt/generate.py`。在 notebook 里对比贪婪/温度/top-k 的输出差异。

- [ ] **Step 4.4：提交**

```bash
git add ch05-pretraining/ch05.ipynb src/gpt/generate.py src/gpt/__init__.py
git commit -m "第 5 章：解码策略（温度采样 + top-k，沉淀 src/gpt/generate.py）"
```

---

## Task 5：加载 OpenAI GPT-2 权重（5.3.2）⭐亮点

**Files:** Modify: `ch05-pretraining/ch05.ipynb` + Create: `ch05-pretraining/gpt_download.py`

- [ ] **Step 5.1：下载脚本 gpt_download.py**

对照官方 `gpt_download.py`：从 OpenAI 下载 GPT-2 124M 权重（~500MB，存到 models/）。
```python
# gpt_download.py：download_and_load_gpt2(model_size="124M", models_dir="models")
# 从 https://openaipublic.blob.core.windows.net/gpt-2 下载 124M 的权重文件
```

- [ ] **Step 5.2：load_weights_into_gpt（权重映射）**

```python
# 把 OpenAI 的权重字典映射到我们的 GPTModel 结构
# 关键：OpenAI 权重的 key 和我们的层命名不同，要逐层对应
def load_weights_into_gpt(gpt, params):
    # 嵌入层、位置编码、每个 transformer block（QKV/norm/ffn）、最终 norm、输出头
    ...  # 对照官方实现，逐层 load_state_dict
```

- [ ] **Step 5.3：加载并生成（立刻通顺！）**

```python
from gpt_download import download_and_load_gpt2
from src.gpt import GPTModel, GPT_CONFIG_124M
# 加载 OpenAI 权重 → 模型立刻能生成通顺文本（不用训练！）
model = GPTModel(GPT_CONFIG_124M)
load_weights_into_gpt(model, params)
# 用 generate 生成，对比从头训练的版本
```

- [ ] **Step 5.4：提交**

```bash
git add ch05-pretraining/gpt_download.py ch05-pretraining/ch05.ipynb
git commit -m "第 5 章：加载 OpenAI GPT-2 权重（让模型立刻生成通顺文本）"
```

---

## Task 6：保存/加载模型 + 绘制损失曲线

**Files:** Modify: `ch05-pretraining/ch05.ipynb`

- [ ] **Step 6.1：torch.save 保存训练好的模型**

```python
torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, "models/gpt_124M_pretrained.pth")
```

- [ ] **Step 6.2：matplotlib 绘制训练/验证损失曲线**

- [ ] **Step 6.3：提交**

```bash
git commit -m "第 5 章：保存/加载模型 + 损失曲线可视化"
```

---

## Task 7：主线 nbconvert 验证 + 笔记 + solution.py

**Files:** notes.md + solution.py

- [ ] **Step 7.1：nbconvert 执行 ch05.ipynb 验证（训练耗时，给长超时）**
- [ ] **Step 7.2：填充 notes.md**（损失/困惑度/训练循环/解码/加载权重，含概念表+踩坑+思考题）
- [ ] **Step 7.3：写 solution.py**（训练/生成/损失整理版 + demo）
- [ ] **Step 7.4：提交**

---

## Task 8：训练工具类 bonus（9 个）

**Files:** `ch05-pretraining/bonus/01-09-*.ipynb`

| bonus | 对应官方 | 主题 |
|-------|---------|------|
| 01-gutenberg | 03 | 在 Project Gutenberg 更大语料上预训练 |
| 02-lr-schedulers | 04 | 学习率调度器（warmup/cosine） |
| 03-hparam-tuning | 05 | 超参搜索 |
| 04-training-speed | 10 | 训练加速（单卡优化 + 多卡 DDP） |
| 05-muon | 18 | Muon 优化器 |
| 06-alt-weight-loading | 02 | 替代权重加载（HF safetensors/transformers/raw PyTorch） |
| 07-mem-efficient-loading | 08 | 内存高效加载大模型 |
| 08-extending-tokenizers | 09 | 扩展 tiktoken 词表 |
| 09-user-interface | 06 | Gradio 交互界面 |

每个：原理讲解 + 参考点 + 概念 demo + 提交。

## Task 9：GPT→Llama bonus

**Files:** `ch05-pretraining/bonus/10-gpt-to-llama.ipynb`（对照官方 07_gpt_to_llama）

GPT→Llama2/3 的改造：RoPE 旋转位置编码、RMSNorm（替代 LayerNorm）、SwiGLU（替代 GELU FFN）。这是见识真实模型架构的重点 bonus。

## Task 10：LLM zoo bonus（8 个）

**Files:** `ch05-pretraining/bonus/11-18-*.ipynb`

| bonus | 主题 |
|-------|------|
| 11-qwen3 | Qwen3/Qwen3.5 实现 |
| 12-gemma3 | Gemma3 实现 |
| 13-gemma4 | Gemma4 E2B/E4B |
| 14-olmo3 | OLMo3 实现 |
| 15-tiny-aya | TinyAya 实现 |
| 16-ch05-with-llms | 用 Llama/Qwen 跑 ch05 训练循环 |
| 17-llama3-standalone | standalone Llama 3.2 |
| 18-llm-zoo-overview | LLM zoo 总览对比（参数/架构/特点） |

---

## 验收标准

**主线：**
- [ ] 训练循环跑通，损失明显下降（the-verdict 上 10 epochs）
- [ ] 训练后生成文本比未训练"像样"
- [ ] 加载 OpenAI 权重后生成通顺文本
- [ ] 温度/top-k 能改变生成风格
- [ ] `from src.gpt import train_model_simple, generate` 可用
- [ ] ch05 notes.md 标记 `[x]`

**bonus：** 18 个 notebook 各自能执行，讲清原理+参考模型+改造点

## 自审清单

- ✅ Spec 覆盖：ch05 主线（损失/训练/解码/加载权重）+ 18 bonus 全覆盖
- ✅ 命名一致：calc_loss_batch/train_model_simple/generate 等全程一致
- ✅ 依赖衔接：复用 ch04 GPTModel，沉淀 training.py + generate.py
- ✅ 训练参数：lr=4e-4, weight_decay=0.1, 10 epochs（the-verdict demo）
