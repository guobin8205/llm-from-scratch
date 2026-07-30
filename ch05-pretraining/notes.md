# 第 05 章 · 无监督预训练

> **状态**：`[x]` 已完成 ｜ `2026-07-31`
>
> **对应原书**：Chapter 05 — Pretraining on Unlabeled Data

---

## 本章目标

把 ch04 搭好的 GPT 模型真正训练起来。核心：**训练循环 → loss 下降 → 加载 OpenAI 预训练权重 → 生成可读文本**。这是把"随机初始化的积木"变成"懂语言的模型"的关键一步。

## 核心内容

### 预训练四件事

1. **训练循环**：数据 → 前向 → 算 loss → 反向 → 更新
2. **下一步预测 loss**：flatten 后交叉熵（和分类不同，这是序列任务）
3. **加载 OpenAI 权重**：跳过烧钱的预训练，直接用官方训好的 GPT-2 权重
4. **文本生成**：训练后用贪婪/采样解码生成文本

### 下一步预测（Next-Token Prediction）

语言模型的训练目标：给定前 N 个 token，预测第 N+1 个。loss 是对整个序列做交叉熵：

```python
logits = model(input_batch)  # [b, seq, vocab]
loss = F.cross_entropy(logits.flatten(0,1), target_batch.flatten())
# target 是 input 右移一位（ch02 滑动窗口已构造好）
```

### 训练循环（贯穿全书的 5 步范式）

```python
for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()         # 清梯度
        loss = calc_loss(x, y)        # 前向 + 算 loss
        loss.backward()               # 反向
        optimizer.step()               # 更新
```

### 加载 OpenAI 权重（关键一步）

demo 训练（2万字语料）不足以让 124M 出好文本。原书提供脚本下载 OpenAI 官方预训练权重并加载：
- OpenAI 用 `tf.Transpose` 权重布局，需转置对齐到我们的参数名
- 加载后生成质量大幅提升

> **weight tying**：OpenAI GPT-2 输出层与 token embedding 共享权重，所以官方 124M；我们未绑定是 163M。

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 预训练 | Pretraining | 无标注文本上做下一步预测 |
| 下一步预测 | Next-token prediction | 预测第 N+1 个 token |
| 交叉熵损失 | Cross-entropy loss | 分类/预测的标准损失 |
| 困惑度 | Perplexity | exp(loss)，越低越好 |
| 权重加载 | Weight loading | 加载预训练权重 |
| 权重绑定 | Weight tying | 输出层共享 embedding 权重 |
| 贪婪解码 | Greedy decoding | 每步取最大概率 token |
| 温度采样 | Temperature sampling | 调节采样随机性 |
| top-k 采样 | Top-k sampling | 只从 top-k 采样 |

## 代码走读

- `ch05.ipynb` — 训练循环 + loss 下降 + 生成（小配置 demo）
- `solution.py` — 完整版：含梯度累积/温度采样/top-k，端到端运行
- `bonus/` — ⭐18 个：现代 LLM zoo + 训练工具
  - 训练类：Gutenberg 语料、lr调度器、超参搜索、训练加速、Muon优化器、权重加载×2、扩展词表、Gradio
  - 架构类：**GPT→Llama**（RoPE/RMSNorm/SwiGLU）、Qwen3、Gemma3、OLMo3、standalone Llama3.2

### 训练循环核心（solution.py）

```python
def train_model_simple(model, train_loader, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0,1), y.flatten())
            loss.backward()
            optimizer.step()
```

### 生成函数（支持温度/top-k）

```python
def generate(model, idx, max_new_tokens, temperature=0.0, top_k=None):
    # temperature>0: 采样；=0: 贪婪
    # top_k: 只从概率最高的 k 个采样
```

## 运行结果

`solution.py` 端到端输出（小配置 demo）：
```
[1] 语料 20,479 字符
[3] 训练（10 epoch）...
Epoch  1/10 | train 10.6420
...
Epoch 10/10 | train 6.0504      ← loss 从 10.6 降到 6.0
[4] 生成: "I had a little ..."
```

> demo 用小配置（128维/2层）+ 未加载预训练权重，输出质量有限。加载 OpenAI 权重后质量跃升。

## 踩坑记录

- **demo 用小配置**：完整 124M 在 demo 语料上跑太慢，主线用 `{emb_dim:128, n_layers:2}` 快速验证流程。
- **数据量太小**：2 万字的 the-verdict 远不够，真实预训练需 TB 级语料。
- **生成质量**：demo 模型未加载预训练权重，生成基本是乱码，这是预期的。

## 思考题 / 扩展

1. 为什么 loss 从 ~10 开始？（vocab 50257，随机预测 ln(50257)≈10.8）
2. 温度采样中，温度高/低分别什么效果？（高→更随机多样；低→更确定保守）
3. 加载 OpenAI 权重后，为什么还要"继续训练"而非直接用？（适配具体任务）
4. 📎 **bonus 10**：`ch05/bonus/10-gpt-to-llama` 把 GPT 改造成真实 Llama 架构（RoPE/RMSNorm/SwiGLU），是见识工业界设计的重点。

---

> 📌 **下一步**：进入 [第 6 章：文本分类微调](../ch06-classification/notes.md)，让模型从"只会续写"变成"能判断情感"。
