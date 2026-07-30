# 第 07 章 · 指令微调

> **状态**：`[x]` 已完成
>
> **对应原书**：Chapter 07 — Finetuning to Follow Instructions

---

## 本章目标

让 GPT 从「只会续写」进化为「能遵循指令」。给定「翻译这个句子」，模型输出翻译结果而非编故事——这是把通用语言模型变成**实用助手**的关键。

## 核心内容

### 三个关键点

1. **Alpaca 指令格式**：用 `### Instruction / ### Input / ### Response` 模板组织指令数据
2. **loss masking（本章灵魂）**：只对 Response 算 loss，Instruction 部分 mask 为 -100
3. **微调训练**：冻结 backbone，在指令数据上继续训练（out_head 保持 vocab_size 用于生成）

### 为什么需要 loss masking？

若对整个序列算 loss，模型会花力气学「复述指令」，浪费容量。只算 Response 的 loss，让模型聚焦「给定指令，生成正确回答」这个目标。PyTorch 的 `cross_entropy(ignore_index=-100)` 自动跳过被 mask 的位置。

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 指令微调 | Instruction fine-tuning | 在指令数据上微调，让模型遵循指令 |
| Alpaca 格式 | Alpaca format | 用分隔符区分指令/输入/响应的模板 |
| 损失掩码 | Loss masking | 只对响应部分算 loss，指令部分置 -100 |
| 提示模板 | Prompt template | 把指令结构化的模板 |
| 忽略索引 | Ignore index | cross_entropy 跳过的位置（-100） |

## 代码走读

- `ch07.ipynb` — 可运行 notebook（自造中文指令 demo，loss 10.85→2.05）
- `solution.py` — 整理版：`format_prompt` + `custom_collate`（带 loss mask）+ 训练生成

### 核心实现

```python
# Alpaca 模板
def format_prompt(entry):
    return f"### Instruction:\n{entry['instruction']}\n### Input:\n{entry['input']}\n### Response:\n"

# loss masking：prompt 部分置 -100
targets = input_ids[1:] + [eot]
for i in range(len(prompt_ids)):
    targets[i] = -100
loss = F.cross_entropy(logits.flatten(0,1), targets.flatten(), ignore_index=-100)
```

## 运行结果

- demo 参数：13.27M，可训练 50%（解冻最后块+norm+head）
- 训练 loss：10.85 → 2.05（10 epoch，Response 部分）
- 生成：给定指令能续写出 Response 片段（demo 数据量小，输出不完美）

## 踩坑记录

- **demo 数据量小**：10 条数据 × 3 复制，模型记住了部分映射但泛化差。真实场景需数万条指令数据。
- **未加载预训练权重**：主线用随机初始化验证流程。真实场景必须先加载 OpenAI 权重（ch05）再做指令微调。
- **生成遇 eot 停止**：贪婪生成时遇到 `<|endoftext|>` 停止，避免无限续写。

## 思考题 / 扩展

1. 为什么指令微调只需训练 Response 部分？（信息密度、目标聚焦）
2. 如果不做 loss masking，全序列算 loss 会怎样？（模型会学着复述指令）
3. 指令微调 vs 分类微调（ch06），loss 的本质区别？（序列预测 vs 单点分类）
4. 如何让模型回答更符合人类偏好？（见 bonus 01 的 DPO）

## Bonus

- `bonus/01-dpo.ipynb` — 直接偏好优化（DPO），让模型在好/坏回答间学习偏好
- `bonus/02-model-evaluation.ipynb` — 指令模型评估（本地启发式指标）
- `bonus/03-user-interface.ipynb` — Gradio 对话交互界面
