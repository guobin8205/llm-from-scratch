# 第 02 章 · 处理文本数据

> **状态**：`[x]` 已完成 ｜ `2026-07-31`
>
> **对应原书**：Chapter 02 — 处理文本数据

---

## 本章目标

把人类可读的文字，转化成 GPT 能消费的训练样本 `(input_batch, target_batch)`。
完成一条完整的数据流水线：**原始文本 → 分词 → token ID → 滑动窗口 → DataLoader 批次**。

这是整个项目的数据基石——后续所有章节的模型都吃这份数据。

## 核心内容

分词器、滑动窗口、dataloader。具体五步：

1. **读取语料**：原书的短篇故事 "The Verdict"（20479 字符）
2. **分词**：先正则简单分词（建立直觉）→ 再用 BPE（真正可用）
3. **token ↔ 整数 ID**：模型只认数字，需要词表做映射
4. **滑动窗口**：把长序列切成无数 `(input, target)` 对，target 右移一位
5. **封装 DataLoader**：批处理，供训练循环迭代

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 分词 | Tokenization | 把文本切成 token 的过程 |
| 字节对编码 | BPE (Byte Pair Encoding) | 把未知词拆成已知子词的分词算法，词表固定、泛化好 |
| 词表 | Vocabulary | 所有 token 的 ID 映射（GPT-2 是 50257） |
| 词元 | Token | 文本被切分后的最小单位 |
| 上下文长度 | Context length / max_length | 一次输入的 token 数 |
| 滑动窗口 | Sliding window | 在长序列上滑动生成训练样本 |
| 步长 | Stride | 每次滑动的距离；通常等于 max_length 避免重叠 |
| 词嵌入 | Embedding | 把 token ID 映射为稠密向量（第 3 章详讲） |
| 特殊 token | Special tokens | `<\|endoftext\|>`（文本结束）、`<\|unk\|>`（未知词） |

## 代码走读

本章三个核心件（整理版见 `solution.py`，复用模块见 `src/gpt/data.py`）：

### 1. `tokenizer.encode(text)` —— tiktoken BPE 分词
- 用 GPT-2 同款分词器，把任意文本切成 token ID 列表
- **为什么不用简单正则分词**：正则词表会无限膨胀，且遇到新词只能映射成 `<unk>`；BPE 把未知词拆成子词，词表固定（50257）且能泛化

### 2. `GPTDatasetV1` —— 滑动窗口 Dataset
- 核心循环：`for i in range(0, len(token_ids) - max_length, stride)`
- 每次取 `input = ids[i:i+max]`、`target = ids[i+1:i+max+1]`（**target 右移一位**）
- 这是自回归语言模型的本质：模型学"看到上文 → 预测下一个词"

### 3. `create_dataloader_v1` —— 批处理包装
- 把 `GPTDatasetV1` 包成 `torch.utils.data.DataLoader`
- 关键参数：`batch_size`（批大小）、`drop_last`（丢弃不满一批的尾样本）

```python
# 核心用法
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4)
for input_batch, target_batch in dataloader:
    ...  # 训练
```

## 运行结果

`solution.py` 端到端输出：
```
[1/3] 已读取语料，共 20479 字符
[2/3] 数据集共 1286 个样本，160 个批次
[3/3] 第一个批次:
  inputs  形状: torch.Size([8, 4]), dtype: torch.int64
  targets 形状: torch.Size([8, 4]), dtype: torch.int64
  inputs[0]:  [40, 367, 2885, 1464]
  targets[0]: [367, 2885, 1464, 1807]   ← 右移一位 ✓
```

notebook 从头到尾执行无报错。

## 踩坑记录

- **Windows 相对路径**：notebook 从章节目录 `ch02-text-data/` 运行，读语料用 `../data/...`；脚本 `solution.py` 用 `Path(__file__)` 定位根目录，避免依赖 cwd。两套路径都要处理。
- **stride 选择**：demo 用 `stride=1` 会让样本高度重叠（数据膨胀但信息冗余）；真实训练用 `stride=max_length` 避免重叠。
- **环境插曲**：原计划用 Python 3.12 venv + cu121 torch，但 2.4GB CUDA 包下载不稳定。后发现系统 Python 3.14 已装好 torch 2.13.0+cu130 且 CUDA 可用，直接改用，省去下载。教训：先查现有环境再假设。

## 思考题 / 扩展

1. 如果把 `stride` 设为 1（最大化样本数）vs 设为 `max_length`（无重叠），对训练有什么不同影响？哪种更可能过拟合？
2. BPE 为什么选"字节对"而不是"字符对"？提示：字节层面可以统一处理任何语言的任何字符。
3. `<|endoftext|>` 这个特殊 token 在训练数据里起什么作用？如果不加会怎样？
4. 📎 **bonus**：尝试 `ch02/05_bpe-from-scratch/`，自己从零实现一个 BPE 分词器，理解合并规则是如何学出来的。

---

> 📌 **下一步**：进入 [第 3 章：注意力机制](../ch03-attention/notes.md)。
> token ID 还是离散整数，第 3 章会先把它们变成**词嵌入（embedding）**，再讲注意力如何让 token 之间"相互关注"。
