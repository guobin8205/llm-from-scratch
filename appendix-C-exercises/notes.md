# 附录 C · 练习题答案索引（Exercise Solutions）

> **性质**：📝 答案索引（每章精选 2-3 道核心练习 + 可运行解答）
> **对应官方**：appendix-C（官方为各章练习答案的链接索引）
> **状态**：`[x]` 已完成

---

## 本附录目标

汇总 ch02-ch07 各章的**精选练习题及答案**。每章从官方原书的练习中挑选 2-3 道最有代表性的核心练习，配可运行代码解答，帮助巩固该章关键概念。

> 📌 官方 appendix-C 是纯索引（链接到各章 `exercise-solutions.ipynb`）。本项目为每章补了精简版答案 notebook，放各章目录下。

---

## 📋 各章练习答案

| 章 | 主题 | 答案文件 | 练习数 |
|----|------|---------|:------:|
| 2 | 文本数据 | [`ch02/exercise-solutions.ipynb`](../ch02-text-data/exercise-solutions.ipynb) | 2 |
| 3 | 注意力机制 | [`ch03/exercise-solutions.ipynb`](../ch03-attention/exercise-solutions.ipynb) | 2 |
| 4 | 从零搭建 GPT | [`ch04/exercise-solutions.ipynb`](../ch04-gpt-from-scratch/exercise-solutions.ipynb) | 2 |
| 5 | 预训练 | [`ch05/exercise-solutions.ipynb`](../ch05-pretraining/exercise-solutions.ipynb) | 3 |
| 6 | 分类微调 | [`ch06/exercise-solutions.ipynb`](../ch06-classification/exercise-solutions.ipynb) | 2 |
| 7 | 指令微调 | [`ch07/exercise-solutions.ipynb`](../ch07-instruction/exercise-solutions.ipynb) | 2 |

---

## 练习题概览（精选题目）

### ch02：文本数据
1. **BPE 编码探究**：用 `tiktoken` 编码一个生造词，观察它如何被拆成子词
2. **数据加载器参数**：对比不同 `stride` / `batch_size` 对样本数和重叠的影响

### ch03：注意力机制
1. **因果掩码验证**：手动构造注意力分数，验证掩码后未来 token 权重为 0
2. **多头维度**：改变 `num_heads` 对输出的影响（保持 `d_out` 不变）

### ch04：从零搭建 GPT
1. **参数量计算**：手算 GPT-2 124M 各组件的参数量，理解 163M vs 124M 的差异
2. **dropout 影响**：对比 train/eval 模式下输出的确定性

### ch05：预训练
1. **损失函数**：验证随机初始化时 loss ≈ ln(vocab) ≈ 10.8
2. **温度采样**：对比不同温度下生成文本的多样性
3. **权重加载**：理解为什么需要转置 OpenAI 的权重

### ch06：分类微调
1. **分类 token 选择**：为什么用最后一个 token 而非第一个？（见 ch06/bonus/01）
2. **冻结策略对比**：冻结 backbone vs 全量微调的参数量与效果

### ch07：指令微调
1. **loss masking 验证**：确认 prompt 部分的梯度为 0
2. **Alpaca 格式**：为什么需要固定模板分隔符

---

> 💡 建议先**自己尝试**每道题，再看答案。卡住时回到对应章节的 `notes.md` 复习原理。
