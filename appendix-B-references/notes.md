# 附录 B · 参考文献与延伸阅读（References & Further Reading）

> **性质**：📖 文献索引（无代码）
> **对应官方**：appendix-B（官方原文为 "No code in this appendix"）
> **状态**：`[x]` 已完成

---

## 本附录目标

整理贯穿全书的关键论文与参考资料，按章节归类。每篇都是理解对应章节原理的「一手来源」，深入某个主题时按图索骥。

> 📌 本附录是文献索引，无代码。原书附录 B 也是纯文献列表。

---

## 📘 主线章节对应论文

### ch01-02：LLM 概述与文本处理
- **GPT-2**：*Language Models are Unsupervised Multitask Learners*（Radford et al., 2019）— 本书复现的模型
- **GPT-3**：*Language Models are Few-Shot Learners*（Brown et al., 2020）— Scaling law 的代表作
- **BPE**：*Neural Machine Translation of Rare Words with Subword Units*（Sennrich et al., 2016）— tiktoken 用的分词算法

### ch03：注意力机制
- **⭐ Attention Is All You Need**（Vaswani et al., 2017）— Transformer 原始论文，全书基石
  - <https://arxiv.org/abs/1706.03762>

### ch04：从零搭建 GPT
- **LayerNorm**：*Layer Normalization*（Ba et al., 2016）
- **GELU**：*Gaussian Error Linear Units*（Hendrycks & Gimpel, 2016）
- **ResNet（残差）**：*Deep Residual Learning for Image Recognition*（He et al., 2016）

### ch05：预训练
- **GPT-2 / GPT-3**（同上）— 预训练范式
- **InstructGPT**：*Training language models to follow instructions with human feedback*（Ouyang et al., 2022）— ch07 RLHF 的来源
- **Weight Tying**：*Using the Output Embedding to Improve Language Models*（Press & Wolf, 2016）— 解释 124M 参数量的来源

### ch06：分类微调
- **ULMFiT**：*Universal Language Model Fine-tuning for Text Classification*（Howard & Ruder, 2018）— 微调范式的奠基
- **Feature-based vs Fine-tuning**：对比提取特征与端到端微调（原书 ch06 实验）

### ch07：指令微调
- **Alpaca**：*Stanford Alpaca*（Taori et al., 2023）— 指令模板格式来源
- **⭐ DPO**：*Direct Preference Optimization*（Rafailov et al., 2023）— ch07 bonus 的核心
  - <https://arxiv.org/abs/2305.18290>

---

## 🔬 附录与扩展主题

### appendix-D：训练循环增强
- **Cosine LR Schedule**：*SGDR: Stochastic Gradient Descent with Warm Restarts*（Loshchilov & Hutter, 2017）

### appendix-E / ext-quantization：LoRA 与量化
- **⭐ LoRA**：*LoRA: Low-Rank Adaptation of Large Language Models*（Hu et al., 2021）
  - <https://arxiv.org/abs/2106.09685>
- **QLoRA**：*QLoRA: Efficient Finetuning of Quantized LLMs*（Dettmers et al., 2023）— 量化+LoRA
- **GPTQ**：*GPTQ: Accurate Post-Training Quantization*（Frantar et al., 2023）

### ch04 bonus：注意力变体
- **GQA**：*GQA: Training Generalized Multi-Query Transformer Models*（Ainslie et al., 2023）— Llama-3
- **MLA**：*DeepSeek-V2*（2024）— 多头潜在注意力
- **SWA**：*Longformer / Mistral 7B*（Beltagy et al., 2020 / Jiang et al., 2023）
- **MoE**：*Mixtral of Experts*（Jiang et al., 2024）

### ch05 bonus：现代 LLM 架构
- **⭐ Llama**：*Llama 2 / Llama 3*（Touvron et al., 2023 / Meta, 2024）— RoPE/RMSNorm/SwiGLU
- **RoPE**：*RoFormer: Enhanced Transformer with Rotary Position Embedding*（Su et al., 2021）
- **RMSNorm**：*Root Mean Square Layer Normalization*（Zhang & Sennrich, 2019）
- **SwiGLU**：*GLU Variants Improve Transformer*（Shazeer, 2020）
- **Muon 优化器**：*Moonshot / Keller Jordan*（2024）

### ext-distillation：知识蒸馏
- **⭐ Distillation**：*Distilling the Knowledge in a Neural Network*（Hinton et al., 2015）— 蒸馏开山之作

### ext-rlhf：RLHF
- **⭐ RLHF**：*Fine-Tuning Language Models from Human Preferences*（Ziegler et al., 2019 / Christie et al.）
- **PPO**：*Proximal Policy Optimization Algorithms*（Schulman et al., 2017）
- **InstructGPT**（同 ch05）— RLHF 三阶段的完整描述

### ext-rag：RAG
- **⭐ RAG**：*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*（Lewis et al., 2020）
  - <https://arxiv.org/abs/2005.11401>

---

## 📚 教材与综合资源

- **本书**：Sebastian Raschka, *Build a Large Language Model (from Scratch)*, Manning, 2024
- **官方代码**：<https://github.com/rasbt/LLMs-from-scratch>
- **The Illustrated Transformer**（Jay Alammar）— 注意力机制的最佳可视化入门
- **The Annotated Transformer**（Harvard NLP）— 逐行实现原始 Transformer

---

> ⭐ 标记的是「必读级」核心论文，建议优先看。
