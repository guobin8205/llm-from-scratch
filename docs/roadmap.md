# 学习路线图

> 本项目是《Build a Large Language Model (from Scratch)》（Sebastian Raschka）的课程代码学习仓库。
> 目标：从零用 PyTorch 实现并训练一个 GPT 模型，并延伸到现代 LLM 架构（Llama/Qwen/Gemma 等）与进阶主题。
>
> 本路线图**完整对齐官方仓库** [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)，覆盖主线 7 章、5 个附录，以及各章的 bonus 补充材料。

## 整体结构

项目内容分四层，按优先级与依赖关系排列：

| 层 | 内容 | 优先级 | 说明 |
|----|------|--------|------|
| 0️⃣ | **附录 A：PyTorch 入门** | 必学前置 | 全书基础，开工前先过 |
| 1️⃣ | **主线 ch01–ch07** | 必学（顺序） | 原书核心，从数据到 GPT 到微调 |
| 2️⃣ | **附录 D / E + 各章 bonus** | 选学 | 训练增强、LoRA、注意力变体、LLM zoo 等 |
| 3️⃣ | **自创扩展 ext-*** | 选学 | 官方没有的主题：蒸馏、量化、RAG 等 |

> 📌 **目录约定**：附录用 `appendix-X/`，各章 bonus 作为子目录放在对应章下（如 `ch04/bonus-kv-cache/`），学到时才创建。自创扩展用 `ext-*`。

---

## 0️⃣ 前置：附录 A — PyTorch 入门

| 目录 | 对应官方 | 主题 | 里程碑 |
|------|---------|------|--------|
| `appendix-A-pytorch/` | `appendix-A/01_main-chapter-code` | 张量、autograd、nn.Module、训练循环、DataLoader、GPU | 掌握 PyTorch 基本功 |
| ↳ `ddp/` | `appendix-A/.../DDP-script*.py` | 分布式数据并行（DDP） | 了解多卡训练（选学） |

> 在进入 ch02 之前完成。没有 PyTorch 基础会全程卡壳。

---

## 1️⃣ 主线：从零构建 GPT（ch01–ch07，严格顺序）

每章产出：`notes.md` + `chXX.ipynb` + `solution.py`（或整理版模块）。

| 章 | 目录 | 主题 | 核心产出 | 官方 bonus |
|----|------|------|---------|-----------|
| 1 | `ch01-introduction/` | 理解 LLM | 笔记为主（无代码） | — |
| 2 | `ch02-text-data/` | 处理文本数据 | 分词/BPE/滑动窗口/dataloader/embedding | BPE从零、tokenizer对比、embedding直觉 |
| 3 | `ch03-attention/` | 注意力机制 | self→causal→multi-head attention | 高效MHA对比、register_buffer讲解 |
| 4 | `ch04-gpt-from-scratch/` | 从零搭建 GPT | LayerNorm/GELU/残差/TransformerBlock/GPT/生成 | ⭐8种注意力变体（见下） |
| 5 | `ch05-pretraining/` | 预训练 | 训练循环/损失/困惑度/加载OpenAI权重/采样解码 | ⭐现代LLM zoo（见下）+ 权重加载 |
| 6 | `ch06-classification/` | 分类微调 | 分类头/SST-2/评估/加载微调模型 | 额外实验、IMDB、sklearn/BERT基线 |
| 7 | `ch07-instruction/` | 指令微调 | 指令格式化/SFT/响应提取/评估 | ⭐DPO偏好优化、数据集工具 |

---

## 2️⃣ 选学：附录 D / E + 各章 bonus

### 附录

| 目录 | 对应官方 | 主题 |
|------|---------|------|
| `appendix-D-training-loop/` | `appendix-D` | 训练循环增强：学习率调度、训练技巧 |
| `appendix-E-lora/` | `appendix-E` | 参数高效微调 LoRA（⭐替代原 ext-lora） |

### ch04 bonus：注意力 / 架构变体（⭐全书最密集的宝藏）

学完 ch04 主线后，这些 bonus 把你的 GPT 改造成各种现代架构：

| bonus 目录 | 对应官方 | 主题 | 参考模型 |
|-----------|---------|------|---------|
| `ch04/bonus-kv-cache/` | `ch04/03_kv-cache` | KV 缓存（推理加速基石） | — |
| `ch04/bonus-gqa/` | `ch04/04_gqa` | 分组查询注意力 | LLaMA-2 |
| `ch04/bonus-mla/` | `ch04/05_mla` | 多头潜注意力 | DeepSeek |
| `ch04/bonus-swa/` | `ch04/06_swa` | 滑动窗口注意力 | Mistral/Gemma |
| `ch04/bonus-moe/` | `ch04/07_moe` | 混合专家 | Mixtral 等 |
| `ch04/bonus-deltanet/` | `ch04/08_deltanet` | Gated DeltaNet（线性注意力） | — |
| `ch04/bonus-dsa/` | `ch04/09_dsa` | DeepSeek 稀疏注意力 | DeepSeek |
| `ch04/bonus-kv-sharing/` | `ch04/10_kv-sharing` | 跨层 KV 共享 | Gemma |

### ch05 bonus：现代 LLM 实现 zoo + 工具

| bonus 目录 | 对应官方 | 主题 |
|-----------|---------|------|
| `ch05/bonus-weight-loading/` | `ch05/02,08` | 替代权重加载（HF safetensors / 内存高效） |
| `ch05/bonus-gpt-to-llama/` | `ch05/07` | ⭐GPT→Llama2/3 架构转换（RoPE/RMSNorm） |
| `ch05/bonus-qwen3/` | `ch05/11,16` | Qwen3 / Qwen3.5 实现 |
| `ch05/bonus-gemma/` | `ch05/12,17` | Gemma3 / Gemma4 实现 |
| `ch05/bonus-olmo3/` | `ch05/13` | OLMo3 实现 |
| `ch05/bonus-tiny-aya/` | `ch05/15` | TinyAya 实现 |
| `ch05/bonus-pretraining-gutenberg/` | `ch05/03` | 在 Gutenberg 语料上继续预训练 |
| `ch05/bonus-hparam-tuning/` | `ch05/05` | 超参搜索 |
| `ch05/bonus-training-speed/` | `ch05/10` | 训练加速（单卡/多卡DDP） |
| `ch05/bonus-muon/` | `ch05/18` | Muon 优化器 |
| `ch05/bonus-extending-tokenizers/` | `ch05/09` | 扩展 tiktoken 词表 |

### ch07 bonus

| bonus 目录 | 对应官方 | 主题 |
|-----------|---------|------|
| `ch07/bonus-dpo/` | `ch05/04` | ⭐直接偏好优化 DPO |
| `ch07/bonus-model-evaluation/` | `ch07/03` | LLM-as-judge 评估（OpenAI/Ollama） |
| `ch07/bonus-dataset-generation/` | `ch07/05` | 合成数据生成 |

### ch02 / ch03 bonus

| bonus 目录 | 对应官方 | 主题 |
|-----------|---------|------|
| `ch02/bonus-bpe-from-scratch/` | `ch02/05` | ⭐BPE 分词器从零实现 |
| `ch02/bonus-embedding-intuition/` | `ch02/03` | embedding vs matmul 直觉 |
| `ch02/bonus-dataloader-intuition/` | `ch02/04` | 滑动窗口可视化 |
| `ch03/bonus-efficient-mha/` | `ch03/02` | 高效多头注意力实现对比 |
| `ch03/bonus-understanding-buffers/` | `ch03/03` | PyTorch register_buffer 讲解 |

---

## 3️⃣ 自创扩展（官方无，基于主线延伸）

| 目录 | 性质 | 说明 | 依赖 |
|------|------|------|------|
| `ext-distillation/` | 🔬 真做 | 知识蒸馏（teacher→student，KL+温度） | ch05 |
| `ext-quantization/` | 🔬 真做 | int8/int4 量化 | ch05 |
| `ext-rlhf/` | 📖 原理 | RLHF 三阶段原理（SFT→RM→PPO） | 随时 |
| `ext-rag/` | 🏗️ 工程化 | LangChain + 向量库 RAG | 独立 |

> 注：原计划的 `ext-lora` 已**移至官方 `appendix-E`**（官方有现成 LoRA 内容）。

---

## 依赖与学习顺序

```
附录A (PyTorch) ──→ ch01 ──→ ch02 ──→ ch03 ──→ ch04 ──→ ch05 ──→ ch06 ──→ ch07
                                          │        │        │        │
                                          │        │        │        ├─ ch07/bonus-dpo ⭐
                                          │        │        │        ├─ appendix-E (LoRA) ⭐
                                          │        │        │        └─ appendix-D (训练增强)
                                          │        │        │
                                          │        │        ├─ ch06/bonus-imdb
                                          │        │        └─ ext-quantization
                                          │        │
                                          │        ├─ ch05/bonus-* (LLM zoo, 权重加载) ⭐
                                          │        ├─ ext-distillation
                                          │        └─ ch05/bonus-gpt-to-llama ⭐
                                          │
                                          ├─ ch04/bonus-* (8种注意力变体) ⭐
                                          └─ ch05 任何 bonus 需先完成 ch04

ch07 完成 ──→ ext-rlhf（原理） / ext-rag（独立demo）
```

⭐ 标记的是高价值推荐 bonus（KV缓存、GQA、GPT→Llama、DPO、BPE从零、LoRA）。

**建议节奏**：附录A → 主线 7 章（一口气走完）→ 按兴趣深入 bonus → 自创扩展。
