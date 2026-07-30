# 学习路线图

> 本项目是《Build a Large Language Model (from Scratch)》（Sebastian Raschka）的课程代码学习仓库。
> 目标：从零用 PyTorch 实现并训练一个 GPT 模型，并延伸到现代 LLM 架构（Llama/Qwen/Gemma 等）与进阶主题。

> ✅ **更新于 2026-07-31**：全部内容已完成——附录 A-E（含文献索引与练习答案）、主线 7 章、4 个扩展主题，以及 bonus + 练习 notebook。所有代码均可运行、已推送到 [GitHub](https://github.com/guobin8205/llm-from-scratch)。

---

## 完成状态总览

| 部分 | 状态 | 说明 |
|------|------|------|
| 附录 A：PyTorch 入门 | ✅ | 张量/autograd/nn.Module/训练循环/DataLoader/GPU |
| 附录 B：参考文献 | ✅ | 按章节归类的关键论文索引 |
| 附录 C：练习题答案 | ✅ | ch02-ch07 各精选 2-3 道核心练习 + 可运行解答 |
| 主线 ch01–ch07 | ✅ | 数据→注意力→GPT→预训练→分类→指令 |
| 附录 D：训练循环增强 | ✅ | lr调度/早停/梯度裁剪/checkpoint |
| 附录 E：LoRA | ✅ | 参数高效微调，只训 0.54% 参数 |
| 扩展 ext-*（4 个） | ✅ | 量化/蒸馏/RAG/RLHF |
| Bonus + 练习 notebook | ✅ | 见各章详细清单 |

每章固定三件套：`notes.md`（中文笔记）+ `*.ipynb`（可运行 notebook）+ `solution.py`（整理版代码）。

---

## 📚 内容分层

| 层 | 内容 | 状态 |
|----|------|------|
| 0️⃣ | **附录 A：PyTorch 入门** | ✅ 前置基础 |
| 1️⃣ | **主线 ch01–ch07**（严格顺序） | ✅ 原书核心 |
| 2️⃣ | **附录 B/C/D/E + 各章 bonus + 练习** | ✅ 选学 |
| 3️⃣ | **扩展 ext-*** | ✅ 官方无的进阶主题 |

---

## 0️⃣ 附录 A — PyTorch 入门 ✅

| 目录 | 主题 |
|------|------|
| [`appendix-A-pytorch/`](../appendix-A-pytorch/) | 张量/autograd/nn.Module/训练循环/DataLoader/GPU 六大支柱 |

> 全书前置。没有 PyTorch 基础会全程卡壳，建议 ch02 前过一遍。

---

## 1️⃣ 主线：从零构建 GPT（ch01–ch07）✅

每章产出：`notes.md` + `*.ipynb` + `solution.py`。

| 章 | 目录 | 主题 | 核心产出 |
|----|------|------|---------|
| 1 | [`ch01-introduction/`](../ch01-introduction/) | 理解 LLM | 两阶段范式、自回归生成（笔记为主） |
| 2 | [`ch02-text-data/`](../ch02-text-data/) | 处理文本数据 | BPE 分词/滑动窗口/DataLoader |
| 3 | [`ch03-attention/`](../ch03-attention/) | 注意力机制 | self→causal→multi-head，Q/K/V，因果掩码 |
| 4 | [`ch04-gpt-from-scratch/`](../ch04-gpt-from-scratch/) | 从零搭建 GPT | LayerNorm/GELU/FFN/残差/TransformerBlock/GPTModel |
| 5 | [`ch05-pretraining/`](../ch05-pretraining/) | 预训练 | 训练循环/下一步预测loss/加载OpenAI权重/生成 |
| 6 | [`ch06-classification/`](../ch06-classification/) | 分类微调 | 分类头替换/冻结backbone/情感分类 |
| 7 | [`ch07-instruction/`](../ch07-instruction/) | 指令微调 | Alpaca格式/loss masking/SFT |

---

## 2️⃣ 选学：附录 B / C / D / E + 各章 bonus + 练习 ✅

### 附录

| 目录 | 主题 | 亮点 |
|------|------|------|
| [`appendix-B-references/`](../appendix-B-references/) | 参考文献索引 | 按章节归类关键论文（Attention/Llama/DPO/LoRA…） |
| [`appendix-C-exercises/`](../appendix-C-exercises/) | 练习题答案索引 | ch02-ch07 各精选 2-3 道核心练习 |
| [`appendix-D-training-loop/`](../appendix-D-training-loop/) | 训练循环增强 | lr调度器(warmup+cosine)/早停/梯度裁剪/checkpoint |
| [`appendix-E-lora/`](../appendix-E-lora/) | LoRA 参数高效微调 | ⭐低秩适配，只训 0.54% 参数准确率 100% |

### 各章练习题答案

| 文件 | 练习主题 |
|------|---------|
| [`ch02/exercise-solutions`](../ch02-text-data/exercise-solutions.ipynb) | BPE 编码探究 / stride 影响 |
| [`ch03/exercise-solutions`](../ch03-attention/exercise-solutions.ipynb) | 因果掩码验证 / 多头维度 |
| [`ch04/exercise-solutions`](../ch04-gpt-from-scratch/exercise-solutions.ipynb) | 参数量计算（124M vs 163M）/ dropout |
| [`ch05/exercise-solutions`](../ch05-pretraining/exercise-solutions.ipynb) | 初始loss=ln(vocab) / 温度采样 / 权重转置 |
| [`ch06/exercise-solutions`](../ch06-classification/exercise-solutions.ipynb) | 最后token分类 / 冻结策略对比 |
| [`ch07/exercise-solutions`](../ch07-instruction/exercise-solutions.ipynb) | loss masking / Alpaca 模板 |

### ch03 bonus（2 个）

| 文件 | 主题 |
|------|------|
| [`efficient-mha.ipynb`](../ch03-attention/bonus/efficient-mha.ipynb) | 权重分割 vs 堆叠两种 MHA 实现效率对比 |
| [`understanding-buffers.ipynb`](../ch03-attention/bonus/understanding-buffers.ipynb) | `register_buffer` 深入讲解 |

### ch04 bonus：⭐8 种注意力变体（全书最密集的架构宝藏）

| 文件 | 主题 | 参考模型 |
|------|------|---------|
| [`01-kv-cache.ipynb`](../ch04-gpt-from-scratch/bonus/01-kv-cache.ipynb) | KV 缓存（推理加速基石，~5×） | — |
| [`02-gqa.ipynb`](../ch04-gpt-from-scratch/bonus/02-gqa.ipynb) | 分组查询注意力 | Llama-3 |
| [`03-mla.ipynb`](../ch04-gpt-from-scratch/bonus/03-mla.ipynb) | 多头潜在注意力（省 88% 缓存） | DeepSeek-V2 |
| [`04-swa.ipynb`](../ch04-gpt-from-scratch/bonus/04-swa.ipynb) | 滑动窗口注意力 | Mistral |
| [`05-moe.ipynb`](../ch04-gpt-from-scratch/bonus/05-moe.ipynb) | 混合专家（稀疏激活） | Mixtral |
| [`06-deltanet.ipynb`](../ch04-gpt-from-scratch/bonus/06-deltanet.ipynb) | 门控 DeltaNet（线性注意力） | Nemotron |
| [`07-dsa.ipynb`](../ch04-gpt-from-scratch/bonus/07-dsa.ipynb) | 差分注意力（双分支相减） | Sakana AI |
| [`08-kv-sharing.ipynb`](../ch04-gpt-from-scratch/bonus/08-kv-sharing.ipynb) | 跨层 KV 共享 | YOCO/CLA |

### ch05 bonus：⭐现代 LLM zoo + 训练工具（18 个）

**训练工具类（01-09）：**

| 文件 | 主题 |
|------|------|
| [`01-gutenberg.ipynb`](../ch05-pretraining/bonus/01-gutenberg.ipynb) | 在更大语料（Gutenberg）上预训练 |
| [`02-lr-schedulers.ipynb`](../ch05-pretraining/bonus/02-lr-schedulers.ipynb) | 学习率调度器（warmup+cosine） |
| [`03-hparam-tuning.ipynb`](../ch05-pretraining/bonus/03-hparam-tuning.ipynb) | 超参网格搜索 |
| [`04-training-speed.ipynb`](../ch05-pretraining/bonus/04-training-speed.ipynb) | 训练加速（混合精度/梯度累积） |
| [`05-muon.ipynb`](../ch05-pretraining/bonus/05-muon.ipynb) | ⭐Muon 优化器（牛顿-舒尔茨正交化） |
| [`06-alt-weight-loading.ipynb`](../ch05-pretraining/bonus/06-alt-weight-loading.ipynb) | 替代权重加载（键名映射） |
| [`07-mem-efficient-loading.ipynb`](../ch05-pretraining/bonus/07-mem-efficient-loading.ipynb) | 内存高效加载（meta device） |
| [`08-extending-tokenizers.ipynb`](../ch05-pretraining/bonus/08-extending-tokenizers.ipynb) | 扩展词表（新增 token） |
| [`09-user-interface.ipynb`](../ch05-pretraining/bonus/09-user-interface.ipynb) | Gradio 生成界面 |

**架构转换类（10-18）：**

| 文件 | 主题 |
|------|------|
| [`10-gpt-to-llama.ipynb`](../ch05-pretraining/bonus/10-gpt-to-llama.ipynb) | ⭐GPT→Llama（RoPE/RMSNorm/SwiGLU 三大改造） |
| [`11-qwen3.ipynb`](../ch05-pretraining/bonus/11-qwen3.ipynb) | Qwen3（QK-Norm） |
| [`12-gemma3.ipynb`](../ch05-pretraining/bonus/12-gemma3.ipynb) | Gemma3（SWA + 缩放 embedding） |
| [`13-gemma4.ipynb`](../ch05-pretraining/bonus/13-gemma4.ipynb) | Gemma4（muP） |
| [`14-olmo3.ipynb`](../ch05-pretraining/bonus/14-olmo3.ipynb) | OLMo3（完全开源 + 可学习缩放） |
| [`15-tiny-aya.ipynb`](../ch05-pretraining/bonus/15-tiny-aya.ipynb) | TinyAya（多语言分词） |
| [`16-ch05-with-llms.ipynb`](../ch05-pretraining/bonus/16-ch05-with-llms.ipynb) | 用 HF 真实 LLM 跑训练流程 |
| [`17-llama3-standalone.ipynb`](../ch05-pretraining/bonus/17-llama3-standalone.ipynb) | 自包含 standalone Llama 3.2 |
| [`18-llm-zoo-overview.ipynb`](../ch05-pretraining/bonus/18-llm-zoo-overview.ipynb) | LLM zoo 总览对比 |

### ch06 bonus（3 个）

| 文件 | 主题 |
|------|------|
| [`01-additional-experiments.ipynb`](../ch06-classification/bonus/01-additional-experiments.ipynb) | 最后token(100%) vs 首token(69%) 对比 |
| [`02-more-datasets.ipynb`](../ch06-classification/bonus/02-more-datasets.ipynb) | IMDb 影评分类（含下载+回退） |
| [`03-user-interface.ipynb`](../ch06-classification/bonus/03-user-interface.ipynb) | Gradio 情感分类界面 |

### ch07 bonus（3 个）

| 文件 | 主题 |
|------|------|
| [`01-dpo.ipynb`](../ch07-instruction/bonus/01-dpo.ipynb) | ⭐DPO 直接偏好优化（margin +11.8→+29.2） |
| [`02-model-evaluation.ipynb`](../ch07-instruction/bonus/02-model-evaluation.ipynb) | 本地启发式评估（关键词/长度/多样性） |
| [`03-user-interface.ipynb`](../ch07-instruction/bonus/03-user-interface.ipynb) | Gradio 指令对话界面 |

---

## 3️⃣ 扩展主题（官方无，基于主线延伸）✅

| 目录 | 性质 | 主题 | 亮点 |
|------|------|------|------|
| [`ext-quantization/`](../ext-quantization/) | 🔬 真做 | int8/int4 量化 | 省 75%/87.5% 显存，QLoRA 衔接 |
| [`ext-distillation/`](../ext-distillation/) | 🔬 真做 | 知识蒸馏 | KL散度+温度，teacher→student |
| [`ext-rag/`](../ext-rag/) | 🏗️ 工程化 | 检索增强生成 | embedding检索+拼prompt生成 |
| [`ext-rlhf/`](../ext-rlhf/) | 📖 原理 | RLHF 三阶段 | SFT→奖励模型→PPO，对比DPO |

---

## 依赖与学习顺序

```
附录A (PyTorch) ──→ ch01 ──→ ch02 ──→ ch03 ──→ ch04 ──→ ch05 ──→ ch06 ──→ ch07
                                          │        │        │        │
                                          │        │        │        ├─ ch07/bonus-dpo ⭐
                                          │        │        │        ├─ appendix-E (LoRA) ⭐
                                          │        │        │        └─ appendix-D (训练增强)
                                          │        │        │
                                          │        │        ├─ ch06/bonus
                                          │        │        └─ ext-quantization (→QLoRA)
                                          │        │
                                          │        ├─ ch05/bonus-* (LLM zoo) ⭐
                                          │        └─ ext-distillation
                                          │
                                          ├─ ch04/bonus-* (8种注意力变体) ⭐
                                          └─ ch03/bonus

ch07 完成 ──→ ext-rlhf（原理） / ext-rag（独立demo）
```

⭐ 标记的是高价值推荐 bonus（GQA、GPT→Llama、DPO、Muon、LoRA）。

**建议节奏**：附录A → 主线 7 章（一口气走完）→ 按兴趣深入 bonus → 扩展主题。
