# 设计文档：《从零构建大模型》课程代码项目

> **日期**：2026-07-30（2026-07-31 修订）
> **书籍**：Sebastian Raschka《Build a Large Language Model (from Scratch)》
> **状态**：已批准（v3，对齐官方仓库全量内容）
> **仓库位置**：`E:\repos\python\llm-from-scratch`
>
> **修订记录（v3）**：核对官方仓库 `rasbt/LLMs-from-scratch` 后，发现 v2 只覆盖了主线骨架，遗漏了 5 个附录与各章大量 bonus。用户决定**全补**。本次修订：纳入附录 A/D/E、各章 bonus（含 ch04 八种注意力变体、ch05 现代 LLM zoo）、将 LoRA 从自创扩展改为官方 appendix-E、补入 DPO。

---

## 1. 项目目标

跟着《Build a Large Language Model (from Scratch)》一书，从零用 PyTorch 实现并训练一个 GPT 模型，覆盖**数据 → 注意力 → 模型 → 预训练 → 微调**全链路。每章产出"可运行代码 + 深度中文注释 + 中文笔记"，复现书中的关键结果。

在原书主线之外，额外加入 5 个进阶扩展主题，作为独立扩展章。

## 2. 已确认的核心决策

> **范围（v3）**：完整对齐官方仓库全量内容 = 主线 7 章 + 5 附录 + 各章 bonus（含注意力变体、LLM zoo）+ 自创扩展。详见 `docs/roadmap.md`。

| 决策项 | 结论 |
|--------|------|
| **技术栈** | Python + PyTorch（忠于原书） |
| **项目位置** | `E:\repos\python\llm-from-scratch`（独立 git 仓库） |
| **学习深度** | 跑通 + 逐行中文注释 + 每章中文笔记 |
| **每章产出** | `*.ipynb`（notebook）+ `solution.py`（整理版代码）+ `notes.md`（笔记），三者都要 |
| **计算环境** | RTX 5070 Ti Laptop 12GB，默认 CUDA，代码写 `device` 抽象 |
| **代码组织** | 方案 A：章节镜像 + `src/gpt/` 共享包 |
| **Python 版本** | 3.12（系统为 3.14，PyTorch CUDA 版不兼容，需用 venv 隔离） |

## 3. 内容分层决策（v3 修订）

项目内容分四层（详见 roadmap.md）：

| 层 | 内容 | 来源 |
|----|------|------|
| 0️⃣ 前置 | 附录 A：PyTorch 入门 | 官方 `appendix-A` |
| 1️⃣ 主线 | ch01–ch07 | 官方主线 |
| 2️⃣ 选学 | 附录 D/E + 各章 bonus | 官方（含 ch04 注意力变体 8 个、ch05 LLM zoo、DPO、BPE从零等） |
| 3️⃣ 自创扩展 | ext-* | 官方无：蒸馏、量化、RLHF原理、RAG |

### 关键修订（vs v2）

- ✅ **LoRA**：从自创 `ext-lora` → 改为学习官方 `appendix-E`（官方有现成 LoRA 内容）
- ✅ **DPO**：新增，来自官方 `ch07/04_preference-tuning-with-dpo`（比纯 RLHF 更实用）
- ✅ **RLHF**：保留为原理笔记（ext-rlhf），DPO 作为其"实战版"补充在 ch07 bonus

### 自创扩展（官方无）

| 扩展章 | 性质 | 说明 |
|--------|------|------|
| 知识蒸馏（distillation） | 真做 | teacher(ch05 GPT-124M)→student(6层)，KL 散度 + 温度 |
| 模型量化（quantization） | 真做 | int8/int4，对比显存/质量，对 12GB 卡实用 |
| RLHF 对齐 | **轻量原理** | 笔记 + 伪代码讲清 SFT→RM→PPO 三阶段 |
| RAG | **工程化** | LangChain + Chroma 向量库，独立 demo |

## 4. 目录结构（v3）

```
llm-from-scratch/
├── README.md / requirements.txt / .gitignore / .gitattributes
├── docs/ (roadmap.md + specs/ + plans/)
│
├── 【0️⃣ 前置】
├── appendix-A-pytorch/            # PyTorch 入门（含 ddp/）
│
├── 【1️⃣ 主线】
├── ch01-introduction/ … ch07-instruction/   # 各章含 bonus-* 子目录（学到再建）
│
├── 【2️⃣ 选学附录】
├── appendix-D-training-loop/      # 训练循环增强
├── appendix-E-lora/               # LoRA（原 ext-lora）
│
├── 【3️⃣ 自创扩展】
├── ext-distillation/ ext-quantization/ ext-rlhf/ ext-rag/
│
├── data/  models/  (gitignore 内容)
└── src/gpt/                       # 跨章复用核心模块
```

> 各章 bonus 作为该章的子目录（如 `ch04/bonus-kv-cache/`），**学到时才创建**，避免一堆空目录。
> 完整 bonus 清单见 `docs/roadmap.md`。

## 5. 章节内容与里程碑

### 0️⃣ 前置：附录 A — PyTorch 入门（ch02 前必学）

| 目录 | 主题 | 里程碑 |
|------|------|--------|
| `appendix-A-pytorch/` | 张量、autograd、nn.Module、训练循环、DataLoader、GPU | 掌握 PyTorch 基本功 |

### 1️⃣ 主线 7 章（严格顺序）

| 章 | 目录 | 主题 | 核心产出 | 高价值 bonus |
|----|------|------|---------|-------------|
| 1 | `ch01-introduction/` | 理解 LLM | 笔记为主 | — |
| 2 | `ch02-text-data/` | 处理文本数据 | 分词/BPE/滑动窗口/dataloader/embedding | ⭐BPE从零 |
| 3 | `ch03-attention/` | 注意力机制 | self→causal→multi-head | 高效MHA、buffers |
| 4 | `ch04-gpt-from-scratch/` | 从零搭建 GPT | LayerNorm/GELU/残差/TransformerBlock/GPT/生成 | ⭐KV缓存/GQA/MLA/SWA/MoE（8种变体） |
| 5 | `ch05-pretraining/` | 预训练 | 训练循环/损失/困惑度/加载OpenAI权重/采样 | ⭐GPT→Llama、LLM zoo |
| 6 | `ch06-classification/` | 分类微调 | 分类头/SST-2/评估 | IMDB、BERT/sklearn基线 |
| 7 | `ch07-instruction/` | 指令微调 | 指令格式化/SFT/响应提取/评估 | ⭐DPO、LLM-as-judge |

### 2️⃣ 选学附录与 bonus

- **附录 D**（训练循环增强）：学习率调度、训练技巧
- **附录 E**（LoRA）：参数高效微调（⭐替代原 ext-lora）
- **ch04 bonus**（8 种注意力变体）：KV缓存、GQA、MLA、SWA、MoE、DeltaNet、DSA、KV-sharing
- **ch05 bonus**（LLM zoo + 工具）：GPT→Llama2/3、Qwen3、Gemma、OLMo3、权重加载、超参搜索、训练加速、Muon
- **ch07 bonus**：DPO、模型评估、数据集生成

### 3️⃣ 自创扩展（官方无）

- `ext-distillation`（知识蒸馏）、`ext-quantization`（量化）、`ext-rlhf`（原理）、`ext-rag`（工程化）

### 学习顺序

```
附录A → ch01→ch07（主线，必顺序）→ 按依赖深入 bonus → 自创扩展
```

**建议节奏**：附录A → 主线 7 章一口气走完 → ch04/ch05 的 bonus（最值得深入）→ 附录 D/E → 自创扩展。

## 6. 每章笔记模板（notes.md）

每章用统一模板，保证可复习：
- **本章目标**：一句话说清解决什么问题
- **核心概念**：关键术语中文解释（附英文对照）
- **代码走读**：本章代码逐模块讲清"为什么这么写"
- **运行结果**：复现的关键输出/图表
- **踩坑记录**：实际遇到的问题与解法
- **思考题/扩展**：留作回顾的提问

## 7. 环境与依赖策略

- 用 **Python 3.12 venv 虚拟环境**隔离（系统 3.14 不兼容 PyTorch CUDA）
- `requirements.txt` 锁定：PyTorch（CUDA 12.x）、NumPy、Matplotlib、tqdm、jupyter
- PyTorch 通过官方 CUDA 索引安装：`--index-url https://download.pytorch.org/whl/cu121`
- 数据集和权重通过脚本下载，**不入库**（`.gitignore` 排除 `data/`、`models/` 内容）

## 8. 提交规范

延续中文提交风格，按章节提交：
- `ch01 初始化项目骨架与笔记模板`
- `第 2 章：处理文本数据（分词/数据集/dataloader）`
- …

## 9. 明确不做（YAGNI）

- ❌ Web 界面 / Gradio demo（超出学习目标）
- ❌ 单元测试框架（靠断言和打印验证即可）
- ❌ 多语言对照（Rust 实现）
- ❌ 分布式 / 多卡训练（单卡 12GB 足够 124M 模型）

## 10. 环境风险提示

- 系统 Python **3.14.0**，PyTorch CUDA 版装不上 → **必须用 Python 3.12 venv**
- GPU: RTX 5070 Ti Laptop, 12227 MiB, driver 596.49 ✅（124M 微调够用）
- 当前未检测到 Python 3.12，需先安装（见 README 环境配置）
