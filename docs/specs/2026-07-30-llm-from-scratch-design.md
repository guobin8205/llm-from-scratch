# 设计文档：《从零构建大模型》课程代码项目

> **日期**：2026-07-30
> **书籍**：Sebastian Raschka《Build a Large Language Model (from Scratch)》
> **状态**：已批准（v2）
> **仓库位置**：`E:\repos\python\llm-from-scratch`

---

## 1. 项目目标

跟着《Build a Large Language Model (from Scratch)》一书，从零用 PyTorch 实现并训练一个 GPT 模型，覆盖**数据 → 注意力 → 模型 → 预训练 → 微调**全链路。每章产出"可运行代码 + 深度中文注释 + 中文笔记"，复现书中的关键结果。

在原书主线之外，额外加入 5 个进阶扩展主题，作为独立扩展章。

## 2. 已确认的核心决策

| 决策项 | 结论 |
|--------|------|
| **技术栈** | Python + PyTorch（忠于原书） |
| **项目位置** | `E:\repos\python\llm-from-scratch`（独立 git 仓库） |
| **学习深度** | 跑通 + 逐行中文注释 + 每章中文笔记 |
| **每章产出** | `*.ipynb`（notebook）+ `solution.py`（整理版代码）+ `notes.md`（笔记），三者都要 |
| **计算环境** | RTX 5070 Ti Laptop 12GB，默认 CUDA，代码写 `device` 抽象 |
| **代码组织** | 方案 A：章节镜像 + `src/gpt/` 共享包 |
| **Python 版本** | 3.12（系统为 3.14，PyTorch CUDA 版不兼容，需用 venv 隔离） |

## 3. 扩展主题决策（独立扩展章）

| 扩展章 | 性质 | 说明 |
|--------|------|------|
| 知识蒸馏（distillation） | 真做 | teacher(ch05 GPT-124M)→student(6层)，KL 散度 + 温度 |
| LoRA / QLoRA | 真做 | 自实现 LoRA 层 + bitsandbytes QLoRA |
| 模型量化（quantization） | 真做 | int8/int4，对比显存/质量，对 12GB 卡实用 |
| RLHF 对齐 | **轻量原理** | 笔记 + 伪代码讲清 SFT→RM→PPO 三阶段，不跑完整对齐 |
| RAG | **工程化** | LangChain + Chroma 向量库，独立 demo（偏离"从零"风格） |

## 4. 目录结构

```
llm-from-scratch/
├── README.md                      # 总览 + 学习路线图
├── requirements.txt               # 依赖锁定（CUDA 版 PyTorch）
├── .gitignore                     # 排除 venv/data/models
├── docs/
│   ├── roadmap.md                 # 详细学习路线图
│   └── specs/                     # 设计文档
│
├── 【主线：从零构建 GPT】
├── ch01-introduction/             # 引言（笔记为主）
├── ch02-text-data/                # 分词/滑动窗口/dataloader
├── ch03-attention/                # self→multi-head→causal
├── ch04-gpt-from-scratch/         # LayerNorm/GELU/Transformer/GPT
├── ch05-pretraining/              # 训练循环+加载OpenAI权重+生成
├── ch06-classification/           # 分类头+SST+评估
├── ch07-instruction/              # 指令数据集+Alpaca式微调
│
├── 【扩展：进阶主题】
├── ext-distillation/              # 知识蒸馏
├── ext-lora/                      # LoRA/QLoRA
├── ext-quantization/              # int8/int4 量化
├── ext-rlhf/                      # RLHF 原理（轻量）
├── ext-rag/                       # 工程化 RAG
│
├── data/  (gitignore 内容)        # 数据集
├── models/ (gitignore 内容)       # 权重
└── src/gpt/                       # 跨章复用核心模块
```

## 5. 章节内容与里程碑

### 主线 7 章（严格顺序）

| 章 | 主题 | 核心代码产出 | 学习里程碑 |
|----|------|-------------|-----------|
| ch01 | 引言 | 无代码，笔记为主 | 建立全局认知 |
| ch02 | 处理文本数据 | 分词器、滑动窗口、dataloader | 把文本变成训练样本 |
| ch03 | 注意力机制 | self/multi-head/causal attention | 理解 Transformer 核心 |
| ch04 | 从零搭建 GPT | LayerNorm/GELU/残差/TransformerBlock/GPT 类 | 拼出完整模型 |
| ch05 | 无监督预训练 | 训练循环、loss 曲线、加载 OpenAI 权重、生成文本 | 训练并生成文本 |
| ch06 | 文本分类微调 | 分类头、SST 数据集、评估 | 适配下游任务 |
| ch07 | 指令微调 | 指令数据集、Alpaca 式微调、损失加权 | 对话式 LLM |

### 扩展章依赖关系

```
主线 ch01→ch07（必须顺序）
        │
        ├─ ch05 完成 ──→ ext-distillation（需要训练好的模型）
        ├─ ch05 完成 ──→ ext-quantization（需要模型权重）
        ├─ ch06 完成 ──→ ext-lora（需要微调基础）
        └─ 随时 ─────→ ext-rlhf（纯原理笔记，不依赖代码）
        └─ 随时 ─────→ ext-rag（独立工程 demo，不强依赖主线）
```

**建议节奏**：先走完主线 7 章 → 再按 distillation→lora→quantization 做"模型压缩"主线（三者高度相关）→ RLHF、RAG 随时穿插。

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
