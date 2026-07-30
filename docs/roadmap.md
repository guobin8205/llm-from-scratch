# 学习路线图

> 本项目是《Build a Large Language Model (from Scratch)》（Sebastian Raschka）的课程代码学习仓库。
> 目标：从零用 PyTorch 实现并训练一个 GPT 模型，覆盖 **数据 → 注意力 → 模型 → 预训练 → 微调** 全链路。

## 整体结构

项目分为 **主线 7 章** + **扩展 5 章**：

- 📘 **主线（ch01–ch07）**：原书核心，严格顺序学习。每章产出 `notebook + solution.py + 中文笔记`。
- 🚀 **扩展（ext-*）**：原书之外的进阶主题，学完主线后按依赖关系选做。

---

## 第一阶段：主线（必须顺序）

| 章 | 目录 | 主题 | 里程碑 |
|----|------|------|--------|
| 1 | `ch01-introduction/` | 引言：理解 LLM | 建立全局认知 |
| 2 | `ch02-text-data/` | 处理文本数据 | 能把文本变成训练样本 |
| 3 | `ch03-attention/` | 注意力机制 | 理解 Transformer 核心 |
| 4 | `ch04-gpt-from-scratch/` | 从零搭建 GPT | 拼出完整模型 |
| 5 | `ch05-pretraining/` | 无监督预训练 | 训练并生成文本 |
| 6 | `ch06-classification/` | 文本分类微调 | 适配下游任务 |
| 7 | `ch07-instruction/` | 指令微调 | 对话式 LLM |

## 第二阶段：扩展（按依赖选做）

```
ch05 完成 ──→ ext-distillation（知识蒸馏：teacher→student）
ch05 完成 ──→ ext-quantization（int8/int4 量化）
ch06 完成 ──→ ext-lora（LoRA/QLoRA 参数高效微调）
随时      ──→ ext-rlhf（RLHF 原理，轻量演示）
随时      ──→ ext-rag（工程化 RAG，独立 demo）
```

**建议节奏**：主线 7 章 → distillation→lora→quantization（模型压缩主线）→ RLHF、RAG 穿插。

---

## 扩展主题性质说明

| 扩展章 | 性质 | 说明 |
|--------|------|------|
| `ext-distillation` | 🔬 真做 | 知识蒸馏，KL 散度 + 温度 |
| `ext-lora` | 🔬 真做 | LoRA/QLoRA 参数高效微调 |
| `ext-quantization` | 🔬 真做 | int8/int4 量化 |
| `ext-rlhf` | 📖 轻量原理 | SFT→RM→PPO 三阶段，笔记 + 伪代码为主 |
| `ext-rag` | 🏗️ 工程化 | LangChain + 向量库，独立 demo |

---

## 每章目录约定

```
chXX-xxx/
├── notes.md          # 中文笔记（统一模板）
├── chXX.ipynb        # 可运行 notebook（含中文注释）
└── solution.py       # 整理后的本章代码
```

## 进度追踪

学习进度记录在各章目录的 `notes.md` 顶部。开始学习某章时，在该文件首行标注状态：
- `[ ]` 待学习
- `[~]` 学习中
- `[x]` 已完成
