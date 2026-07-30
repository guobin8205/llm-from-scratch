# 从零构建大模型（LLM from Scratch）

> 跟着 Sebastian Raschka《Build a Large Language Model (from Scratch)》一书，从零用 PyTorch 实现并训练一个 GPT 模型。
>
> 覆盖全链路：**文本数据 → 注意力机制 → GPT 模型 → 预训练 → 微调**，外加 5 个进阶扩展主题。

---

## 📚 学习路线

### 📘 主线（原书 7 章，严格顺序）

| 章 | 目录 | 主题 |
|----|------|------|
| 1 | [`ch01-introduction/`](ch01-introduction/) | 引言：理解 LLM |
| 2 | [`ch02-text-data/`](ch02-text-data/) | 处理文本数据（分词/数据集/dataloader） |
| 3 | [`ch03-attention/`](ch03-attention/) | 注意力机制（self/multi-head/causal） |
| 4 | [`ch04-gpt-from-scratch/`](ch04-gpt-from-scratch/) | 从零搭建 GPT |
| 5 | [`ch05-pretraining/`](ch05-pretraining/) | 无监督预训练 + 加载 OpenAI 权重 |
| 6 | [`ch06-classification/`](ch06-classification/) | 文本分类微调 |
| 7 | [`ch07-instruction/`](ch07-instruction/) | 指令微调 |

### 🚀 扩展（进阶主题，学完主线后选做）

| 扩展章 | 性质 | 主题 |
|--------|------|------|
| [`ext-distillation/`](ext-distillation/) | 🔬 真做 | 知识蒸馏（teacher→student） |
| [`ext-lora/`](ext-lora/) | 🔬 真做 | LoRA/QLoRA 参数高效微调 |
| [`ext-quantization/`](ext-quantization/) | 🔬 真做 | int8/int4 模型量化 |
| [`ext-rlhf/`](ext-rlhf/) | 📖 原理 | RLHF 对齐（轻量演示） |
| [`ext-rag/`](ext-rag/) | 🏗️ 工程化 | RAG（LangChain + 向量库） |

详细路线见 [`docs/roadmap.md`](docs/roadmap.md)。

---

## ⚙️ 环境配置

### 目标环境

- **Python 3.12**（⚠️ 系统 Python 3.14 与 PyTorch CUDA 版不兼容，必须用 3.12）
- **PyTorch（CUDA 版）** — GPU: RTX 5070 Ti Laptop 12GB

### 步骤

```bash
# 1. 安装 Python 3.12（若尚未安装）
#    Windows 推荐用 winget：
#    winget install Python.Python.3.12
#    或从 https://www.python.org/downloads/ 下载

# 2. 进入项目目录
cd E:\repos\python\llm-from-scratch

# 3. 创建虚拟环境（务必指定 3.12）
py -3.12 -m venv .venv

# 4. 激活虚拟环境
.venv\Scripts\activate        # Windows (Git Bash: source .venv/Scripts/activate)

# 5. 升级 pip
python -m pip install --upgrade pip

# 6. 安装 PyTorch（CUDA 版，通过官方索引）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 若驱动较新支持 cu124，可改用 cu124 索引

# 7. 安装其余依赖
pip install -r requirements.txt

# 8. 验证 GPU 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 验证结果

最后一行应输出：
```
CUDA: True | NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

---

## 📂 项目结构

```
llm-from-scratch/
├── ch01-introduction/ … ch07-instruction/   # 主线 7 章
├── ext-distillation/ ext-lora/ …            # 扩展 5 章
├── data/      # 数据集（gitignore，含获取说明）
├── models/    # 训练权重（gitignore）
├── src/gpt/   # 跨章节复用的核心模块
└── docs/      # 路线图与设计文档
```

每章目录约定：
- `notes.md` — 中文笔记（统一模板）
- `chXX.ipynb` — 可运行 notebook（含中文注释）
- `solution.py` — 整理后的本章代码

---

## 📝 学习约定

- **设备无关**：所有训练代码统一用 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`。
- **中文注释**：代码逐行注释解释"为什么这么写"。
- **笔记模板**：每章 `notes.md` 含 目标 / 核心概念 / 代码走读 / 运行结果 / 踩坑记录 / 思考题。

---

## 📖 参考资料

- 原书：Sebastian Raschka, *Build a Large Language Model (from Scratch)*, Manning, 2024.
- 原书官方代码：<https://github.com/rasbt/LLMs-from-scratch>
- PyTorch CUDA 安装指南：<https://pytorch.org/get-started/locally/>
