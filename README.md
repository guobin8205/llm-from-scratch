# 从零构建大模型（LLM from Scratch）

> 跟着 Sebastian Raschka《Build a Large Language Model (from Scratch)》一书，从零用 PyTorch 实现并训练一个 GPT 模型，并延伸到现代 LLM 架构与进阶主题。
>
> 覆盖全链路：**文本数据 → 注意力机制 → GPT 模型 → 预训练 → 微调**，外加附录、bonus 与 4 个扩展主题。

![status](https://img.shields.io/badge/状态-全部完成-brightgreen) ![chapters](https://img.shields.io/badge/主线-7章+附录A/D/E-blue) ![bonus](https://img.shields.io/badge/bonus-34个notebook-orange) ![python](https://img.shields.io/badge/Python-3.14-yellow)

---

## 📚 目录导航（可点击直达）

> ✅ 全部内容已完成，代码均可运行。完整路线图见 [`docs/roadmap.md`](docs/roadmap.md)。

### 📘 主线（原书 7 章，严格顺序）

| 章 | 目录 | 主题 | 状态 |
|----|------|------|:----:|
| — | [`appendix-A-pytorch/`](appendix-A-pytorch/) | **PyTorch 入门**（前置基础） | ✅ |
| 1 | [`ch01-introduction/`](ch01-introduction/) | 引言：理解 LLM | ✅ |
| 2 | [`ch02-text-data/`](ch02-text-data/) | 处理文本数据（分词/BPE/滑动窗口） | ✅ |
| 3 | [`ch03-attention/`](ch03-attention/) | 注意力机制（self/causal/multi-head） | ✅ |
| 4 | [`ch04-gpt-from-scratch/`](ch04-gpt-from-scratch/) | 从零搭建 GPT-2 124M | ✅ |
| 5 | [`ch05-pretraining/`](ch05-pretraining/) | 预训练 + 加载 OpenAI 权重 | ✅ |
| 6 | [`ch06-classification/`](ch06-classification/) | 文本分类微调 | ✅ |
| 7 | [`ch07-instruction/`](ch07-instruction/) | 指令微调（SFT） | ✅ |

### 🔬 附录 D / E（选学）

| 目录 | 主题 | 亮点 | 状态 |
|------|------|------|:----:|
| [`appendix-D-training-loop/`](appendix-D-training-loop/) | 训练循环增强 | lr调度/早停/梯度裁剪/checkpoint | ✅ |
| [`appendix-E-lora/`](appendix-E-lora/) | LoRA 参数高效微调 | ⭐只训 0.54% 参数 | ✅ |

### ⭐ 各章 Bonus（34 个 notebook）

<details>
<summary><b>ch04：8 种注意力变体</b>（点击展开）</summary>

| 文件 | 主题 | 参考模型 |
|------|------|---------|
| [`01-kv-cache`](ch04-gpt-from-scratch/bonus/01-kv-cache.ipynb) | KV 缓存（推理加速 ~5×） | — |
| [`02-gqa`](ch04-gpt-from-scratch/bonus/02-gqa.ipynb) | 分组查询注意力 | Llama-3 |
| [`03-mla`](ch04-gpt-from-scratch/bonus/03-mla.ipynb) | 多头潜在注意力（省 88% 缓存） | DeepSeek |
| [`04-swa`](ch04-gpt-from-scratch/bonus/04-swa.ipynb) | 滑动窗口注意力 | Mistral |
| [`05-moe`](ch04-gpt-from-scratch/bonus/05-moe.ipynb) | 混合专家 | Mixtral |
| [`06-deltanet`](ch04-gpt-from-scratch/bonus/06-deltanet.ipynb) | 门控 DeltaNet | Nemotron |
| [`07-dsa`](ch04-gpt-from-scratch/bonus/07-dsa.ipynb) | 差分注意力 | Sakana AI |
| [`08-kv-sharing`](ch04-gpt-from-scratch/bonus/08-kv-sharing.ipynb) | 跨层 KV 共享 | YOCO |

</details>

<details>
<summary><b>ch05：现代 LLM zoo + 训练工具（18 个）</b>（点击展开）</summary>

**训练工具类：** [`01-gutenberg`](ch05-pretraining/bonus/01-gutenberg.ipynb)（大语料）｜[`02-lr-schedulers`](ch05-pretraining/bonus/02-lr-schedulers.ipynb)（调度器）｜[`03-hparam-tuning`](ch05-pretraining/bonus/03-hparam-tuning.ipynb)（超参搜索）｜[`04-training-speed`](ch05-pretraining/bonus/04-training-speed.ipynb)（混合精度/梯度累积）｜⭐[`05-muon`](ch05-pretraining/bonus/05-muon.ipynb)（Muon 优化器）｜[`06-alt-weight-loading`](ch05-pretraining/bonus/06-alt-weight-loading.ipynb)｜[`07-mem-efficient-loading`](ch05-pretraining/bonus/07-mem-efficient-loading.ipynb)｜[`08-extending-tokenizers`](ch05-pretraining/bonus/08-extending-tokenizers.ipynb)｜[`09-user-interface`](ch05-pretraining/bonus/09-user-interface.ipynb)

**架构转换类：** ⭐[`10-gpt-to-llama`](ch05-pretraining/bonus/10-gpt-to-llama.ipynb)（RoPE/RMSNorm/SwiGLU）｜[`11-qwen3`](ch05-pretraining/bonus/11-qwen3.ipynb)｜[`12-gemma3`](ch05-pretraining/bonus/12-gemma3.ipynb)｜[`13-gemma4`](ch05-pretraining/bonus/13-gemma4.ipynb)｜[`14-olmo3`](ch05-pretraining/bonus/14-olmo3.ipynb)｜[`15-tiny-aya`](ch05-pretraining/bonus/15-tiny-aya.ipynb)｜[`16-ch05-with-llms`](ch05-pretraining/bonus/16-ch05-with-llms.ipynb)｜[`17-llama3-standalone`](ch05-pretraining/bonus/17-llama3-standalone.ipynb)｜[`18-llm-zoo-overview`](ch05-pretraining/bonus/18-llm-zoo-overview.ipynb)

</details>

<details>
<summary><b>ch03 / ch06 / ch07 bonus</b>（点击展开）</summary>

- **ch03**：[`efficient-mha`](ch03-attention/bonus/efficient-mha.ipynb)｜[`understanding-buffers`](ch03-attention/bonus/understanding-buffers.ipynb)
- **ch06**：[`01-additional-experiments`](ch06-classification/bonus/01-additional-experiments.ipynb)（最后token vs 首token）｜[`02-more-datasets`](ch06-classification/bonus/02-more-datasets.ipynb)（IMDb）｜[`03-user-interface`](ch06-classification/bonus/03-user-interface.ipynb)
- **ch07**：⭐[`01-dpo`](ch07-instruction/bonus/01-dpo.ipynb)（直接偏好优化）｜[`02-model-evaluation`](ch07-instruction/bonus/02-model-evaluation.ipynb)｜[`03-user-interface`](ch07-instruction/bonus/03-user-interface.ipynb)

</details>

### 🚀 扩展主题（官方无，基于主线延伸）

| 目录 | 性质 | 主题 | 亮点 | 状态 |
|------|------|------|------|:----:|
| [`ext-quantization/`](ext-quantization/) | 🔬 真做 | 模型量化 | int8/int4，省 75% 显存，QLoRA 衔接 | ✅ |
| [`ext-distillation/`](ext-distillation/) | 🔬 真做 | 知识蒸馏 | KL散度+温度，teacher→student | ✅ |
| [`ext-rag/`](ext-rag/) | 🏗️ 工程化 | 检索增强生成 | embedding检索+拼prompt生成 | ✅ |
| [`ext-rlhf/`](ext-rlhf/) | 📖 原理 | RLHF 对齐 | SFT→奖励模型→PPO，对比 DPO | ✅ |

---

## ⚙️ 环境配置

### 当前环境（已验证可用）

本项目直接使用系统 Python，**无需虚拟环境**：

- **Python 3.14.0**
- **PyTorch 2.13.0 + cu130**（CUDA 可用）
- **GPU**: NVIDIA GeForce RTX 5070 Ti Laptop 12GB
- 其他依赖：numpy、tiktoken、pandas、gradio、jupyter

### 从零配置（如换机器）

```bash
# 1. 安装 PyTorch（CUDA 版，见 https://pytorch.org/get-started/locally/）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. 安装其余依赖
pip install tiktoken pandas gradio jupyter ipykernel

# 3. 验证 GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 📂 项目结构

```
llm-from-scratch/
├── appendix-A-pytorch/          # PyTorch 入门
├── ch01-introduction/ … ch07-instruction/   # 主线 7 章
├── appendix-D-training-loop/    # 训练循环增强
├── appendix-E-lora/             # LoRA
├── ext-quantization/ ext-distillation/ ext-rag/ ext-rlhf/   # 扩展 4 主题
├── src/gpt/                     # 跨章节复用的核心模块（data/attention/model）
├── data/                        # 数据集（the-verdict.txt 等）
├── models/                      # 训练权重（gitignore）
└── docs/                        # 路线图与设计文档
```

每章目录约定：
- `notes.md` — 中文笔记（统一模板）
- `*.ipynb` — 可运行 notebook（含中文注释，代码全部验证通过）
- `solution.py` — 整理后的本章代码（端到端可运行）

---

## 📝 学习约定

- **设备无关**：所有训练代码统一用 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- **中文注释**：代码逐行解释"为什么这么写"
- **可运行**：所有 notebook 和 solution.py 均在 CUDA 环境执行验证通过
- **笔记模板**：每章 `notes.md` 含 目标 / 核心概念（中英对照）/ 代码走读 / 运行结果 / 踩坑记录 / 思考题

---

## 📖 参考资料

- 原书：Sebastian Raschka, *Build a Large Language Model (from Scratch)*, Manning, 2024.
- 原书官方代码：<https://github.com/rasbt/LLMs-from-scratch>
- 详细路线图：[`docs/roadmap.md`](docs/roadmap.md)
