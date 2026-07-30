# 扩展主题 · 模型量化（Quantization）

> **性质**：🔬 真做
> **依赖**：ch05 模型权重
> **状态**：`[x]` 已完成

---

## 本章目标

把 fp32 权重压成 int8/int4，省 75%~87.5% 显存，推理更快，质量可控——部署大模型到消费级硬件的关键。

## 核心内容

### 量化精度对比

| 精度 | 每参数字节 | 7B 模型显存 | 误差 |
|------|----------|-----------|------|
| fp32 | 4 | 28 GB | 基准 |
| fp16 | 2 | 14 GB | 极小 |
| **int8** | 1 | 7 GB | 几乎无损 |
| **int4** | 0.5 | 3.5 GB | 可接受 |

### 对称量化原理

`scale = max(|W|) / 127`，量化 `q = round(W / scale)`，反量化 `W ≈ q × scale`。

### QLoRA（量化 + LoRA）

4bit 量化基座(冻结) + LoRA 旁路(训练) = 单卡微调大模型。与附录 E 呼应。

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 后训练量化 | PTQ (Post-Training Quantization) | 训练后直接量化 |
| 量化感知训练 | QAT (Quantization-Aware Training) | 训练时模拟量化 |
| 对称量化 | Symmetric quantization | 以 0 为中心 |
| 缩放系数 | Scale | fp32↔int 的转换因子 |
| NF4 | NormalFloat 4-bit | QLoRA 用的正态分布最优 4bit |

## 代码走读

- `ext-quantization.ipynb` — int8/int4 量化 + 误差分析 + QLoRA 衔接

## 思考题

1. 为什么 int8 误差远小于 int4？（分辨率：127 vs 7 个离散值）
2. 量化为什么能加速？（整数运算比浮点快，内存带宽减半）
3. QLoRA 为何能单卡微调 70B？（4bit 基座占 35GB + LoRA 极小参数）

> 💡 真实工程用 `bitsandbytes` / `GPTQ` / `AWQ` 库做生产级量化。
