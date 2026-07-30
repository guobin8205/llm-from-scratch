# 扩展主题 · 知识蒸馏（Knowledge Distillation）

> **性质**：🔬 真做
> **依赖**：ch05 训练好的 GPT-124M
> **状态**：`[x]` 已完成

---

## 本章目标

用大模型（teacher）的输出指导小模型（student）训练，让 student 用更少参数接近 teacher 效果——模型压缩经典技术。

## 核心内容

### 蒸馏原理

teacher 软标签（高温 softmax）比 hard label（one-hot）信息更丰富，透露「类间相似度」等暗知识。

### 蒸馏损失：KL 散度 + 温度

$$\mathcal{L} = T^2 \cdot \text{KL}(\text{softmax}(z_s/T) \| \text{softmax}(z_t/T))$$

- **温度 T**：softmax 时除以 T，分布更平滑。T 越高信息越丰富但越柔和
- **乘 T²**：补偿温度导致的梯度缩小
- 通常 T=2~4

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 教师/学生模型 | Teacher / Student | 大模型指导小模型 |
| 软标签 | Soft label | 高温 softmax 的概率分布 |
| 硬标签 | Hard label | one-hot 真实标签 |
| 温度 | Temperature | softmax 软化程度 |
| KL 散度 | KL divergence | 两分布差异度量 |
| 暗知识 | Dark knowledge | 软标签含的类间关系 |

## 代码走读

- `ext-distillation.ipynb` — 蒸馏损失实现 + GPT 12层→2层 demo（KL loss 持续下降）

## 思考题

1. 为什么软标签比 hard label 信息更丰富？（含所有类的相对概率）
2. 温度太高/太低会怎样？（太高几乎均匀无信息；太低接近 one-hot）
3. 蒸馏 vs 量化，各自适合什么场景？（蒸馏换模型结构变小；量化换存储精度降低）

> 💡 应用：把大模型部署到手机/边缘设备，或线上推理加速。
