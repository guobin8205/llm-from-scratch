# 附录 E · LoRA 参数高效微调

> **性质**：🔬 真做
> **对应官方**：appendix-E
> **依赖**：ch06/ch07 后学
> **状态**：`[x]` 已完成

---

## 本附录目标

用 LoRA（低秩适配）把可训练参数压到 **< 1%**，却达到接近全量微调的效果——工业界最主流的微调技术。

## 核心内容

### 原理

全量微调要更新权重 W（d×k）。LoRA 冻结 W，只训练低秩矩阵 A、B：

$$\Delta W = B \cdot A \quad (B: d{\times}r,\ A: r{\times}k,\ r \ll d,k)$$

- **B 零初始化**：训练初期 ΔW=0，不破坏原模型
- **A 正常初始化**：开始训练后逐渐学习
- **α/r 缩放**：alpha 控制更新强度，通常 alpha=2r

### 参数对比（emb_dim=768, r=8）

| | 全量微调 | LoRA |
|---|---|---|
| 可训练参数 | 590K | 12K（**2%**）|
| 整个 GPT（小配置） | 100% | **0.54%** |

### 优势

1. **省显存**：单卡可微调 70B 模型
2. **可叠加**：不同任务训不同 LoRA，切换即用，不互相干扰
3. **不破坏原模型**：原权重冻结，LoRA 可随时卸载

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 低秩适配 | Low-Rank Adaptation | 用低秩矩阵近似权重变化 |
| 参数高效微调 | PEFT | Parameter-Efficient Fine-Tuning |
| 秩 | Rank | LoRA 的 r，控制瓶颈维度 |
| 缩放因子 | Scaling | alpha/r，控制 LoRA 更新强度 |
| 旁路 | Side branch / adapter | 冻结主干，旁路训练 |

## 代码走读

- `appendix-E.ipynb` — LoRA 层实现 + 替换 GPT 全部 Linear + 分类微调 demo

### 核心实现

```python
class LoRALayer(nn.Module):
    def __init__(self, layer, r=8, alpha=16):
        for p in layer.parameters(): p.requires_grad = False  # 冻结原层
        self.A = nn.Parameter(torch.randn(in_f, r) * 0.01)   # A 正常
        self.B = nn.Parameter(torch.zeros(r, out_f))          # B 零初始化
    def forward(self, x):
        return self.layer(x) + self.scaling * (x @ (self.A @ self.B))
```

## 踩坑记录

- **必须先冻结所有参数再替换**：否则 embedding/out_head 仍可训练，达不到 <1% 的效果。
- **B 零初始化是关键**：保证训练开始时模型行为与原模型完全一致。
- **只替换 trf_blocks 的 Linear**：embedding 和任务头（分类头）保持原始可训练状态。

## 思考题

1. 为什么 ΔW 是低秩的？（学新任务所需的变化在低维子空间）
2. r 设大设小的权衡？（大→接近全量微调效果但参数多；小→省但可能欠拟合）
3. LoRA 为什么能叠加多个任务？（不同 LoRA 互不干扰，切换即可）
4. QLoRA（量化+LoRA）相比 LoRA 有何优势？（4bit 量化进一步省显存）
