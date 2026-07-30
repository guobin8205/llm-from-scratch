# 附录 D · 训练循环增强（Bells & Whistles）

> **性质**：🔬 真做
> **对应官方**：appendix-D
> **依赖**：ch05 后学
> **状态**：`[x]` 已完成

---

## 本附录目标

把 ch05 的朴素训练循环升级成生产级：学习率调度、train/val 划分、早停、梯度裁剪、checkpoint。

## 核心内容

### 生产级训练五件套

| 技术 | 解决什么问题 | 要点 |
|------|------------|------|
| **lr 调度器** | 固定 lr 效果差 | warmup 升温防发散 + cosine 衰减精细收敛 |
| **train/val 划分** | 评估泛化 | 监控 val loss，不只看 train |
| **早停** | 防过拟合 | val loss 连续 N 轮不降就停 |
| **梯度裁剪** | 防梯度爆炸 | `clip_grad_norm_(max_norm=1.0)` |
| **checkpoint** | 防中断丢失 | 定期存 model+optimizer+epoch |

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 学习率调度 | LR scheduling | 训练中动态调 lr |
| 预热 | Warmup | 初期 lr 从小到大 |
| 余弦衰减 | Cosine decay | lr 按余弦曲线降到 min |
| 早停 | Early stopping | val 不改善就停 |
| 梯度裁剪 | Gradient clipping | 限制梯度范数 |
| 检查点 | Checkpoint | 存训练状态 |

## 代码走读

- `appendix-D.ipynb` — 五件套逐个可运行 demo

### CosineLR 核心

```python
class CosineWithWarmup:
    def step(self):
        if step < warmup:  lr = base * step/warmup          # 线性升温
        else:              lr = min + 0.5*(base-min)*(1+cos)  # 余弦衰减
```

## 踩坑记录

- **lr 调度与 ch05 bonus 重叠**：ch05/02-lr-schedulers 已讲过调度器，本附录补充早停/裁剪/checkpoint。
- **梯度爆炸**：GPT 这类深层网络必须裁剪，否则 loss 频繁变 NaN。
- **weights_only**：PyTorch 新版 `torch.load` 默认不信任 pickle，需 `weights_only=True`。

## 思考题

1. 为什么 warmup 能防初期发散？（初始参数随机，大 lr 会让 loss 爆炸）
2. 早停的 patience 设太大/太小会怎样？（太大过拟合，太小可能早停误判）
3. 梯度裁剪的 max_norm 通常设多少？（LLM 一般 1.0）
