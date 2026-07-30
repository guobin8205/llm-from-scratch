# 扩展主题 · RLHF 对齐原理

> **性质**：📖 轻量原理
> **依赖**：不依赖代码，随时可学
> **状态**：`[x]` 已完成

---

## 本章目标

理解 RLHF（基于人类反馈的强化学习）——让 LLM 回答符合人类偏好。ChatGPT 的核心技术，比 ch07 的 DPO 更完整但更复杂。

## 核心内容

### RLHF 三阶段

```
1. SFT（监督微调）  → ch07，让模型学会遵循指令
2. RM（奖励模型）   → 训练一个能给回答打分的模型
3. PPO（强化学习）  → 用 RM 的分数当奖励，优化策略模型
```

### 阶段 2：奖励模型

分类头改打分头（输出标量）。损失让 chosen 分数高于 rejected：
$$\mathcal{L}_{RM} = -\log\sigma(r(x,y_c) - r(x,y_r))$$

### 阶段 3：PPO

策略梯度 $\nabla J = \mathbb{E}[r(x,y) \cdot \nabla\log\pi(y|x)]$，奖励高的回答增大其概率。PPO 还加 KL 惩罚防止策略偏离 SFT 太远。

### RLHF vs DPO

| | RLHF | DPO (ch07) |
|---|---|---|
| 阶段 | SFT→RM→PPO | SFT→直接优化 |
| 需要奖励模型？ | ✅ 要单独训练 | ❌ |
| 用强化学习？ | ✅ PPO | ❌ 纯监督 |
| 复杂度 | 高（4 个模型） | 低（2 个） |

> DPO 证明偏好数据可直接优化策略，跳过 RM 和 PPO，是 2023 年重要突破。

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 人类反馈强化学习 | RLHF | Reinforcement Learning from Human Feedback |
| 奖励模型 | Reward Model | 给回答打分的模型 |
| 策略梯度 | Policy Gradient | RL 的优化方向 |
| 对齐 | Alignment | 让模型符合人类偏好 |
| 偏好数据 | Preference data | chosen vs rejected 对 |

## 代码走读

- `ext-rlhf.ipynb` — 奖励模型训练 + 策略梯度简化 demo

## 思考题

1. RLHF 为何需要 4 个模型？（策略、参考、奖励、价值模型，PPO 用）
2. PPO 里的 KL 惩罚起什么作用？（防止策略偏离 SFT 模型太远，保持通用性）
3. DPO 为什么能跳过 RM？（数学证明偏好数据可直接推出最优策略）

> 💡 真实实现用 `trl` 库的 `PPOTrainer`；本附录聚焦原理理解。
