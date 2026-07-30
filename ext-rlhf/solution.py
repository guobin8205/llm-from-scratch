"""扩展主题：RLHF 对齐原理（轻量 demo）。

三阶段：SFT → 奖励模型(RM) → PPO。
本 demo 实现奖励模型训练 + 策略梯度简化版，聚焦原理理解。

运行：python ext-rlhf/solution.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 阶段 2：奖励模型
# ---------------------------------------------------------------------------
class RewardModel(nn.Module):
    """给回答打分的模型（输出标量，越高越好）。"""

    def __init__(self, vocab_size=1000, emb_dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.score = nn.Linear(emb_dim, 1)

    def forward(self, token_ids):
        feat = self.emb(token_ids).mean(dim=1)
        return self.score(feat).squeeze(-1)


def reward_loss(chosen_scores, rejected_scores):
    """让好回答分数高于坏回答。"""
    return -F.logsigmoid(chosen_scores - rejected_scores).mean()


def train_reward_model():
    """训练奖励模型 demo。"""
    torch.manual_seed(0)
    rm = RewardModel()
    opt = torch.optim.AdamW(rm.parameters(), lr=0.01)
    chosen = torch.randint(0, 1000, (8, 10))
    rejected = torch.randint(0, 1000, (8, 10))

    print("[阶段2] 训练奖励模型（好回答分数 > 坏回答）:")
    for epoch in range(30):
        opt.zero_grad()
        c, r = rm(chosen), rm(rejected)
        loss = reward_loss(c, r)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == 29:
            margin = (c.mean() - r.mean()).item()
            print(f"  epoch {epoch}: loss {loss.item():.4f}, 分数差 {margin:+.3f}")
    print("  ✓ 奖励模型学会给好回答更高分\n")


# ---------------------------------------------------------------------------
# 阶段 3：PPO / 策略梯度简化
# ---------------------------------------------------------------------------
def policy_gradient_demo():
    """策略梯度 demo：让策略学会选高奖励动作。"""
    torch.manual_seed(0)
    policy = nn.Linear(8, 3)
    opt = torch.optim.AdamW(policy.parameters(), lr=0.05)

    def fake_reward(action_idx):
        rewards = torch.tensor([0.1, 0.3, 1.0])  # 动作 2 奖励最高
        return rewards[action_idx]

    print("[阶段3] 策略梯度 demo（优化策略选高奖励动作）:")
    for epoch in range(50):
        state = torch.randn(4, 8)
        logits = policy(state)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze(-1)
        reward = fake_reward(action)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)
        loss = -(log_prob * reward).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == 49:
            print(f"  epoch {epoch}: 平均动作 {action.float().mean():.2f}（趋近 2=高奖励）")
    print("  ✓ 策略逐渐偏向高奖励动作\n")


def main():
    print("=== 扩展主题：RLHF 对齐原理 ===\n")
    print("三阶段: SFT(监督微调) → RM(奖励模型) → PPO(强化学习)\n")
    print("（SFT 已在 ch07 完成，这里演示 RM 和 PPO 原理）\n")
    train_reward_model()
    policy_gradient_demo()
    print("[对比] RLHF 复杂(4模型) vs DPO(ch07) 简化(2模型)")
    print("[done] 真实实现用 trl 库的 PPOTrainer；本 demo 聚焦原理。")


if __name__ == "__main__":
    main()
