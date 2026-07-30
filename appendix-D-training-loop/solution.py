"""附录 D 整理版：训练循环增强（生产级五件套）。

把 ch05 的朴素循环升级：lr 调度器 + train/val 划分 + 早停 + 梯度裁剪 + checkpoint。

运行：python appendix-D-training-loop/solution.py
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------------
# 1. 学习率调度器：warmup + cosine decay
# ---------------------------------------------------------------------------
class CosineWithWarmup:
    """warmup 线性升温 → cosine 余弦降到 min_lr。"""

    def __init__(self, optimizer, num_warmup, num_training, base_lr,
                 min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.num_warmup = num_warmup
        self.num_training = num_training
        self.base_lr = base_lr
        self.min_lr = base_lr * min_lr_ratio
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.num_warmup:
            lr = self.base_lr * self.step_num / self.num_warmup
        else:
            progress = ((self.step_num - self.num_warmup) /
                        (self.num_training - self.num_warmup))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


# ---------------------------------------------------------------------------
# 2. 带早停 + 梯度裁剪的训练循环
# ---------------------------------------------------------------------------
def train_with_early_stopping(model, train_loader, val_loader, optimizer,
                              num_epochs, patience=5, max_grad_norm=1.0,
                              scheduler=None):
    """生产级训练循环：lr 调度 + 早停 + 梯度裁剪。"""
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # 训练
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x).squeeze(), y.float())
            loss.backward()
            # 梯度裁剪（防爆炸）
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # 验证
        model.eval()
        val_loss = 0
        m = 0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += nn.functional.mse_loss(
                    model(x).squeeze(), y.float()).item()
                m += 1
        val_loss /= m

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1}: val {val_loss:.4f}"
              f"{' *' if improved else ''}")
        if epochs_no_improve >= patience:
            print(f"早停：val loss 连续 {patience} 轮未改善。")
            break
    return best_val_loss


# ---------------------------------------------------------------------------
# 3. Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"]


def main():
    torch.manual_seed(0)
    print("=== 附录 D：训练循环增强 ===\n")

    # 造数据
    X = torch.randn(100, 4)
    Y = (X.sum(1) > 0).long()
    train_dl = DataLoader(TensorDataset(X[:80], Y[:80]), batch_size=16, shuffle=True)
    val_dl = DataLoader(TensorDataset(X[80:], Y[80:]), batch_size=16)

    model = nn.Linear(4, 1)
    base_lr = 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    # lr 调度：50 步 warmup + 200 步 cosine
    scheduler = CosineWithWarmup(optimizer, num_warmup=10,
                                 num_training=200, base_lr=base_lr)

    print("[训练] lr调度 + 早停(patience=3) + 梯度裁剪(1.0):\n")
    train_with_early_stopping(
        model, train_dl, val_dl, optimizer,
        num_epochs=20, patience=3, max_grad_norm=1.0, scheduler=scheduler,
    )

    # checkpoint demo
    print("\n[Checkpoint] 保存/加载测试:")
    save_checkpoint(model, optimizer, epoch=5, path="data/ckpt_demo.pt")
    model2 = nn.Linear(4, 1)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    ep = load_checkpoint(model2, opt2, "data/ckpt_demo.pt")
    print(f"  恢复到 epoch {ep}，权重一致: "
          f"{torch.allclose(model.state_dict()['weight'], model2.state_dict()['weight'])}")
    import os
    os.remove("data/ckpt_demo.pt")


if __name__ == "__main__":
    main()
