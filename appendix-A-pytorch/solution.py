"""附录 A 整理版：PyTorch 入门速查。

把 notebook 的核心浓缩成一个可运行脚本：张量 → autograd → nn.Module → 训练循环 → DataLoader。
用一个「拟合 y=2x+1」的小任务串起全部概念。

运行：python appendix-A-pytorch/solution.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def main():
    torch.manual_seed(42)
    print("=== 附录 A：PyTorch 入门速查 ===\n")

    # 1. 张量
    print("[1] 张量")
    x = torch.randn(3, 4)
    print(f"  shape={tuple(x.shape)}, dtype={x.dtype}")

    # 2. autograd
    print("\n[2] autograd")
    w = torch.tensor(3.0, requires_grad=True)
    y = w ** 2
    y.backward()
    print(f"  d(x²)/dx at x=3 = {w.grad}（应为 6）")

    # 3. nn.Module + 训练循环 + DataLoader（用拟合任务串起来）
    print("\n[3] nn.Module + 训练循环 + DataLoader")
    print("  任务：拟合 y = 2x + 1")

    class LinearDataset(Dataset):
        def __init__(self, n=100):
            self.x = torch.linspace(-1, 1, n).unsqueeze(1)
            self.y = 2 * self.x + 1 + 0.1 * torch.randn(n, 1)
        def __len__(self): return len(self.x)
        def __getitem__(self, i): return self.x[i], self.y[i]

    dataloader = DataLoader(LinearDataset(100), batch_size=16, shuffle=True)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(50):
        for x_batch, y_batch in dataloader:
            pred = model(x_batch)                          # 前向
            loss = nn.functional.mse_loss(pred, y_batch)   # loss
            optimizer.zero_grad()                          # 清梯度
            loss.backward()                                # 反向
            optimizer.step()                               # 更新
        if epoch % 10 == 0:
            print(f"  epoch {epoch}: loss {loss.item():.4f}")

    print(f"\n  学到: w={model.weight.item():.3f}（应为2）, b={model.bias.item():.3f}（应为1）")

    # 4. GPU
    print("\n[4] GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  可用设备: {device}")
    print("  .to(device) 统一模型和数据设备")


if __name__ == "__main__":
    main()
