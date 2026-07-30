# 附录 A · PyTorch 入门

> **性质**：🔧 前置基础
> **对应官方**：appendix-A
> **状态**：`[x]` 已完成

---

## 本附录目标

全书的基础前置。主线 ch02-ch07 全用 PyTorch，本附录浓缩速查核心：张量 → autograd → nn.Module → 训练循环 → DataLoader → GPU。

## 核心内容

### PyTorch 6 大支柱

| 概念 | 作用 | 关键 API |
|------|------|---------|
| 张量 Tensor | 多维数组，GPU 运算 | `torch.tensor()`, `torch.randn()` |
| autograd | 自动求导 | `requires_grad=True`, `.backward()` |
| nn.Module | 构建模型/层 | `__init__` 定义参数, `forward` 定义计算 |
| 训练循环 | 优化参数 | 5 步：前向→loss→清梯度→反向→更新 |
| DataLoader | 批量加载 | `Dataset.__getitem__`, `DataLoader(batch_size)` |
| GPU 加速 | 并行运算 | `.to(device)`, `torch.device("cuda")` |

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 张量 | Tensor | 多维数组，带梯度追踪 |
| 自动求导 | Autograd | 自动计算梯度 |
| 模块 | Module | 神经网络层的基类 |
| 优化器 | Optimizer | 更新参数（SGD/AdamW） |
| 数据加载器 | DataLoader | 批量/打乱/并行加载数据 |
| 设备 | Device | CPU/GPU |

## 代码走读

- `appendix-A.ipynb` — 可运行速查：6 大支柱各一节，全可执行

### 训练循环 5 步范式（贯穿全书）

```python
for epoch in range(num_epochs):
    pred = model(X)              # 1. 前向
    loss = loss_fn(pred, Y)      # 2. 算 loss
    optimizer.zero_grad()        # 3. 清梯度（重要！）
    loss.backward()              # 4. 反向求梯度
    optimizer.step()             # 5. 更新参数
```

## 思考题

1. 为什么每次反向前要 `zero_grad()`？（梯度会累积，不清零会叠加）
2. `parameters()` 怎么自动收集到所有可训练参数？（nn.Module 递归注册）
3. 为什么模型和数据要在同一 device？

> 💡 没有这些基础，后续 ch02-ch07 会全程卡壳。建议在 ch02 前过一遍。
