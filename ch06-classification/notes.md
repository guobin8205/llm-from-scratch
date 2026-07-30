# 第 06 章 · 文本分类微调

> **状态**：`[x]` 已完成
>
> **对应原书**：Chapter 06 — Finetuning for Text Classification

---

## 本章目标

把前几章造的「只会续写」的 GPT，改造成能判断**情感**（正面/负面）的专用分类器。这是「预训练 + 微调」范式落地到具体任务的标准操作。

## 核心内容

### 三步改造

1. **换输出头**：`out_head` 从 `emb_dim → 50257`（续写头）替换为 `emb_dim → num_classes`（分类头）
2. **冻结 backbone**：GPT 主体不训练，只训「最后一个 Transformer 块 + final_norm + 分类头」，省算力、防小数据过拟合
3. **分类交叉熵微调**：loss 从续写（整序列）改为分类（最后一个 token 对标签）

### 为什么用最后一个 token？

因果注意力的最后一个 token「看过」整句话，聚合了全部信息，最适合做整体判断。这与 BERT 用 `[CLS]` token 类似。

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 微调 | Fine-tuning | 在预训练模型上用任务数据继续训练 |
| 分类头 | Classification head | 把隐状态映射到类别数的线性层 |
| 冻结 | Freezing | 固定部分参数不训练 |
| 最后 token 分类 | Last-token classification | 用序列末位 token 的输出做分类 |
| 训练/验证/测试集 | Train/Val/Test split | 划分数据评估泛化能力 |
| 准确率 | Accuracy | 预测正确的比例 |

## 代码走读

- `ch06.ipynb` — 可运行 notebook（自造中文情感 demo，loss 0.82→0.04，acc 100%）
- `solution.py` — 整理版：含 `build_classifier`、`train_classifier`、`evaluate_model`，train/val/test 三分

### 核心实现

```python
# 分类头改造
model.out_head = nn.Linear(emb_dim, num_classes)
def classify(model, x): return model(x)[:, -1, :]   # 最后 token

# 冻结 backbone，只解冻最后块 + norm + head
for p in model.parameters(): p.requires_grad = False
for p in model.trf_blocks[-1].parameters(): p.requires_grad = True
```

## 运行结果

- demo 参数：6.83M 总参，可训练 198K（仅 2.9%）
- 训练 loss：0.82 → 0.32（15 epoch）
- 评估：train/val 100%、test 66.7%（小数据 test 仅供参考）

## 踩坑记录

- **数据量太小**：16 条 demo 只能验证流程，test 准确率波动大。真实场景需 SST-2/IMDb（bonus 02）。
- **pad 位置**：分类任务对位置不敏感，右侧 pad 即可（区别于续写需左侧 pad 防信息泄漏）。
- **冻结策略**：直接冻结全部会导致分类头训练不充分，需解冻最后块提供特征变换。

## 思考题 / 扩展

1. 用第一个 token 而非最后一个做分类，效果会怎样？（见 bonus 01）
2. 全量微调 vs 冻结微调，参数量与效果的权衡？
3. 如果类别数变成 10（细粒度情感），需要改哪里？
4. 如何防止小数据过拟合？（早停、dropout、更激进的冻结）

## Bonus

- `bonus/01-additional-experiments.ipynb` — 最后 token vs 首 token / 不同输入长度
- `bonus/02-more-datasets.ipynb` — IMDb 5 万影评分类（含下载+回退）
- `bonus/03-user-interface.ipynb` — Gradio 情感分类交互界面
