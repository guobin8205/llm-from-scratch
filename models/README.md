# 模型目录

此目录存放训练保存的模型权重（`.pth` / `.pt`）。**权重文件不入库**（已在 `.gitignore` 中排除）。

## 主要权重

| 文件 | 来源 | 说明 |
|------|------|------|
| `gpt_124M_pretrained.pth` | ch05 预训练 / 加载 OpenAI 权重 | 主线训练成果，后续蒸馏/量化复用 |
| `gpt_classification.pth` | ch06 分类微调 | 分类任务权重 |
| `gpt_instruction.pth` | ch07 指令微调 | 指令微调权重 |
| `gpt_student_distilled.pth` | ext-distillation | 蒸馏得到的 student 模型 |

> 首次训练各章后会自动生成对应权重文件。
