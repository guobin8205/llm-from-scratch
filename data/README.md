# 数据目录

此目录存放各章训练/推理用到的数据集。**数据文件不入库**（已在 `.gitignore` 中排除），仅保留获取说明。

## 各章数据

| 章 | 数据集 | 获取方式 |
|----|--------|----------|
| ch02 / ch05 | "The Verdict"（短篇故事，用于预训练 demo） | 原书提供，见 ch02 notebook |
| ch05 | OpenWebText / TinyShakespeare（可选更大语料） | 原书代码自动下载 |
| ch06 | SST-2（斯坦福情感分析） | 通过原书脚本下载 |
| ch07 | Alpaca 指令数据集 | 原书代码自动下载 |

> 下载脚本在各章的 notebook / `solution.py` 中，首次运行会自动拉取。
