# 扩展主题 · 工程化 RAG（检索增强生成）

> **性质**：🏗️ 工程化
> **依赖**：独立 demo，不强依赖主线
> **状态**：`[x]` 已完成

---

## 本章目标

在生成前先检索外部知识库，把相关文档拼进 prompt，让 LLM 回答有据可依——解决幻觉和知识过时。

## 核心内容

### RAG 三步流程

1. **索引**：文档切块 → 算 embedding → 存向量库
2. **检索**：query embedding → 余弦相似度 → top-k 召回
3. **生成**：检索文档拼进 prompt → LLM 生成

### RAG vs 微调

| | RAG | 微调 |
|---|---|---|
| 适合 | 知识频繁更新 | 能力/风格内化 |
| 更新成本 | 换文档即可 | 需重训 |
| 溯源 | ✅ 可引用 | ❌ |
| 幻觉 | 大幅减少 | 不一定 |

### 关键概念（中英对照）

| 中文 | English | 说明 |
|------|---------|------|
| 检索增强生成 | RAG | Retrieval-Augmented Generation |
| 向量库 | Vector store | 存 embedding 的数据库 |
| 嵌入 | Embedding | 文本的向量表示 |
| 余弦相似度 | Cosine similarity | 向量方向相似度 |
| 召回 | Retrieval / Recall | 取 top-k 相关文档 |
| 重排序 | Reranking | 二次精排检索结果 |

## 代码走读

- `ext-rag.ipynb` — 索引 + 检索 + 生成端到端 demo（hash 模拟 embedding）

## 踩坑记录

- **demo 用 hash embedding**：避免 sentence-transformers 依赖，相似度仅供参考。
- **真实工程栈**：`sentence-transformers`(embedding) + `Chroma`/`FAISS`(向量库) + `LangChain`(编排)。
- **chunk 切分**：文档太大需切块，切分策略影响检索质量。

## 思考题

1. RAG 能完全消除幻觉吗？（不能，但大幅减少；检索到错误文档仍可能出错）
2. 如何提升检索质量？（embedding 模型质量、chunk 策略、rerank、hybrid search）
3. RAG 和微调能否结合？（能，先微调能力再 RAG 外挂知识）

> 💡 进阶：多路召回(hybrid)、重排序(rerank)、查询改写(query rewriting)。
