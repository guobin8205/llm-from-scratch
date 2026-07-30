"""扩展主题：检索增强生成（RAG）。

文档库 embedding + 余弦相似度检索 + 拼接生成。
demo 用 hash 模拟 embedding，避免 sentence-transformers 依赖。

运行：python ext-rag/solution.py
"""

import torch
import torch.nn.functional as F

# 知识库
DOCUMENTS = [
    "GPT 是一种基于 Transformer 的自回归语言模型，由 OpenAI 提出",
    "注意力机制让模型动态关注输入序列的不同部分，是 Transformer 的核心",
    "预训练是在大规模无标注文本上训练，学习语言的通用表示",
    "微调是在预训练模型基础上，用有标注数据继续训练以适应特定任务",
    "BERT 使用双向 Transformer 编码器，适合理解类任务如分类",
    "LoRA 通过低秩矩阵适配实现参数高效微调，只训练不到 1% 的参数",
    "量化把模型权重从 fp32 压缩到 int8/int4，大幅减少显存占用",
    "RLHF 通过人类反馈的强化学习，让模型输出更符合人类偏好",
]

EMB_DIM = 64


def embed(text, dim=EMB_DIM):
    """文本 → 向量（demo 用 hash 模拟，真实用 sentence-transformers）。"""
    torch.manual_seed(hash(text) % (2**31))
    return torch.randn(dim)


def build_index(documents):
    """索引：把所有文档 embed 成向量库。"""
    return torch.stack([embed(d) for d in documents])


def retrieve(query, doc_embeddings, k=2):
    """检索：query embedding → 余弦相似度 → top-k。"""
    q_emb = embed(query)
    sims = F.cosine_similarity(q_emb.unsqueeze(0), doc_embeddings)
    scores, idx = sims.topk(k)
    return [(DOCUMENTS[i], scores[n].item()) for n, i in enumerate(idx)]


def rag_answer(query, doc_embeddings, k=2):
    """完整 RAG：检索 → 拼 prompt（交给 LLM 生成）。"""
    sources = retrieve(query, doc_embeddings, k=k)
    context = "\n".join(f"[{i+1}] {d}" for i, (d, _) in enumerate(sources))
    prompt = (f"根据以下参考资料回答问题。\n\n参考资料：\n{context}\n\n"
              f"问题：{query}\n回答：")
    return prompt, sources


def main():
    print("=== 扩展主题：检索增强生成 (RAG) ===\n")

    doc_emb = build_index(DOCUMENTS)
    print(f"[1] 知识库: {len(DOCUMENTS)} 篇文档, embedding 维度 {EMB_DIM}")

    queries = ["什么是注意力机制", "怎么省显存", "LoRA 是什么"]
    print(f"\n[2] 检索测试:")
    for q in queries:
        print(f"\n  查询: {q!r}")
        for doc, score in retrieve(q, doc_emb, k=2):
            print(f"    [{score:+.3f}] {doc[:30]}...")

    print(f"\n[3] 端到端 RAG demo:")
    query = "怎么让微调更省参数"
    prompt, sources = rag_answer(query, doc_emb, k=2)
    print(f"  问题: {query}")
    print(f"  检索到 {len(sources)} 篇文档作为依据")
    print(f"  → 拼 prompt 交给 LLM 生成有据可依的回答")
    print(f"\n[done] 真实工程用 sentence-transformers + Chroma + LangChain。")


if __name__ == "__main__":
    main()
