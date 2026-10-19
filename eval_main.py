
import rag
import numpy as np
import jieba

# 增加了几道专有名词题，这是 BM25 的强项
EVAL_CASES = [
    {"query": "FAISS 是哪家公司开源的？", "keywords": ["Facebook"]},
    {"query": "Chroma 默认使用什么作为持久化存储？", "keywords": ["SQLite"]},
    {"query": "Milvus 依赖了哪些组件？", "keywords": ["etcd", "MinIO"]},
    {"query": "Qdrant 是用什么语言编写的？", "keywords": ["Rust"]},
    {"query": "pgvector 是哪个数据库的扩展？", "keywords": ["PostgreSQL"]},
    {"query": "十亿级数据必须使用什么方案？", "keywords": ["分布式方案"]},
    {"query": "生产系统通常采用哪两种检索方式结合？", "keywords": ["BM25", "混合检索"]},
    {"query": "如果要和现有的 SQL 生态融合选哪个？", "keywords": ["pgvector"]},
    {"query": "只做算法实验应该选哪个数据库？", "keywords": ["FAISS"]},
    {"query": "Qdrant 的特色功能是什么？", "keywords": ["过滤检索"]},
    {"query": "为什么纯向量检索不够用？", "keywords": ["专有名词", "编号"]},
    {"query": "如果数据量只有十万级以内，选什么数据库？", "keywords": ["pgvector", "Chroma"]},
    {"query": "怎样解决纯向量检索查编号失效的问题？", "keywords": ["BM25", "rerank"]},
    {"query": "起步阶段建议把精力放在哪两块？", "keywords": ["切块策略", "Prompt"]},
    {"query": "如果团队没有专职运维，优先选哪些数据库？", "keywords": ["pgvector", "Chroma", "Qdrant"]},
    {"query": "FAISS 提供了持久化和元数据过滤能力吗？", "keywords": ["不提供", "numpy"]},
    {"query": "Milvus 的架构重吗？依赖哪些外部组件？", "keywords": ["etcd", "MinIO", "Pulsar"]},
    {"query": "pgvector 的最大优势是什么？", "keywords": ["不引入新系统", "Postgres"]},
]


def build_eval_set(chunks):
    eval_set = []
    for item in EVAL_CASES:
        true_indices = set()
        for i, chunk in enumerate(chunks):
            if any(kw in chunk["text"] for kw in item["keywords"]):
                true_indices.add(i)
        if true_indices:
            eval_set.append({"query": item["query"], "true_indices": true_indices})
    return eval_set


def calc_recall_at_k(scores, true_indices, k):
    top_k = [idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:k]]
    # 只要命中1个就算1.0
    if any(idx in true_indices for idx in top_k):
        return 1.0
    return 0.0


def main():
    rag.init()
    rag.retrieve_with_rerank("初始化模型", rag._chunks, rag._vecs, rag._bm25, k=1)
    chunks = rag._chunks
    vecs = rag._vecs
    bm25 = rag._bm25

    eval_set = build_eval_set(chunks)
    print(f"成功构建 {len(eval_set)} 个有效评测项。\n")

    # 为了公平对比，我们写一个通用的评分函数
    def evaluate_pipeline(name, retrieve_func, alpha=0.5, use_rerank=False):
        k_list = (1, 3, 5)
        results = {k: [] for k in k_list}

        for item in eval_set:
            query = item["query"]
            true_indices = item["true_indices"]

            # 1. 获取向量分数
            vec_scores = rag._full_vector_scores(query, chunks)
            bm25_scores = np.array(bm25.get_scores(list(jieba.cut(query))))

            def norm(x):
                if x.max() - x.min() < 1e-10: return np.zeros_like(x)
                return (x - x.min()) / (x.max() - x.min())

            if retrieve_func == "pure_vector":
                final_scores = [(i, float(vec_scores[i])) for i in range(len(chunks))]
            elif retrieve_func == "hybrid":
                combined = alpha * norm(vec_scores) + (1 - alpha) * norm(bm25_scores)
                final_scores = [(i, float(combined[i])) for i in range(len(chunks))]

            # 2. 如果用 Rerank，取前 10 个重排
            if use_rerank:
                top_10_idx = [idx for idx, _ in sorted(final_scores, key=lambda x: x[1], reverse=True)[:10]]
                pairs = [(query, chunks[i]["text"]) for i in top_10_idx]
                rerank_scores = rag._reranker.compute_score(pairs)
                final_scores = [(top_10_idx[i], float(rerank_scores[i])) for i in range(len(top_10_idx))]

            # 3. 计算指标
            for k in k_list:
                r = calc_recall_at_k(final_scores, true_indices, k)
                results[k].append(r)

        print(f"--- 【{name}】 ---")
        for k in k_list:
            avg = sum(results[k]) / len(results[k])
            print(f"  Recall@{k} = {avg:.4f}")
        return results

    print(">>> 开始对比评测...\n")

    # 1. 纯向量
    evaluate_pipeline("纯向量检索", "pure_vector")

    # 2. 混合检索 (对比不同 alpha)
    evaluate_pipeline("混合检索 (alpha=0.5)", "hybrid", alpha=0.5)
    evaluate_pipeline("混合检索 (alpha=0.7)", "hybrid", alpha=0.7)
    evaluate_pipeline("混合检索 (alpha=0.9)", "hybrid", alpha=0.9)

    # 3. 混合 + 重排 (选你最好的 alpha 跑)
    evaluate_pipeline("混合检索(alpha=0.7) + BGE重排", "hybrid", alpha=0.7, use_rerank=True)


if __name__ == "__main__":
    main()