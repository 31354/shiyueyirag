# eval.py —— RAG 检索质量评测
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import rag

rag.init()

eval_set = [
    # === 定义类 ===
    {"query": "什么是向量数据库？", "true_indices": {1}},
    {"query": "向量数据库和传统数据库有什么区别？", "true_indices": {1}},
    {"query": "向量数据库在 RAG 里扮演什么角色？", "true_indices": {1}},

    # === 对比类 ===
    {"query": "FAISS 和 Milvus 有什么区别？", "true_indices": {3, 5}},
    {"query": "Chroma 和 Milvus 有什么区别？", "true_indices": {4, 5}},
    {"query": "Qdrant 和 pgvector 有什么区别？", "true_indices": {6, 7}},
    {"query": "FAISS 是什么？它有什么优缺点？", "true_indices": {3}},

    # === 选型类 ===
    {"query": "向量数据库选型看哪些维度？", "true_indices": {8}},
    {"query": "没有运维团队应该选什么？", "true_indices": {8}},
    {"query": "十亿级数据应该用什么向量库？", "true_indices": {8}},
    {"query": "十万级以内推荐用什么？", "true_indices": {8}},

    # === 误区类 ===
    {"query": "向量数据库有哪些常见误区？", "true_indices": {9}},
    {"query": "向量库越大越好吗？", "true_indices": {9}},
    {"query": "换了向量库检索效果会变好吗？", "true_indices": {9}},

    # === 实践类 ===
    {"query": "起步阶段推荐用什么？", "true_indices": {10}},
    {"query": "什么时候该迁移向量库？", "true_indices": {10}},
    {"query": "切块策略和向量库选型哪个更重要？", "true_indices": {10}},

    # === 单方案细节 ===
    {"query": "Chroma 适合什么场景？", "true_indices": {4}},
    {"query": "Qdrant 的优势是什么？", "true_indices": {6}},
    {"query": "pgvector 适合多大数据量？", "true_indices": {7}},
]

print("\n标准答案：")
for item in eval_set:
    print(f"  [{item['query']}] -> chunk {sorted(item['true_indices'])}")

# 通过 rag.xxx 访问，拿到 init 后的最新值
r_bm25 = rag.evaluate_recall_bm25(rag._chunks, rag._vecs, rag._bm25, eval_set, k_list=[1, 3, 5])
r_rerank = rag.evaluate_recall(rag._chunks, rag._vecs, rag._bm25, eval_set, k_list=[1, 3, 5])

print("\n===== 对比 =====")
for k in [1, 3, 5]:
    diff = r_rerank[k] - r_bm25[k]
    sign = "+" if diff >= 0 else ""
    print(f"Recall@{k}: BM25={r_bm25[k]:.4f}  +Rerank={r_rerank[k]:.4f}  {sign}{diff:.4f}")