# eval.py —— RAG 检索质量评测
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import rag

rag.init()

eval_set = [
    {"query": "什么是向量数据库？", "true_indices": {1}},
    {"query": "FAISS 和 Milvus 有什么区别？", "true_indices": {3, 5}},
    {"query": "向量数据库选型看哪些维度？", "true_indices": {8}},
    {"query": "向量数据库有哪些常见误区？", "true_indices": {9}},
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