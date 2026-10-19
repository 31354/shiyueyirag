# metrics.py

def recall_at_k(scores, true_indices, k):
    """
    计算标准召回率 (Recall@K)。
    scores:      列表，每个元素是 (句子/块编号, 分数)
    true_indices: 真正相关的编号集合
    k:           取前 K 个
    """
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    top_k = [idx for idx, _ in ranked[:k]]
    true_set = set(true_indices)
    hit = set(top_k) & true_set
    return len(hit) / len(true_set) if true_set else 0.0


def hit_rate_at_k(scores, true_indices, k):
    """
    计算命中率 (Hit Rate@K)。
    只要前 K 个结果中包含至少 1 个正确结果，就记为 1 分，否则为 0 分。
    """
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    top_k = [idx for idx, _ in ranked[:k]]
    true_set = set(true_indices)
    # 判断交集是否非空
    return 1.0 if set(top_k) & true_set else 0.0


def mrr_at_k(scores, true_indices, k):
    """
    计算平均倒数排名 (MRR@K)。
    看第一个命中的正确答案排在第几位，排第1得1分，第2得1/2，第3得1/3。
    （这是目前企业级 RAG 最常用的核心指标之一）
    """
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    true_set = set(true_indices)

    for rank, (idx, _) in enumerate(ranked[:k], start=1):
        if idx in true_set:
            return 1.0 / rank
    return 0.0