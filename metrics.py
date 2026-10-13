# metrics.py

def recall_at_k(scores, true_indices, k):
    """
    scores:      列表，每个元素是 (句子/块编号, 分数)
    true_indices: 真正相关的编号集合
    k:           取前 K 个
    """
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    top_k = [idx for idx, _ in ranked[:k]]
    true_set = set(true_indices)
    hit = set(top_k) & true_set
    return len(hit) / len(true_set) if true_set else 0.0