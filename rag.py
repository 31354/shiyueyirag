# rag.py —— RAG 层：加载 / 切块 / embedding / Milvus 检索 / rerank / 初始化
import os
import sys
import json
import time
import pickle
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

import numpy as np
import jieba
from zhipuai import ZhipuAI
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker
from pymilvus import MilvusClient, DataType

from jiexi import parse_document
from metrics import recall_at_k

sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag")

# ============================================================
# 配置
# ============================================================
client = ZhipuAI(api_key=os.environ.get("ZHIPU_API_KEY"))
if not client.api_key:
    raise SystemExit("请先设置环境变量 ZHIPU_API_KEY")

DOC_DIR = os.environ.get("DOC_DIR", r"D:\shiyueyi")
CACHE_DIR = os.environ.get("RAG_CACHE_DIR", r"D:\shiyueyirag\.rag_cache")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embedding-3")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "glm-4-air-250414")
SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".xlsx",
                  ".html", ".htm", ".md", ".txt"}

# ---- Milvus 配置 ----
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "rag_chunks")
# [改] 维度不再写死，首次 embed 时探测并缓存；这里仅作为 fallback
FALLBACK_VECTOR_DIM = int(os.environ.get("VECTOR_DIM", "2048"))
RERANKER_PATH = os.environ.get("RERANKER_PATH", r"D:\models\bge-reranker-v2-m3")
RERANKER_DEVICE = os.environ.get("RERANKER_DEVICE", None)  # 如 "cuda" / "cpu"

# [改] embedding 重试 / 并发参数
EMBED_BATCH = 64
EMBED_MAX_RETRY = 5
EMBED_WORKERS = 4

os.makedirs(CACHE_DIR, exist_ok=True)

_milvus = None
_milvus_lock = threading.Lock()
_reranker = None
_reranker_lock = threading.Lock()


def get_milvus():
    global _milvus
    if _milvus is None:
        with _milvus_lock:
            if _milvus is None:
                _milvus = MilvusClient(MILVUS_URI)
    return _milvus


# ============================================================
# 1. 文档加载
# ============================================================
def load_one(path):
    return parse_document(str(path))


def load_docs(root_dir):
    docs = []
    root = Path(root_dir)
    if not root.exists():
        raise SystemExit(f"目录不存在：{root_dir}")

    for path in root.rglob("*"):
        if any(part.startswith(".") for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            try:
                text = load_one(path)
                if text.strip():
                    docs.append((text, str(path)))
                    log.info("已加载 %s (%d 字符)", path, len(text))
            except Exception:
                # [改] 记录堆栈而不是静默吞掉
                log.exception("跳过 %s", path)
    return docs


# ============================================================
# 2. 切块
# ============================================================
def build_chunks_by_heading(docs, chunk_size=500, chunk_overlap=50):
    import re

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        keep_separator=True,
    )

    chunks = []
    for text, source in docs:
        if not text or not text.strip():
            continue
        parts = re.split(r"\n(?=#{2,3} )", text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            pieces = [part] if len(part) <= chunk_size else splitter.split_text(part)
            for piece in pieces:
                piece = piece.strip()
                if piece:
                    chunks.append({"text": piece, "source": source})
    return chunks


# ============================================================
# 3. embedding
# ============================================================
_vector_dim_cache = None


def _embed_batch_with_retry(batch_texts):
    last_err = None
    for attempt in range(1, EMBED_MAX_RETRY + 1):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
            # 按 index 排序，防止返回乱序
            data = sorted(resp.data, key=lambda d: d.index)
            if len(data) != len(batch_texts):
                log.warning("embedding 返回条数不符：请求 %d，返回 %d", len(batch_texts), len(data))
            return [d.embedding for d in data]
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            log.warning("embedding 失败(第 %d 次)，%.1fs 后重试：%s", attempt, wait, e)
            time.sleep(wait)
    raise RuntimeError(f"embedding 连续失败 {EMBED_MAX_RETRY} 次") from last_err


def embed(texts):
    """[改] 支持并发 + 重试，结果顺序与输入一致。"""
    if not texts:
        return np.zeros((0, FALLBACK_VECTOR_DIM), dtype=np.float32)

    batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
    results = [None] * len(batches)

    def _task(batch_idx):
        return batch_idx, _embed_batch_with_retry(batches[batch_idx])

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as ex:
        futures = [ex.submit(_task, i) for i in range(len(batches))]
        for fut in as_completed(futures):
            idx, vecs = fut.result()
            results[idx] = vecs

    all_vecs = []
    for r in results:
        all_vecs.extend(r)
    return np.array(all_vecs, dtype=np.float32)


def probe_vector_dim():
    """[改] 启动时探测一次 embedding 维度，避免写死。"""
    global _vector_dim_cache
    if _vector_dim_cache is None:
        v = embed(["dim_probe"])
        _vector_dim_cache = int(v.shape[1])
        log.info("embedding 维度探测结果：%d", _vector_dim_cache)
    return _vector_dim_cache


def cache_key(docs, chunk_size=500, chunk_overlap=50):
    h = hashlib.md5()
    config = {
        "chunk_strategy": "heading_recursive_v2",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": EMBED_MODEL,
    }
    h.update(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for text, source in docs:
        h.update(source.encode("utf-8"))
        h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_path(name: str, key: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}_{key}.pkl")


def _load_pickle(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        log.exception("读取缓存失败：%s", path)
        return None


def _save_pickle(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)  # 原子替换


# ============================================================
# Milvus Collection 管理
# ============================================================
def _ensure_collection(client_milvus, dim):
    """创建 Collection + 索引，已存在则跳过"""
    if client_milvus.has_collection(COLLECTION_NAME):
        return

    schema = client_milvus.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("source", DataType.VARCHAR, max_length=1024)

    client_milvus.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    index_params = client_milvus.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client_milvus.create_index(collection_name=COLLECTION_NAME, index_params=index_params)


def _collection_matches(m, key, n_chunks):
    """[改] 用 cache_key 校验索引版本，而不是只看块数。"""
    if not m.has_collection(COLLECTION_NAME):
        return False
    try:
        stats = m.get_collection_stats(COLLECTION_NAME)
        existing = stats.get("row_count", 0)
    except Exception:
        return False
    if existing != n_chunks:
        return False
    marker = _cache_path("index_version", key)
    if not os.path.exists(marker):
        return False
    return True


def _write_version_marker(key, n_chunks):
    _save_pickle(_cache_path("index_version", key), {"chunks": n_chunks})


def build_index(docs, chunk_size=500, chunk_overlap=50):
    """切块 -> embedding -> 写入 Milvus。返回 (chunks, milvus_client)"""
    chunks = build_chunks_by_heading(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise SystemExit("没有切出任何块")

    m = get_milvus()
    key = cache_key(docs, chunk_size, chunk_overlap)

    # [改] 版本一致才复用；否则重建
    if _collection_matches(m, key, len(chunks)):
        log.info("Milvus 中索引版本一致（%d 块），跳过 embedding", len(chunks))
        m.load_collection(COLLECTION_NAME)
        return chunks, m

    if m.has_collection(COLLECTION_NAME):
        log.info("索引版本不一致，重建集合")
        m.drop_collection(COLLECTION_NAME)

    dim = probe_vector_dim()
    _ensure_collection(m, dim)

    log.info("共 %d 个块，开始计算 embedding...", len(chunks))
    vecs = embed([c["text"] for c in chunks])
    vecs = np.asarray(vecs, dtype=np.float32)

    if vecs.ndim != 2 or len(vecs) != len(chunks):
        raise ValueError("Embedding 数量或维度异常")
    if not np.isfinite(vecs).all():
        raise ValueError("Embedding 包含 NaN 或无穷大")
    if vecs.shape[1] != dim:
        raise ValueError(f"向量维度 {vecs.shape[1]} 与探测维度 {dim} 不一致")

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    if np.any(norms <= 1e-10):
        raise ValueError("Embedding 中存在零向量")
    vecs /= norms

    data = [
        {"vector": vecs[i].tolist(), "text": chunks[i]["text"], "source": chunks[i]["source"]}
        for i in range(len(chunks))
    ]

    BATCH = 1000
    for i in range(0, len(data), BATCH):
        m.insert(collection_name=COLLECTION_NAME, data=data[i:i + BATCH])
        log.info("已写入 %d/%d", min(i + BATCH, len(data)), len(data))

    m.flush(COLLECTION_NAME)
    m.load_collection(COLLECTION_NAME)
    _write_version_marker(key, len(chunks))
    log.info("索引已写入 Milvus 并加载（%s）", COLLECTION_NAME)
    return chunks, m


# ============================================================
# 4. 检索
# ============================================================
def _milvus_search(query_vec, k):
    """统一走 Milvus 向量检索"""
    m = get_milvus()
    results = m.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec.tolist()],
        limit=k,
        output_fields=["text", "source"],
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )
    return results[0]


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    if x.max() - x.min() < 1e-10:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


def _rrf_fuse(rankings, k=60, weights=None):
    """[改] RRF 融合多路排序，对分数尺度不敏感。"""
    if weights is None:
        weights = [1.0] * len(rankings)
    score = {}
    for ranking, w in zip(rankings, weights):
        for rank, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + w / (k + rank + 1)
    return score


def _hybrid_topn(query, chunks, bm25, alpha=0.5, top_n=100, use_rrf=False):
    """[改] 混合检索：向量取 Top-N + BM25 Top-N，不再全量拉 Milvus。
    返回 [(chunk_idx, score), ...] 按分数降序。"""
    q = embed([query])[0]
    q /= np.linalg.norm(q) + 1e-10
    n_vec = min(top_n, len(chunks))
    vec_hits = _milvus_search(q, n_vec)

    key_to_idx = {f"{c['source']}||{c['text']}": i for i, c in enumerate(chunks)}
    vec_rank = []
    vec_score_map = {}
    for h in vec_hits:
        key = f"{h['entity']['source']}||{h['entity']['text']}"
        idx = key_to_idx.get(key)
        if idx is not None:
            vec_rank.append(idx)
            vec_score_map[idx] = float(h["distance"])

    bm25_raw = np.array(bm25.get_scores(list(jieba.cut(query))))
    bm25_top = list(np.argsort(bm25_raw)[::-1][:top_n])

    if use_rrf:
        fused = _rrf_fuse([vec_rank, bm25_top])
        merged = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return merged

    # 加权融合（只在两路候选并集上计算）
    candidate = set(vec_rank) | set(bm25_top)
    vec_arr = np.array([vec_score_map.get(i, 0.0) for i in range(len(chunks))])
    fused_scores = alpha * _norm(vec_arr) + (1 - alpha) * _norm(bm25_raw)
    merged = [(i, float(fused_scores[i])) for i in candidate]
    merged.sort(key=lambda kv: kv[1], reverse=True)
    return merged


def retrieve(query, chunks, k=5):
    q = embed([query])[0]
    q /= np.linalg.norm(q) + 1e-10
    hits = _milvus_search(q, k)
    return [{
        "text": h["entity"]["text"],
        "source": h["entity"]["source"],
        "score": float(h["distance"]),
    } for h in hits]


def build_bm25(chunks):
    tokenized = [list(jieba.cut(c["text"])) for c in chunks]
    return BM25Okapi(tokenized)


def hybrid_retrieve(query, chunks, bm25, k=5, alpha=0.5, top_n=100, use_rrf=False):
    merged = _hybrid_topn(query, chunks, bm25, alpha=alpha, top_n=top_n, use_rrf=use_rrf)
    return [{
        "text": chunks[i]["text"],
        "source": chunks[i]["source"],
        "score": float(s),
    } for i, s in merged[:k]]


def get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                kwargs = {"use_fp16": True}
                if RERANKER_DEVICE:
                    kwargs["device"] = RERANKER_DEVICE
                try:
                    _reranker = FlagReranker(RERANKER_PATH, **kwargs)
                except Exception:
                    log.warning("use_fp16 加载失败，回退 fp32")
                    _reranker = FlagReranker(RERANKER_PATH, use_fp16=False)
    return _reranker


def retrieve_with_rerank(query, chunks, bm25, k=5, recall_k=50, alpha=0.5,
                         top_n=100, use_rrf=False):
    """[改] recall_k 默认提到 50，rerank 前先混合召回更多候选。"""
    merged = _hybrid_topn(query, chunks, bm25, alpha=alpha, top_n=top_n, use_rrf=use_rrf)
    top = merged[:recall_k]
    if not top:
        return []

    rr = get_reranker()
    pairs = [(query, chunks[i]["text"]) for i, _ in top]
    rerank_scores = rr.compute_score(pairs)

    order = np.argsort(rerank_scores)[::-1][:k]
    return [{
        "text": chunks[top[i][0]]["text"],
        "source": chunks[top[i][0]]["source"],
        "score": float(rerank_scores[i]),
    } for i in order]


# ============================================================
# 全局状态 + 初始化
# ============================================================
_chunks = None
_bm25 = None
_init_lock = threading.Lock()


def init(doc_dir=None, chunk_size=500, chunk_overlap=50, force=False):
    """显式构建索引。Agent 启动时调一次。"""
    global _chunks, _bm25
    with _init_lock:
        if _chunks is not None and not force:
            return

        doc_dir = doc_dir or DOC_DIR
        log.info("加载文档：%s", doc_dir)
        docs = load_docs(doc_dir)
        if not docs:
            raise SystemExit(f"在 {doc_dir} 里没找到可加载的文档")
        log.info("共加载 %d 个文档", len(docs))

        chunks, _ = build_index(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        key = cache_key(docs, chunk_size, chunk_overlap)

        # [改] BM25 分词结果落盘缓存，避免每次启动重算
        bm25_cache = _cache_path("bm25_tokens", key)
        tokens = _load_pickle(bm25_cache)
        if tokens is None or len(tokens) != len(chunks):
            log.info("构建 BM25 分词缓存...")
            tokens = [list(jieba.cut(c["text"])) for c in chunks]
            _save_pickle(bm25_cache, tokens)
        bm25 = BM25Okapi(tokens)

        _chunks = chunks
        _bm25 = bm25
        log.info("RAG 初始化完成，共 %d 个块", len(_chunks))


def retrieve_for_agent(query: str, top_k: int = 5) -> list[dict]:
    """[改] Agent 入口，返回结构化结果（text/source/score），不再把路径拼进正文。"""
    if _chunks is None:
        raise RuntimeError("请先调用 rag.init()")
    return retrieve_with_rerank(query, _chunks, _bm25, k=top_k)


def format_hits(hits: list[dict]) -> str:
    """需要拼 prompt 时用这个。"""
    return "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)


# ============================================================
# 5. 评测（可选）
# ============================================================
def build_eval_set_by_keyword(queries, chunks, keywords_map):
    eval_set = []
    for q in queries:
        kws = keywords_map.get(q, [])
        true_indices = {i for i, c in enumerate(chunks) if any(kw in c["text"] for kw in kws)}
        eval_set.append({"query": q, "true_indices": true_indices})
    return eval_set


def _eval_common(chunks, bm25, eval_set, k_list, use_rerank, alpha=0.5, top_n=100,
                 recall_k=50, verbose=True):
    results = {k: [] for k in k_list}
    rr = get_reranker() if use_rerank else None

    for item in eval_set:
        query = item["query"]
        true_indices = set(item["true_indices"])
        if not true_indices:
            continue

        merged = _hybrid_topn(query, chunks, bm25, alpha=alpha, top_n=top_n)

        if use_rerank:
            top = merged[:recall_k]
            pairs = [(query, chunks[i]["text"]) for i, _ in top]
            rs = rr.compute_score(pairs)
            order = np.argsort(rs)[::-1]
            scores = [(top[i][0], float(rs[i])) for i in order]
        else:
            scores = [(i, s) for i, s in merged]

        for k in k_list:
            r = recall_at_k(scores, true_indices, k)
            results[k].append(r)
            if verbose:
                log.info("[%s] Recall@%d = %.3f", query, k, r)

    summary = {k: (sum(v) / len(v) if v else 0.0) for k, v in results.items()}
    tag = "Rerank" if use_rerank else "BM25 混合"
    print(f"\n===== 召回率评测结果（{tag}） =====")
    for k, v in summary.items():
        print(f"Recall@{k} = {v:.4f}  (共 {len(results[k])} 个 query)")
    return summary


def evaluate_recall_bm25(chunks, bm25, eval_set, k_list=(1, 3, 5), verbose=True):
    return _eval_common(chunks, bm25, eval_set, k_list, use_rerank=False, verbose=verbose)


def evaluate_recall(chunks, bm25, eval_set, k_list=(1, 3, 5), verbose=True):
    return _eval_common(chunks, bm25, eval_set, k_list, use_rerank=True, verbose=verbose)


if __name__ == "__main__":
    print("开始检查 Milvus 索引状态……")
    docs = load_docs(DOC_DIR)
    if not docs:
        raise SystemExit(f"在 {DOC_DIR} 中没有找到可加载的文档")

    chunks = build_chunks_by_heading(docs, chunk_size=500, chunk_overlap=50)
    print(f"当前文档数量：{len(docs)}")
    print(f"当前文本块数量：{len(chunks)}")

    m = get_milvus()
    print(f"Milvus collection 存在：{m.has_collection(COLLECTION_NAME)}")
    if m.has_collection(COLLECTION_NAME):
        stats = m.get_collection_stats(COLLECTION_NAME)
        print(f"Milvus 中块数：{stats.get('row_count', 0)}")