# rag.py —— RAG 层：加载 / 切块 / embedding / Milvus 检索 / rerank / 初始化
import os
import sys
import json
import hashlib
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

# ============================================================
# 配置
# ============================================================
client = ZhipuAI(api_key=os.environ.get("ZHIPU_API_KEY"))
if not client.api_key:
    raise SystemExit("请先设置环境变量 ZHIPU_API_KEY")

DOC_DIR = os.environ.get("DOC_DIR", r"D:\shiyueyi")
CACHE_DIR = r"D:\shiyueyirag\.rag_cache"
EMBED_MODEL = "embedding-3"
CHAT_MODEL = "glm-4-air-250414"
SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".xlsx",
                  ".html", ".htm", ".md", ".txt"}

# ---- Milvus 配置 ----
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "rag_chunks"
VECTOR_DIM = 2048  # 智谱 embedding-3 的维度
RERANKER_PATH = os.environ.get("RERANKER_PATH", r"D:\models\bge-reranker-v2-m3")

_milvus = None


def get_milvus():
    global _milvus
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
                    print(f"  已加载 {path}  ({len(text)} 字符)")
            except Exception as e:
                print(f"  跳过 {path}：{e}")
    return docs


# ============================================================
# 2. 切块
# ============================================================
def build_chunks(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""]
    )
    chunks = []
    for text, source in docs:
        for piece in splitter.split_text(text):
            if piece.strip():
                chunks.append({"text": piece.strip(), "source": source})
    return chunks


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
            if len(part) <= chunk_size:
                pieces = [part]
            else:
                pieces = splitter.split_text(part)
            for piece in pieces:
                piece = piece.strip()
                if piece:
                    chunks.append({"text": piece, "source": source})
    return chunks


# ============================================================
# 3. embedding
# ============================================================
def embed(texts):
    all_vecs = []
    batch = 64
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts[i:i + batch],
        )
        all_vecs.extend([d.embedding for d in resp.data])
    return np.array(all_vecs, dtype=np.float32)


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

    client_milvus.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )

    index_params = client_milvus.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client_milvus.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
    )


def build_index(docs, chunk_size=500, chunk_overlap=50):
    """切块 -> embedding -> 写入 Milvus。返回 (chunks, milvus_client)"""
    chunks = build_chunks_by_heading(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise SystemExit("没有切出任何块")

    m = get_milvus()

    # 如果 collection 已存在且数量一致，直接复用
    if m.has_collection(COLLECTION_NAME):
        stats = m.get_collection_stats(COLLECTION_NAME)
        existing = stats.get("row_count", 0)
        if existing == len(chunks):
            print(f"Milvus 中已有 {existing} 个块，跳过 embedding")
            m.load_collection(COLLECTION_NAME)
            return chunks, m
        print(f"Milvus 中块数({existing})与当前({len(chunks)})不一致，重建集合")
        m.drop_collection(COLLECTION_NAME)

    _ensure_collection(m, VECTOR_DIM)

    print(f"共 {len(chunks)} 个块，开始计算 embedding...")
    vecs = embed([c["text"] for c in chunks])
    vecs = np.asarray(vecs, dtype=np.float32)

    if vecs.ndim != 2 or len(vecs) != len(chunks):
        raise ValueError("Embedding 数量或维度异常")
    if not np.isfinite(vecs).all():
        raise ValueError("Embedding 包含 NaN 或无穷大")
    if vecs.shape[1] != VECTOR_DIM:
        raise ValueError(f"向量维度 {vecs.shape[1]} 与配置 {VECTOR_DIM} 不一致")

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    if np.any(norms <= 1e-10):
        raise ValueError("Embedding 中存在零向量")
    vecs /= norms

    data = [
        {
            "vector": vecs[i].tolist(),
            "text": chunks[i]["text"],
            "source": chunks[i]["source"],
        }
        for i in range(len(chunks))
    ]

    BATCH = 1000
    for i in range(0, len(data), BATCH):
        m.insert(collection_name=COLLECTION_NAME, data=data[i:i+BATCH])
        print(f"  已写入 {min(i+BATCH, len(data))}/{len(data)}")

    m.flush(COLLECTION_NAME)
    m.load_collection(COLLECTION_NAME)
    print(f"索引已写入 Milvus 并加载（{COLLECTION_NAME}）")
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


def _full_vector_scores(query, chunks):
    """返回长度为 len(chunks) 的向量分数组（对齐 chunks 顺序）"""
    q = embed([query])[0]
    q /= np.linalg.norm(q) + 1e-10
    hits = _milvus_search(q, len(chunks))
    key_to_idx = {f"{c['source']}||{c['text']}": i for i, c in enumerate(chunks)}
    scores = np.zeros(len(chunks))
    for h in hits:
        key = f"{h['entity']['source']}||{h['entity']['text']}"
        idx = key_to_idx.get(key)
        if idx is not None:
            scores[idx] = float(h["distance"])
    return scores


def retrieve(query, chunks, vecs, k=5):
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


def hybrid_retrieve(query, chunks, vecs, bm25, k=5, alpha=0.5):
    vec_scores = _full_vector_scores(query, chunks)
    bm25_scores = np.array(bm25.get_scores(list(jieba.cut(query))))

    def norm(x):
        if x.max() - x.min() < 1e-10:
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    final = alpha * norm(vec_scores) + (1 - alpha) * norm(bm25_scores)
    idx = np.argsort(final)[::-1][:k]
    return [{
        "text": chunks[i]["text"],
        "source": chunks[i]["source"],
        "score": float(final[i]),
    } for i in idx]


_reranker = None


def retrieve_with_rerank(query, chunks, vecs, bm25, k=5, recall_k=10, alpha=0.5):
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(RERANKER_PATH, use_fp16=False)

    vec_scores = _full_vector_scores(query, chunks)
    bm25_scores = np.array(bm25.get_scores(list(jieba.cut(query))))

    def norm(x):
        if x.max() - x.min() < 1e-10:
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    final = alpha * norm(vec_scores) + (1 - alpha) * norm(bm25_scores)
    top_idx = np.argsort(final)[::-1][:recall_k]

    pairs = [(query, chunks[i]["text"]) for i in top_idx]
    rerank_scores = _reranker.compute_score(pairs)

    order = np.argsort(rerank_scores)[::-1][:k]

    return [{
        "text": chunks[top_idx[i]]["text"],
        "source": chunks[top_idx[i]]["source"],
        "score": float(rerank_scores[i]),
    } for i in order]


# ============================================================
# 全局状态 + 初始化
# ============================================================
_chunks = None
_vecs = None
_bm25 = None


def init(doc_dir=None):
    """显式构建索引。Agent 启动时调一次。"""
    global _chunks, _vecs, _bm25
    if _chunks is not None:
        return

    doc_dir = doc_dir or DOC_DIR
    print("加载文档：")
    docs = load_docs(doc_dir)
    if not docs:
        raise SystemExit(f"在 {doc_dir} 里没找到可加载的文档")
    print(f"共加载 {len(docs)} 个文档")

    _chunks, _ = build_index(docs)
    _vecs = None
    _bm25 = build_bm25(_chunks)
    print(f"RAG 初始化完成，共 {len(_chunks)} 个块")


def retrieve_for_agent(query: str, top_k: int = 5) -> list[str]:
    """Agent 唯一入口。返回纯文本列表。"""
    if _chunks is None:
        raise RuntimeError("请先调用 rag.init()")
    hits = retrieve_with_rerank(query, _chunks, _vecs, _bm25, k=top_k)
    return [f"[{h['source']}]\n{h['text']}" for h in hits]


# ============================================================
# 5. 评测（可选）
# ============================================================
def build_eval_set_by_keyword(queries, chunks, keywords_map):
    eval_set = []
    for q in queries:
        kws = keywords_map.get(q, [])
        true_indices = set()
        for i, c in enumerate(chunks):
            if any(kw in c["text"] for kw in kws):
                true_indices.add(i)
        eval_set.append({"query": q, "true_indices": true_indices})
    return eval_set


def evaluate_recall_bm25(chunks, vecs, bm25, eval_set, k_list=(1, 3, 5), verbose=True):
    results = {k: [] for k in k_list}
    for item in eval_set:
        query = item["query"]
        true_indices = set(item["true_indices"])
        if not true_indices:
            continue

        vec_scores = _full_vector_scores(query, chunks)
        bm25_scores = np.array(bm25.get_scores(list(jieba.cut(query))))

        def norm(x):
            if x.max() - x.min() < 1e-10:
                return np.zeros_like(x)
            return (x - x.min()) / (x.max() - x.min())

        final = 0.5 * norm(vec_scores) + 0.5 * norm(bm25_scores)
        scores = [(i, float(final[i])) for i in range(len(chunks))]

        for k in k_list:
            r = recall_at_k(scores, true_indices, k)
            results[k].append(r)

    print("\n===== BM25 混合 =====")
    summary = {}
    for k in k_list:
        if results[k]:
            avg = sum(results[k]) / len(results[k])
            summary[k] = avg
            print(f"Recall@{k} = {avg:.4f}")
    return summary


def evaluate_recall(chunks, vecs, bm25, eval_set, k_list=(1, 3, 5), verbose=True):
    results = {k: [] for k in k_list}
    for item in eval_set:
        query = item["query"]
        true_indices = set(item["true_indices"])
        if not true_indices:
            continue

        vec_scores = _full_vector_scores(query, chunks)
        bm25_scores = np.array(bm25.get_scores(list(jieba.cut(query))))

        def norm(x):
            if x.max() - x.min() < 1e-10:
                return np.zeros_like(x)
            return (x - x.min()) / (x.max() - x.min())

        final = 0.5 * norm(vec_scores) + 0.5 * norm(bm25_scores)
        top_idx = np.argsort(final)[::-1][:10]

        global _reranker
        if _reranker is None:
            _reranker = FlagReranker(RERANKER_PATH, use_fp16=False)

        pairs = [(query, chunks[i]["text"]) for i in top_idx]
        rerank_scores = _reranker.compute_score(pairs)

        reranked_order = sorted(
            range(len(top_idx)),
            key=lambda i: rerank_scores[i],
            reverse=True
        )
        scores = [(top_idx[i], float(rerank_scores[i])) for i in reranked_order]

        for k in k_list:
            r = recall_at_k(scores, true_indices, k)
            results[k].append(r)
            if verbose:
                print(f"  [{query}] Recall@{k} = {r:.3f}")

    print("\n===== 召回率评测结果（Rerank） =====")
    summary = {}
    for k in k_list:
        if results[k]:
            avg = sum(results[k]) / len(results[k])
            summary[k] = avg
            print(f"Recall@{k} = {avg:.4f}  (共 {len(results[k])} 个 query)")
    return summary


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