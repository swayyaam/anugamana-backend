"""
Hybrid retrieval: dense (ChromaDB) + sparse (BGE-M3 lexical weights) fused with RRF.

Architecture:
  - HyDE vector + expansion query vectors all searched
  - Dense: gita_verses collection + gita_purport collection (top_k=15 each)
  - Sparse: sparse_index (top_k=15)
  - RRF fusion: score = 1/(60 + rank)
  - Group by verse_id: verse score = max score across all its vectors
  - Return top 10 verses for reranking
"""

import asyncio
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import chromadb
from FlagEmbedding import BGEM3FlagModel

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
SPARSE_FILE = DATA_DIR / "sparse_index.pkl"

TOP_K = 15          # candidates per search
RRF_K = 60          # RRF constant
TOP_VERSES = 10     # verses passed to reranker


@lru_cache(maxsize=1)
def _load_model() -> BGEM3FlagModel:
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


@lru_cache(maxsize=1)
def _load_chroma():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    verses_col = client.get_collection("gita_verses")
    purport_col = client.get_collection("gita_purport")
    return verses_col, purport_col


@lru_cache(maxsize=1)
def _load_sparse() -> dict:
    with open(SPARSE_FILE, "rb") as f:
        return pickle.load(f)


def _embed(texts: list[str]) -> tuple[np.ndarray, list[dict]]:
    model = _load_model()
    out = model.encode(
        texts,
        batch_size=len(texts),
        max_length=512,
        return_dense=True,
        return_sparse=True,
    )
    return out["dense_vecs"], out["lexical_weights"]


def _sparse_search(query_weights: dict, top_k: int) -> list[tuple[str, float]]:
    """Score all docs against query lexical weights, return top_k (doc_id, score)."""
    sparse_index = _load_sparse()
    scores: dict[str, float] = {}
    for token_id, q_weight in query_weights.items():
        token_key = str(token_id)
        if token_key not in sparse_index:
            continue
        for doc_id, d_weight in sparse_index[token_key].items():
            scores[doc_id] = scores.get(doc_id, 0.0) + float(q_weight) * float(d_weight)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _dense_search(collection, vector: np.ndarray, top_k: int) -> list[dict]:
    results = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=top_k,
        include=["metadatas", "distances"],
    )
    hits = []
    for doc_id, meta, dist in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"id": doc_id, "metadata": meta, "distance": dist})
    return hits


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def _fuse_and_group(all_hit_lists: list[list[tuple[str, float | dict]]]) -> dict[str, float]:
    """
    RRF fusion across multiple ranked lists.
    Each list is either dense hits (list of dicts with 'id') or sparse hits (list of (id, score)).
    Returns {verse_id: rrf_score} grouped by verse.
    """
    doc_scores: dict[str, float] = {}

    for hit_list in all_hit_lists:
        for rank, hit in enumerate(hit_list):
            if isinstance(hit, dict):
                doc_id = hit["id"]
            else:
                doc_id = hit[0]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + _rrf_score(rank)

    # Group by verse_id (take max score across all vectors for same verse)
    verse_scores: dict[str, float] = {}
    for doc_id, score in doc_scores.items():
        # doc_id format: "2.47_meaning", "2.47_purport_0", etc.
        verse_id = doc_id.rsplit("_", 1)[0]
        # Handle purport IDs: "2.47_purport_0" → verse_id = "2.47"
        if "_purport_" in doc_id:
            verse_id = doc_id.split("_purport_")[0]
        elif "_meaning" in doc_id or "_translation" in doc_id:
            verse_id = doc_id.rsplit("_", 1)[0]

        if verse_id not in verse_scores or score > verse_scores[verse_id]:
            verse_scores[verse_id] = score

    return verse_scores


def retrieve(hyde_text: str, all_queries: list[str]) -> list[dict]:
    """
    Full hybrid retrieval.
    Returns list of verse dicts (top TOP_VERSES), each with verse metadata + rrf_score.
    """
    verses_col, purport_col = _load_chroma()

    # Embed HyDE text + all expansion queries
    texts_to_embed = [hyde_text] + all_queries
    dense_vecs, sparse_weights = _embed(texts_to_embed)

    hyde_vec = dense_vecs[0]
    query_vecs = dense_vecs[1:]
    hyde_sparse = sparse_weights[0]
    query_sparse_list = sparse_weights[1:]

    all_hit_lists = []

    # Dense: HyDE vector → verses + purport
    all_hit_lists.append(_dense_search(verses_col, hyde_vec, TOP_K))
    all_hit_lists.append(_dense_search(purport_col, hyde_vec, TOP_K))

    # Dense: each expansion query → verses + purport
    for q_vec in query_vecs:
        all_hit_lists.append(_dense_search(verses_col, q_vec, TOP_K))
        all_hit_lists.append(_dense_search(purport_col, q_vec, TOP_K))

    # Sparse: HyDE + each expansion
    all_hit_lists.append(_sparse_search(hyde_sparse, TOP_K))
    for q_sparse in query_sparse_list:
        all_hit_lists.append(_sparse_search(q_sparse, TOP_K))

    # RRF fusion + group by verse
    verse_scores = _fuse_and_group(all_hit_lists)

    # Sort by score, take top TOP_VERSES
    top_verse_ids = sorted(verse_scores, key=lambda v: verse_scores[v], reverse=True)[:TOP_VERSES]

    # Fetch full metadata for top verses from ChromaDB
    if not top_verse_ids:
        return []

    meaning_ids = [f"{vid}_meaning" for vid in top_verse_ids]
    results = verses_col.get(ids=meaning_ids, include=["metadatas"])

    id_to_meta = {
        doc_id: meta
        for doc_id, meta in zip(results["ids"], results["metadatas"])
    }

    verses = []
    for vid in top_verse_ids:
        meta = id_to_meta.get(f"{vid}_meaning")
        if meta:
            verses.append({**meta, "rrf_score": verse_scores[vid]})

    return verses
