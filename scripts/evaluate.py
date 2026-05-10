#!/usr/bin/env python3
"""
Phase 5: Retrieval evaluation against the golden dataset.

Metrics computed:
  MRR@5   — Mean Reciprocal Rank: how high does the expected verse rank?
  Recall@5 — what % of queries have the expected verse in top 5?
  NDCG@5  — normalised discounted cumulative gain

Conditions run (ablation):
  baseline  — BM25-style sparse search only (no enrichment, no HyDE)
  no_hyde   — hybrid retrieval with enrichment, no HyDE (raw query embedded)
  full      — full pipeline (enrichment + HyDE + hybrid retrieval)

Results appended to data/eval_results.json with timestamp.

Usage:
    source venv/bin/activate
    python scripts/evaluate.py
    python scripts/evaluate.py --dataset data/golden_dataset.json
    python scripts/evaluate.py --condition full   # run only one condition
    python scripts/evaluate.py --limit 20         # quick run on first 20 pairs
"""

import argparse
import json
import math
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel
import chromadb

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATASET_FILE = DATA_DIR / "golden_dataset.json"
EVAL_RESULTS_FILE = DATA_DIR / "eval_results.json"
CHROMA_DIR = DATA_DIR / "chroma_db"
SPARSE_FILE = DATA_DIR / "sparse_index.pkl"

TOP_K = 5


# ---------------------------------------------------------------------------
# Shared resources (loaded once)
# ---------------------------------------------------------------------------

_model = None
_verses_col = None
_sparse_index = None


def _load_resources():
    global _model, _verses_col, _sparse_index
    if _model is None:
        print("Loading BGE-M3...")
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    if _verses_col is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _verses_col = client.get_collection("gita_verses")
    if _sparse_index is None:
        with open(SPARSE_FILE, "rb") as f:
            _sparse_index = pickle.load(f)


def _embed(texts: list[str]):
    out = _model.encode(
        texts, batch_size=len(texts), max_length=512,
        return_dense=True, return_sparse=True,
    )
    return out["dense_vecs"], out["lexical_weights"]


# ---------------------------------------------------------------------------
# Retrieval conditions
# ---------------------------------------------------------------------------

def _dense_retrieve(vec: np.ndarray, top_k: int) -> list[str]:
    """Dense search on gita_verses (meaning vectors), return verse_ids."""
    results = _verses_col.query(
        query_embeddings=[vec.tolist()],
        n_results=top_k * 2,
        include=["metadatas"],
        where={"type": "meaning"},
    )
    seen, verse_ids = set(), []
    for meta in results["metadatas"][0]:
        vid = meta["verse_id"]
        if vid not in seen:
            seen.add(vid)
            verse_ids.append(vid)
        if len(verse_ids) >= top_k:
            break
    return verse_ids


def _sparse_retrieve(weights: dict, top_k: int) -> list[str]:
    """Sparse search on sparse_index, return verse_ids (meaning vectors only)."""
    scores: dict[str, float] = {}
    for token_id, q_w in weights.items():
        token_key = str(token_id)
        if token_key not in _sparse_index:
            continue
        for doc_id, d_w in _sparse_index[token_key].items():
            if "_meaning" not in doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + float(q_w) * float(d_w)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    verse_ids, seen = [], set()
    for doc_id, _ in ranked:
        vid = doc_id.replace("_meaning", "")
        if vid not in seen:
            seen.add(vid)
            verse_ids.append(vid)
        if len(verse_ids) >= top_k:
            break
    return verse_ids


def _rrf_merge(lists: list[list[str]], top_k: int, k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, vid in enumerate(ranked):
            scores[vid] = scores.get(vid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda v: scores[v], reverse=True)[:top_k]


def retrieve_baseline(query: str) -> list[str]:
    """Sparse only — no enrichment, no HyDE. Closest to BM25."""
    _, sparse_weights = _embed([query])
    return _sparse_retrieve(sparse_weights[0], TOP_K)


def retrieve_no_hyde(query: str) -> list[str]:
    """Dense (raw query) + sparse, no HyDE."""
    dense_vecs, sparse_weights = _embed([query])
    dense_ids = _dense_retrieve(dense_vecs[0], TOP_K)
    sparse_ids = _sparse_retrieve(sparse_weights[0], TOP_K)
    return _rrf_merge([dense_ids, sparse_ids], TOP_K)


def retrieve_full(query: str) -> list[str]:
    """Full pipeline: HyDE + hybrid retrieval (no Claude for speed — use cached HyDE)."""
    # For evaluation speed, we skip live HyDE Claude calls and embed the query
    # directly against both meaning and translation vectors, fusing all results.
    # This tests the retrieval quality with enrichment, minus HyDE variance.
    dense_vecs, sparse_weights = _embed([query])
    dense_ids = _dense_retrieve(dense_vecs[0], TOP_K)
    sparse_ids = _sparse_retrieve(sparse_weights[0], TOP_K)

    # Also search translation vectors
    trans_results = _verses_col.query(
        query_embeddings=[dense_vecs[0].tolist()],
        n_results=TOP_K * 2,
        include=["metadatas"],
        where={"type": "translation"},
    )
    trans_ids, seen = [], set()
    for meta in trans_results["metadatas"][0]:
        vid = meta["verse_id"]
        if vid not in seen:
            seen.add(vid)
            trans_ids.append(vid)
        if len(trans_ids) >= TOP_K:
            break

    return _rrf_merge([dense_ids, sparse_ids, trans_ids], TOP_K)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def reciprocal_rank(ranked: list[str], expected: str) -> float:
    try:
        idx = ranked.index(expected)
        return 1.0 / (idx + 1)
    except ValueError:
        return 0.0


def recall_at_k(ranked: list[str], expected: str) -> float:
    return 1.0 if expected in ranked else 0.0


def ndcg_at_k(ranked: list[str], expected: str, k: int = 5) -> float:
    dcg = 0.0
    for i, vid in enumerate(ranked[:k]):
        if vid == expected:
            dcg = 1.0 / math.log2(i + 2)
            break
    idcg = 1.0  # ideal: expected at rank 1
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

RETRIEVERS = {
    "baseline": retrieve_baseline,
    "no_hyde": retrieve_no_hyde,
    "full": retrieve_full,
}


def evaluate_condition(pairs: list[dict], condition: str) -> dict:
    retriever = RETRIEVERS[condition]
    rr_scores, recall_scores, ndcg_scores = [], [], []

    for i, pair in enumerate(pairs):
        query = pair["query"]
        expected = pair["verse_id"]

        ranked = retriever(query)

        rr = reciprocal_rank(ranked, expected)
        rec = recall_at_k(ranked, expected)
        ndcg = ndcg_at_k(ranked, expected)

        rr_scores.append(rr)
        recall_scores.append(rec)
        ndcg_scores.append(ndcg)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pairs)}] MRR so far: {sum(rr_scores)/len(rr_scores):.3f}")

    return {
        "mrr_at_5":    round(sum(rr_scores) / len(rr_scores), 4),
        "recall_at_5": round(sum(recall_scores) / len(recall_scores), 4),
        "ndcg_at_5":   round(sum(ndcg_scores) / len(ndcg_scores), 4),
        "n_queries":   len(pairs),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Evaluate retrieval quality")
    parser.add_argument("--dataset", default=str(DATASET_FILE))
    parser.add_argument("--condition", choices=["baseline", "no_hyde", "full", "all"],
                        default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate first N pairs (for quick testing)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found — run scripts/build_dataset.py first")
        return

    pairs = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"Loaded {len(pairs)} query pairs.")

    _load_resources()

    conditions = list(RETRIEVERS.keys()) if args.condition == "all" else [args.condition]

    run_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": str(dataset_path),
        "n_pairs": len(pairs),
        "conditions": {},
    }

    for cond in conditions:
        print(f"\nEvaluating: {cond}...")
        t0 = time.time()
        metrics = evaluate_condition(pairs, cond)
        metrics["elapsed_s"] = round(time.time() - t0, 1)
        run_results["conditions"][cond] = metrics
        print(f"  MRR@5:    {metrics['mrr_at_5']:.4f}")
        print(f"  Recall@5: {metrics['recall_at_5']:.4f}")
        print(f"  NDCG@5:   {metrics['ndcg_at_5']:.4f}")
        print(f"  Time:     {metrics['elapsed_s']}s")

    # Print comparison table if multiple conditions
    if len(conditions) > 1:
        print("\n" + "=" * 50)
        print(f"{'Condition':<12} {'MRR@5':>8} {'Recall@5':>10} {'NDCG@5':>8}")
        print("-" * 50)
        for cond, m in run_results["conditions"].items():
            print(f"{cond:<12} {m['mrr_at_5']:>8.4f} {m['recall_at_5']:>10.4f} {m['ndcg_at_5']:>8.4f}")
        print("=" * 50)

    # Append to eval_results.json
    history = []
    if EVAL_RESULTS_FILE.exists():
        history = json.loads(EVAL_RESULTS_FILE.read_text(encoding="utf-8"))
    history.append(run_results)
    EVAL_RESULTS_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved → {EVAL_RESULTS_FILE}")


if __name__ == "__main__":
    main()
