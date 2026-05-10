"""
Cross-encoder reranking + MMR diversity.

Cross-encoder (ms-marco-MiniLM-L-6-v2): reads (query, translation) together,
far more accurate than vector cosine similarity alone.

MMR (Maximal Marginal Relevance): after reranking, select top 5 results
that balance relevance vs diversity — prevents returning 5 near-identical verses.
"""

from functools import lru_cache
import numpy as np
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MMR_LAMBDA = 0.7        # 0 = pure diversity, 1 = pure relevance
TOP_RESULTS = 5


@lru_cache(maxsize=1)
def _load_cross_encoder() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=1)
def _load_embed_model() -> BGEM3FlagModel:
    # Reuse BGE-M3 already loaded in retrieval module
    from app.services.retrieval import _load_model
    return _load_model()


def _mmr(
    candidates: list[dict],
    query: str,
    top_n: int,
    lambda_: float,
) -> list[dict]:
    """
    Select top_n diverse results from candidates using MMR.
    candidates must have 'cross_score' and 'translation' fields.
    """
    if len(candidates) <= top_n:
        return candidates

    model = _load_embed_model()

    # Embed all translations + query for similarity computation
    texts = [c["translation"] for c in candidates]
    vecs = model.encode(texts, batch_size=len(texts), max_length=256, return_dense=True)["dense_vecs"]

    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    vecs = vecs / norms

    scores = np.array([c["cross_score"] for c in candidates])
    # Normalize scores to [0,1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(top_n):
        if not remaining:
            break
        if not selected_indices:
            # First: pick highest relevance
            best = max(remaining, key=lambda i: scores[i])
        else:
            # MMR: relevance - max similarity to already selected
            selected_vecs = vecs[selected_indices]
            best_score = -np.inf
            best = remaining[0]
            for i in remaining:
                sim_to_selected = float(np.max(vecs[i] @ selected_vecs.T))
                mmr_score = lambda_ * scores[i] - (1 - lambda_) * sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = i
        selected_indices.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected_indices]


def rerank(query: str, verses: list[dict]) -> list[dict]:
    """
    Cross-encode (query, translation) pairs, then apply MMR.
    Returns top TOP_RESULTS diverse, relevant verses.
    """
    if not verses:
        return []

    cross_encoder = _load_cross_encoder()

    pairs = [(query, v["translation"]) for v in verses]
    scores = cross_encoder.predict(pairs)

    for verse, score in zip(verses, scores):
        verse["cross_score"] = float(score)

    # Sort by cross-encoder score
    ranked = sorted(verses, key=lambda v: v["cross_score"], reverse=True)

    # MMR diversity pass
    diverse = _mmr(ranked, query, TOP_RESULTS, MMR_LAMBDA)

    return diverse
