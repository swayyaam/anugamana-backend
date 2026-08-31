"""
Cross-encoder reranking + MMR diversity + absolute relevance calibration.

Cross-encoder (ms-marco-MiniLM-L-6-v2) reads (query, translation) jointly, which
is substantially more accurate than comparing vectors independently.

MMR (Maximal Marginal Relevance) then trades a little relevance for diversity so
five near-identical verses are not returned.

Calibration (added 2026-08-31 — audit E-02)
-------------------------------------------
The route used to min-max normalise scores *within the result set*, which forced
the lowest-ranked result to exactly 0.0 — always below the drop threshold, so a
request for three verses could never return three. Worse, the top result was
forced to 1.0 whether it was an excellent match or noise, so the score shown to
users carried no information about relevance at all.

Scores are now mapped through a sigmoid to an absolute relevance probability that
is comparable across queries. When the cross-encoder is unavailable we fall back
to RRF ordering and report `score_type="rrf"` so downstream code knows the value
is ordinal, not calibrated, and must not be thresholded.
"""

from functools import lru_cache

import numpy as np
import structlog
from sentence_transformers import CrossEncoder

from app.config import MMR_LAMBDA, RERANK_MODEL, TOP_RESULTS

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _load_cross_encoder() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=1)
def _load_embed_model():
    from app.services.retrieval import _load_model
    return _load_model()


def _sigmoid(x: float) -> float:
    # ms-marco cross-encoders emit an unbounded logit; the sigmoid recovers the
    # relevance probability the model was trained against.
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


def _mmr(candidates: list[dict], top_n: int, lambda_: float) -> list[dict]:
    """Select `top_n` results balancing relevance against redundancy."""
    if len(candidates) <= top_n:
        return candidates

    model = _load_embed_model()
    texts = [c["translation"] for c in candidates]
    vecs = model.encode(
        texts, batch_size=len(texts), max_length=256, return_dense=True
    )["dense_vecs"]

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms == 0, 1, norms)

    scores = np.array([c["relevance"] for c in candidates], dtype=float)

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_n:
        if not selected:
            best = max(remaining, key=lambda i: scores[i])
        else:
            selected_vecs = vecs[selected]
            best, best_score = remaining[0], -np.inf
            for i in remaining:
                redundancy = float(np.max(vecs[i] @ selected_vecs.T))
                mmr_score = lambda_ * scores[i] - (1 - lambda_) * redundancy
                if mmr_score > best_score:
                    best_score, best = mmr_score, i
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]


def rerank(
    query: str,
    verses: list[dict],
    *,
    use_cross_encoder: bool = True,
    use_mmr: bool = True,
    top_n: int = TOP_RESULTS,
) -> tuple[list[dict], list[str], str]:
    """
    Score, diversify, and calibrate.

    Returns (verses, degraded_stages, score_type) where score_type is
    "cross_encoder" (relevance is an absolute probability, safe to threshold) or
    "rrf" (relevance is ordinal only — never threshold it).
    """
    if not verses:
        return [], [], "none"

    degraded: list[str] = []
    score_type = "cross_encoder"

    if use_cross_encoder:
        try:
            cross_encoder = _load_cross_encoder()
            pairs = [(query, v["translation"]) for v in verses]
            for verse, logit in zip(verses, cross_encoder.predict(pairs)):
                verse["cross_score"] = float(logit)
                verse["relevance"] = _sigmoid(float(logit))
        except Exception as e:
            logger.warning("cross_encoder_failed", error=str(e))
            degraded.append("reranker")
            score_type = "rrf"
    else:
        score_type = "rrf"

    if score_type == "rrf":
        # Ordinal fallback: preserve ranking, but make it obvious the number is
        # not a calibrated probability by spacing results evenly below 1.0.
        ordered = sorted(verses, key=lambda v: v.get("rrf_score", 0.0), reverse=True)
        for position, verse in enumerate(ordered):
            verse["cross_score"] = verse.get("rrf_score", 0.0)
            verse["relevance"] = round(1.0 / (position + 1), 4)

    ranked = sorted(verses, key=lambda v: v["relevance"], reverse=True)

    if use_mmr:
        try:
            ranked = _mmr(ranked, top_n, MMR_LAMBDA)
        except Exception as e:
            logger.warning("mmr_failed", error=str(e))
            degraded.append("mmr")
            ranked = ranked[:top_n]
    else:
        ranked = ranked[:top_n]

    return ranked, degraded, score_type
