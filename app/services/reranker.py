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

import threading
from functools import lru_cache

import numpy as np
import structlog
from sentence_transformers import CrossEncoder

from app.config import MMR_LAMBDA, RERANK_MODEL, TOP_RESULTS

logger = structlog.get_logger(__name__)


_CE_LOCK = threading.Lock()
_PREDICT_LOCK = threading.Lock()
_cross_encoders: dict[str, CrossEncoder] = {}


def _load_cross_encoder(model_name: str = RERANK_MODEL) -> CrossEncoder:
    """
    Serialised construction, cached per model — see retrieval._load_model.

    Parameterised by model so a reranker can be swapped as an ablation
    condition rather than a code change. ms-marco-MiniLM measured ROC AUC
    0.4579 on this corpus (worse than random) and was removed from the served
    pipeline; any replacement has to earn its place the same way.
    """
    cached = _cross_encoders.get(model_name)
    if cached is not None:
        return cached
    with _CE_LOCK:
        cached = _cross_encoders.get(model_name)
        if cached is None:
            cached = CrossEncoder(model_name, device=_best_device())
            _cross_encoders[model_name] = cached
        return cached


@lru_cache(maxsize=1)
def _load_embed_model():
    from app.services.retrieval import _load_model
    return _load_model()


def _best_device() -> str | None:
    """
    Prefer Apple Silicon's GPU when present. Not a micro-optimisation: on CPU a
    568M-parameter reranker took so long over the benchmark that a comparison run
    appeared to hang, while on MPS the same work takes under a minute.
    Returns None so sentence-transformers picks for itself elsewhere.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return None


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

    # Relevance for the MMR trade-off comes from the candidates' current order,
    # so MMR cannot smuggle the cross-encoder's ranking back in when that has
    # deliberately been switched off.
    scores = np.array(
        [1.0 / (i + 1) for i in range(len(candidates))], dtype=float
    )

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
    cross_encoder_reorders: bool = True,
    reranker_model: str = RERANK_MODEL,
    use_mmr: bool = True,
    top_n: int = TOP_RESULTS,
) -> tuple[list[dict], list[str], str]:
    """
    Score, diversify, and calibrate.

    Returns (verses, degraded_stages, score_type) where score_type is
    "cross_encoder" (relevance is an absolute probability, safe to threshold) or
    "rrf" (relevance is ordinal only — never threshold it).

    `cross_encoder_reorders` separates two jobs the cross-encoder was doing at
    once. Measured on the benchmark, letting it *reorder* costs 0.0379 nDCG@10
    (p < 0.001, losing on 238 queries) because ms-marco is badly out of domain
    here. But it is still the only source of an absolute relevance probability,
    which the API needs for a meaningful score and a usable confidence
    threshold. With this False we keep the calibrated score and drop the
    reordering, which is what the evidence supports. See RESULTS.md section 6.
    """
    if not verses:
        return [], [], "none"

    degraded: list[str] = []
    score_type = "cross_encoder"

    if use_cross_encoder:
        try:
            cross_encoder = _load_cross_encoder(reranker_model)
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

    if score_type == "cross_encoder" and not cross_encoder_reorders:
        # Keep fusion order; the cross-encoder contributes calibration only.
        ranked = sorted(verses, key=lambda v: v.get("rrf_score", 0.0), reverse=True)
    else:
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
