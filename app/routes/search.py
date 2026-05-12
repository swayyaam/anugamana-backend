"""
POST /search      — full RAG pipeline with graceful degradation
GET  /metrics     — rolling system metrics
POST /feedback    — thumbs up/down on a response

Degradation: every stage has a fallback. The pipeline never returns HTTP 500.
Failed stages are reported in query_meta.degraded_stages so clients/monitoring
can react appropriately.
"""

import asyncio
import threading
import time

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.limiter import limiter
from app.services import guardrail, hyde, retrieval, reranker, rag
from app.services import feedback_logger, judge

logger = structlog.get_logger(__name__)
router = APIRouter()

OFF_TOPIC_MESSAGE = (
    "Anugamana is designed for spiritual and philosophical guidance. "
    "Try asking about a life situation, an emotion, or a concept from the Bhagavad Gita."
)

# Confidence threshold constants — tune after evaluating real query data
MIN_CONFIDENCE = 0.1   # normalized score below this → drop the verse
LOW_CONFIDENCE_GAP = 0.3  # if best-to-worst gap < this → flag as low_confidence


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class VerseResult(BaseModel):
    verse_id: str
    chapter: int
    verse: int
    devanagari: str
    sanskrit: str
    translation: str
    score: float
    ai_guidance: str


class QueryMeta(BaseModel):
    guardrail:            str
    retrieval_ms:         int
    rerank_ms:            int
    generation_ms:        int
    total_ms:             int
    response_id:          int | None = None
    degraded_stages:      list[str] = []
    confidence_filtered:  int = 0    # verses dropped by confidence threshold
    low_confidence:       bool = False  # True when scores form a tight cluster


class SearchResponse(BaseModel):
    results: list[VerseResult]
    query_meta: QueryMeta


class FeedbackRequest(BaseModel):
    response_id: int
    rating: int  # +1 or -1


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

def _run_judge(response_id: int, query: str, top_verse: dict, guidance: str) -> None:
    try:
        result = judge.judge(
            query=query,
            verse_id=top_verse["verse_id"],
            translation=top_verse["translation"],
            guidance=guidance,
        )
        score = result.get("score")
        if score is not None:
            feedback_logger.update_faith_score(response_id, float(score))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/")
def home():
    return {"message": "Anugamana — Bhagavad Gita semantic search", "status": "online"}


@router.post("/search", response_model=SearchResponse)
@limiter.limit("20/minute")
async def search(request: Request, payload: SearchRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    top_k = max(1, min(payload.top_k, 5))
    t_start = time.time()
    degraded: list[str] = []

    # 1. Guardrail — never raises (fails open internally)
    guard_result = await guardrail.classify(query)
    if guard_result == "off_topic":
        logger.info("guardrail_rejected", query=query)
        raise HTTPException(status_code=422, detail=OFF_TOPIC_MESSAGE)

    # 2. HyDE + expansion — independent fallbacks, returns 3-tuple
    hyde_text, all_queries, hyde_degraded = await hyde.transform(query)
    degraded.extend(hyde_degraded)

    # 3. Retrieval
    t_retrieval = time.time()
    loop = asyncio.get_event_loop()
    try:
        candidates = await loop.run_in_executor(
            None, retrieval.retrieve, hyde_text, all_queries
        )
    except Exception as e:
        logger.error("retrieval_failed", error=str(e))
        candidates = []
        degraded.append("retrieval")
    retrieval_ms = int((time.time() - t_retrieval) * 1000)

    if not candidates:
        response_id = _safe_log_response(query, hyde_text, [], None,
                                         int((time.time() - t_start) * 1000))
        return SearchResponse(
            results=[],
            query_meta=QueryMeta(
                guardrail=guard_result,
                retrieval_ms=retrieval_ms,
                rerank_ms=0,
                generation_ms=0,
                total_ms=int((time.time() - t_start) * 1000),
                response_id=response_id,
                degraded_stages=degraded,
            ),
        )

    # 4. Rerank + MMR — returns 2-tuple with degraded stages
    t_rerank = time.time()
    top_verses, rerank_degraded = await loop.run_in_executor(
        None, reranker.rerank, query, candidates
    )
    degraded.extend(rerank_degraded)
    top_verses = top_verses[:top_k]
    rerank_ms = int((time.time() - t_rerank) * 1000)

    # 5. Confidence threshold — normalize scores to [0,1], drop outliers
    confidence_filtered = 0
    low_confidence = False

    if top_verses:
        scores = [v.get("cross_score", v.get("rrf_score", 0.0)) for v in top_verses]
        min_s, max_s = min(scores), max(scores)
        gap = max_s - min_s

        if gap > 0:
            for verse, s in zip(top_verses, scores):
                verse["normalized_score"] = round((s - min_s) / gap, 4)
        else:
            for verse in top_verses:
                verse["normalized_score"] = 1.0  # all equal scores, keep all

        # Flag tight clusters where no verse stands out clearly
        if gap < LOW_CONFIDENCE_GAP:
            low_confidence = True

        # Drop outliers below threshold
        before = len(top_verses)
        top_verses = [v for v in top_verses if v["normalized_score"] >= MIN_CONFIDENCE]
        confidence_filtered = before - len(top_verses)

        if confidence_filtered:
            logger.info("confidence_filtered", dropped=confidence_filtered,
                        gap=round(gap, 3))

    # 6. RAG generation — per-verse failures return "" internally
    t_gen = time.time()
    guidances = await loop.run_in_executor(
        None, rag.generate_batch, query, top_verses
    )
    generation_ms = int((time.time() - t_gen) * 1000)
    total_ms = int((time.time() - t_start) * 1000)

    # 7. Feedback logging — swallow failures, never block response
    verse_ids = [v["verse_id"] for v in top_verses]
    response_id = _safe_log_response(
        query, hyde_text, verse_ids,
        verse_ids[0] if verse_ids else None, total_ms,
    )

    # 8. Async judge — already daemon thread, already safe
    if top_verses and guidances and guidances[0] and response_id:
        threading.Thread(
            target=_run_judge,
            args=(response_id, query, top_verses[0], guidances[0]),
            daemon=True,
        ).start()

    # 9. Build response
    results = []
    for verse, guidance in zip(top_verses, guidances):
        results.append(VerseResult(
            verse_id=verse["verse_id"],
            chapter=int(verse["chapter"]),
            verse=int(verse["verse"]),
            devanagari=verse.get("devanagari", ""),
            sanskrit=verse.get("sanskrit", ""),
            translation=verse["translation"],
            score=verse.get("normalized_score", 0.0),
            ai_guidance=guidance,
        ))

    logger.info("search_complete", query=query, results=len(results),
                total_ms=total_ms, degraded=degraded,
                confidence_filtered=confidence_filtered)

    return SearchResponse(
        results=results,
        query_meta=QueryMeta(
            guardrail=guard_result,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            response_id=response_id,
            degraded_stages=degraded,
            confidence_filtered=confidence_filtered,
            low_confidence=low_confidence,
        ),
    )


def _safe_log_response(
    query: str,
    hyde_query: str,
    verse_ids: list[str],
    top_verse_id: str | None,
    latency_ms: int,
) -> int | None:
    try:
        return feedback_logger.log_response(
            query=query,
            hyde_query=hyde_query,
            verse_ids=verse_ids,
            top_verse_id=top_verse_id,
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.warning("feedback_log_failed", error=str(e))
        return None


@router.get("/metrics")
def metrics(days: int = 7):
    return feedback_logger.get_metrics(window_days=days)


@router.post("/feedback")
def feedback(payload: FeedbackRequest):
    try:
        feedback_logger.log_feedback(payload.response_id, payload.rating)
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
