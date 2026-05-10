"""
POST /search      — full RAG pipeline
GET  /metrics     — rolling system metrics
POST /feedback    — thumbs up/down on a response
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
    guardrail: str
    retrieval_ms: int
    rerank_ms: int
    generation_ms: int
    total_ms: int
    response_id: int | None = None


class SearchResponse(BaseModel):
    results: list[VerseResult]
    query_meta: QueryMeta


class FeedbackRequest(BaseModel):
    response_id: int
    rating: int  # +1 or -1


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

def _run_judge(
    response_id: int,
    query: str,
    top_verse: dict,
    guidance: str,
) -> None:
    """Run LLM-as-judge and update faith_score. Runs in a background thread."""
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
        pass  # never let judge failure surface to user


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

    # 1. Guardrail
    guard_result = await guardrail.classify(query)
    if guard_result == "off_topic":
        logger.info("guardrail_rejected", query=query)
        raise HTTPException(status_code=422, detail=OFF_TOPIC_MESSAGE)

    # 2. HyDE + expansion
    hyde_text, all_queries = await hyde.transform(query)

    # 3. Retrieval
    t_retrieval = time.time()
    loop = asyncio.get_event_loop()
    candidates = await loop.run_in_executor(
        None, retrieval.retrieve, hyde_text, all_queries
    )
    retrieval_ms = int((time.time() - t_retrieval) * 1000)

    if not candidates:
        response_id = feedback_logger.log_response(
            query=query, hyde_query=hyde_text,
            verse_ids=[], top_verse_id=None,
            latency_ms=int((time.time() - t_start) * 1000),
        )
        return SearchResponse(
            results=[],
            query_meta=QueryMeta(
                guardrail=guard_result,
                retrieval_ms=retrieval_ms,
                rerank_ms=0, generation_ms=0,
                total_ms=int((time.time() - t_start) * 1000),
                response_id=response_id,
            ),
        )

    # 4. Rerank + MMR
    t_rerank = time.time()
    top_verses = await loop.run_in_executor(
        None, reranker.rerank, query, candidates
    )
    top_verses = top_verses[:top_k]
    rerank_ms = int((time.time() - t_rerank) * 1000)

    # 5. RAG generation
    t_gen = time.time()
    guidances = await loop.run_in_executor(
        None, rag.generate_batch, query, top_verses
    )
    generation_ms = int((time.time() - t_gen) * 1000)
    total_ms = int((time.time() - t_start) * 1000)

    # 6. Log response
    verse_ids = [v["verse_id"] for v in top_verses]
    response_id = feedback_logger.log_response(
        query=query,
        hyde_query=hyde_text,
        verse_ids=verse_ids,
        top_verse_id=verse_ids[0] if verse_ids else None,
        latency_ms=total_ms,
    )

    # 7. Async faithfulness judge (non-blocking)
    if top_verses and guidances:
        threading.Thread(
            target=_run_judge,
            args=(response_id, query, top_verses[0], guidances[0]),
            daemon=True,
        ).start()

    # 8. Build response
    results = []
    for verse, guidance in zip(top_verses, guidances):
        results.append(VerseResult(
            verse_id=verse["verse_id"],
            chapter=int(verse["chapter"]),
            verse=int(verse["verse"]),
            devanagari=verse.get("devanagari", ""),
            sanskrit=verse.get("sanskrit", ""),
            translation=verse["translation"],
            score=round(verse.get("cross_score", verse.get("rrf_score", 0.0)), 4),
            ai_guidance=guidance,
        ))

    logger.info("search_complete", query=query, results=len(results), total_ms=total_ms)

    return SearchResponse(
        results=results,
        query_meta=QueryMeta(
            guardrail=guard_result,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            response_id=response_id,
        ),
    )


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
