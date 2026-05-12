"""
POST /search      — full RAG pipeline with query routing + graceful degradation
GET  /metrics     — rolling system metrics
POST /feedback    — thumbs up/down on a response

Query routing detects the query type before the pipeline runs:
  direct_lookup — verse reference ("2.47", "BG 6.5") → skip HyDE/retrieval/reranker
  sanskrit      — Devanagari script → skip HyDE, embed raw query
  semantic      — everything else → full pipeline

Degradation: every stage has a fallback. Never returns HTTP 500.
"""

import asyncio
import re
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

# Confidence threshold — tune after evaluating real query data
MIN_CONFIDENCE = 0.1
LOW_CONFIDENCE_GAP = 0.3

# Query routing patterns
_VERSE_REF_RE = re.compile(
    r'\b(?:bg|gita|verse|ch)?\s*(\d{1,2})[.\s](\d{1,3})\b',
    re.IGNORECASE,
)
# "chapter 2 verse 47" — two separate numbers with "verse" keyword between them
_VERSE_REF_LONG_RE = re.compile(
    r'\bchapter\s+(\d{1,2})\s+verse\s+(\d{1,3})\b',
    re.IGNORECASE,
)
_DEVANAGARI_RE = re.compile(r'[ऀ-ॿ]')


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
    query_route:          str = "semantic"
    retrieval_ms:         int
    rerank_ms:            int
    generation_ms:        int
    total_ms:             int
    response_id:          int | None = None
    degraded_stages:      list[str] = []
    confidence_filtered:  int = 0
    low_confidence:       bool = False


class SearchResponse(BaseModel):
    results: list[VerseResult]
    query_meta: QueryMeta


class FeedbackRequest(BaseModel):
    response_id: int
    rating: int  # +1 or -1


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

def _classify_query(query: str) -> tuple[str, dict]:
    """
    Returns (query_type, meta).
    direct_lookup → {"verse_id": "2.47"}
    sanskrit      → {}
    semantic      → {}
    """
    m = _VERSE_REF_LONG_RE.search(query) or _VERSE_REF_RE.search(query)
    if m:
        return "direct_lookup", {"verse_id": f"{m.group(1)}.{m.group(2)}"}
    if _DEVANAGARI_RE.search(query):
        return "sanskrit", {}
    return "semantic", {}


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

    # 2. Query routing
    query_route, route_meta = _classify_query(query)
    loop = asyncio.get_event_loop()

    # Initialise variables that may be set by either branch
    top_verses: list[dict] = []
    retrieval_ms = 0
    rerank_ms = 0
    confidence_filtered = 0
    low_confidence = False

    # ── DIRECT LOOKUP ──────────────────────────────────────────────────────
    if query_route == "direct_lookup":
        t_retrieval = time.time()
        verse_id = route_meta["verse_id"]
        direct_results = await loop.run_in_executor(
            None, retrieval.retrieve_by_verse_id, verse_id
        )
        retrieval_ms = int((time.time() - t_retrieval) * 1000)

        if not direct_results:
            logger.info("direct_lookup_fallback", verse_id=verse_id)
            query_route = "semantic"  # fall through to semantic path below
        else:
            for v in direct_results:
                v["normalized_score"] = 1.0
                v["cross_score"] = 1.0
            top_verses = direct_results[:top_k]

    # ── SEMANTIC / SANSKRIT ────────────────────────────────────────────────
    if query_route != "direct_lookup":
        # Sanskrit: skip HyDE Claude call, embed raw query directly
        if query_route == "sanskrit":
            hyde_text: str = query
            all_queries: list[str] = [query]
            hyde_degraded: list[str] = []
        else:
            hyde_text, all_queries, hyde_degraded = await hyde.transform(query)
        degraded.extend(hyde_degraded)

        # Retrieval
        t_retrieval = time.time()
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
            response_id = _safe_log_response(query, query, [], None,
                                             int((time.time() - t_start) * 1000))
            return SearchResponse(
                results=[],
                query_meta=QueryMeta(
                    guardrail=guard_result,
                    query_route=query_route,
                    retrieval_ms=retrieval_ms,
                    rerank_ms=0,
                    generation_ms=0,
                    total_ms=int((time.time() - t_start) * 1000),
                    response_id=response_id,
                    degraded_stages=degraded,
                ),
            )

        # Rerank + MMR
        t_rerank = time.time()
        top_verses, rerank_degraded = await loop.run_in_executor(
            None, reranker.rerank, query, candidates
        )
        degraded.extend(rerank_degraded)
        top_verses = top_verses[:top_k]
        rerank_ms = int((time.time() - t_rerank) * 1000)

        # Confidence threshold
        if top_verses:
            scores = [v.get("cross_score", v.get("rrf_score", 0.0)) for v in top_verses]
            min_s, max_s = min(scores), max(scores)
            gap = max_s - min_s

            if gap > 0:
                for verse, s in zip(top_verses, scores):
                    verse["normalized_score"] = round((s - min_s) / gap, 4)
            else:
                for verse in top_verses:
                    verse["normalized_score"] = 1.0

            if gap < LOW_CONFIDENCE_GAP:
                low_confidence = True

            before = len(top_verses)
            top_verses = [v for v in top_verses if v["normalized_score"] >= MIN_CONFIDENCE]
            confidence_filtered = before - len(top_verses)

            if confidence_filtered:
                logger.info("confidence_filtered", dropped=confidence_filtered,
                            gap=round(gap, 3))

    # ── SHARED: RAG + LOGGING + RESPONSE ──────────────────────────────────
    t_gen = time.time()
    guidances = await loop.run_in_executor(
        None, rag.generate_batch, query, top_verses
    )
    generation_ms = int((time.time() - t_gen) * 1000)
    total_ms = int((time.time() - t_start) * 1000)

    verse_ids = [v["verse_id"] for v in top_verses]
    response_id = _safe_log_response(
        query,
        route_meta.get("verse_id", query) if query_route == "direct_lookup" else query,
        verse_ids,
        verse_ids[0] if verse_ids else None,
        total_ms,
    )

    if top_verses and guidances and guidances[0] and response_id:
        threading.Thread(
            target=_run_judge,
            args=(response_id, query, top_verses[0], guidances[0]),
            daemon=True,
        ).start()

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

    logger.info("search_complete", query=query, route=query_route,
                results=len(results), total_ms=total_ms, degraded=degraded)

    return SearchResponse(
        results=results,
        query_meta=QueryMeta(
            guardrail=guard_result,
            query_route=query_route,
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
