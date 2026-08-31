"""
HTTP surface. All pipeline logic lives in app/services/pipeline.py so that the
evaluation harness exercises the same code path this endpoint does.

  POST /search    — crisis routing, guardrail, retrieval, generation
  POST /feedback  — thumbs up/down on a response
  GET  /metrics   — rolling operational metrics
  GET  /health    — liveness + which components loaded
"""

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from app.limiter import limiter
from app.services import feedback_logger, judge, pipeline
from app.services.rag import _commentary_for
from app.services.safety import CRISIS_RESPONSE

logger = structlog.get_logger(__name__)
router = APIRouter()

OFF_TOPIC_MESSAGE = (
    "Anugamana is designed for spiritual and philosophical guidance. "
    "Try asking about a life situation, an emotion, or a concept from the Bhagavad Gita."
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


class VerseResult(BaseModel):
    verse_id: str
    chapter: int
    verse: int
    devanagari: str = ""
    sanskrit: str = ""
    translation: str
    score: float = Field(
        ...,
        description="Absolute relevance in [0,1] when score_type is 'cross_encoder'. "
                    "Ordinal only when score_type is 'rrf'.",
    )
    ai_guidance: str = ""


class QueryMeta(BaseModel):
    status: str = "ok"            # ok | off_topic | crisis | no_results
    guardrail: str = "relevant"
    safety: str = "safe"
    query_route: str = "semantic"
    score_type: str = "none"
    retrieval_ms: int = 0
    rerank_ms: int = 0
    generation_ms: int = 0
    total_ms: int = 0
    response_id: int | None = None
    degraded_stages: list[str] = []
    confidence_filtered: int = 0
    low_confidence: bool = False


class SearchResponse(BaseModel):
    results: list[VerseResult] = []
    message: str | None = None
    query_meta: QueryMeta


class FeedbackRequest(BaseModel):
    response_id: int = Field(..., ge=1)
    rating: int = Field(..., description="+1 helpful, -1 not helpful")


# ---------------------------------------------------------------------------
# Background work
# ---------------------------------------------------------------------------

async def _judge_response(response_id: int, query: str, verse: dict, guidance: str) -> None:
    """Score the top result after the user already has their answer."""
    try:
        commentary, _ = _commentary_for(verse)
        scores = await judge.judge(
            query=query,
            verse_id=verse["verse_id"],
            translation=verse["translation"],
            commentary=commentary,
            guidance=guidance,
        )
        if scores.get("score") is not None:
            feedback_logger.update_judge_scores(response_id, scores)
    except Exception as e:
        logger.warning("judge_task_failed", response_id=response_id, error=str(e))


def _log(query: str, result: pipeline.PipelineResult) -> int | None:
    try:
        return feedback_logger.log_response(
            query=query,
            hyde_query=None,
            verse_ids=result.verse_ids,
            top_verse_id=result.verse_ids[0] if result.verse_ids else None,
            latency_ms=result.timings.get("total_ms", 0),
            query_route=result.route,
            degraded=result.degraded,
        )
    except Exception as e:
        logger.warning("feedback_log_failed", error=str(e))
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/")
def home():
    return {"message": "Anugamana — Bhagavad Gita semantic search", "status": "online"}


@router.get("/health")
def health():
    from app.services import retrieval

    components = {}
    try:
        verses_col, purport_col = retrieval._load_chroma()
        components["chroma_verses"] = verses_col.count()
        components["chroma_purport"] = purport_col.count() if purport_col else 0
    except Exception as e:
        components["chroma"] = f"error: {e}"
    return {"status": "online", "components": components}


@router.post("/search", response_model=SearchResponse)
@limiter.limit("20/minute")
async def search(
    request: Request,
    payload: SearchRequest,
    background: BackgroundTasks,
) -> SearchResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    config = pipeline.SERVED
    if payload.top_k != config.top_k:
        from dataclasses import replace
        config = replace(config, top_k=payload.top_k)

    result = await pipeline.run(query, config)

    meta = QueryMeta(
        guardrail=result.guardrail,
        safety=result.safety,
        query_route=result.route,
        score_type=result.score_type,
        retrieval_ms=result.timings.get("retrieval_ms", 0),
        rerank_ms=result.timings.get("rerank_ms", 0),
        generation_ms=result.timings.get("generation_ms", 0),
        total_ms=result.timings.get("total_ms", 0),
        degraded_stages=result.degraded,
        confidence_filtered=result.confidence_filtered,
        low_confidence=result.low_confidence,
    )

    # Crisis and off-topic are ordinary, expected outcomes — not HTTP errors.
    # The previous 422 was indistinguishable from a schema validation failure.
    if result.safety == "crisis":
        logger.warning("crisis_response_served")
        meta.status = "crisis"
        return SearchResponse(results=[], message=CRISIS_RESPONSE, query_meta=meta)

    if result.guardrail == "off_topic":
        meta.status = "off_topic"
        return SearchResponse(results=[], message=OFF_TOPIC_MESSAGE, query_meta=meta)

    response_id = _log(query, result)
    meta.response_id = response_id

    if not result.verses:
        meta.status = "no_results"
        return SearchResponse(
            results=[],
            message="No verse in the text matched that closely. Try describing the "
                    "situation in more detail.",
            query_meta=meta,
        )

    if response_id and result.guidances and result.guidances[0]:
        background.add_task(
            _judge_response, response_id, query, result.verses[0], result.guidances[0]
        )

    results = [
        VerseResult(
            verse_id=verse["verse_id"],
            chapter=int(verse["chapter"]),
            verse=int(verse["verse"]),
            devanagari=verse.get("devanagari", ""),
            sanskrit=verse.get("sanskrit", ""),
            translation=verse["translation"],
            score=round(float(verse.get("relevance", 0.0)), 4),
            ai_guidance=guidance,
        )
        for verse, guidance in zip(result.verses, result.guidances)
    ]

    logger.info(
        "search_complete",
        route=result.route,
        results=len(results),
        total_ms=meta.total_ms,
        degraded=result.degraded,
    )
    return SearchResponse(results=results, query_meta=meta)


@router.get("/metrics")
def metrics(days: int = 7):
    return feedback_logger.get_metrics(window_days=days)


@router.post("/feedback")
@limiter.limit("30/minute")
def feedback(request: Request, payload: FeedbackRequest):
    if payload.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating must be +1 or -1")
    try:
        feedback_logger.log_feedback(payload.response_id, payload.rating)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "ok"}
