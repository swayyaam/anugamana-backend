"""
POST /search — full RAG pipeline endpoint.

Pipeline:
  1. Input guardrail (on-topic check)
  2. HyDE + query expansion (parallel Claude calls)
  3. Hybrid retrieval (dense ChromaDB + sparse BGE-M3, RRF fusion)
  4. Cross-encoder reranking + MMR diversity
  5. RAG generation (Claude, per verse)
  6. Latency logging
"""

import asyncio
import time

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.limiter import limiter
from app.services import guardrail, hyde, retrieval, reranker, rag

logger = structlog.get_logger(__name__)
router = APIRouter()

OFF_TOPIC_MESSAGE = (
    "Anugamana is designed for spiritual and philosophical guidance. "
    "Try asking about a life situation, an emotion, or a concept from the Bhagavad Gita."
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3  # number of verses to return (max 5)


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


class SearchResponse(BaseModel):
    results: list[VerseResult]
    query_meta: QueryMeta


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
        return SearchResponse(
            results=[],
            query_meta=QueryMeta(
                guardrail=guard_result,
                retrieval_ms=retrieval_ms,
                rerank_ms=0,
                generation_ms=0,
                total_ms=int((time.time() - t_start) * 1000),
            ),
        )

    # 4. Rerank + MMR
    t_rerank = time.time()
    top_verses = await loop.run_in_executor(
        None, reranker.rerank, query, candidates
    )
    top_verses = top_verses[:top_k]
    rerank_ms = int((time.time() - t_rerank) * 1000)

    # 5. RAG generation (sequential per verse, Haiku is fast)
    t_gen = time.time()
    guidances = await loop.run_in_executor(
        None, rag.generate_batch, query, top_verses
    )
    generation_ms = int((time.time() - t_gen) * 1000)

    # 6. Build response
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

    total_ms = int((time.time() - t_start) * 1000)
    logger.info(
        "search_complete",
        query=query,
        results=len(results),
        total_ms=total_ms,
    )

    return SearchResponse(
        results=results,
        query_meta=QueryMeta(
            guardrail=guard_result,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        ),
    )
