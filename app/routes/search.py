import asyncio
import structlog
from fastapi import APIRouter, HTTPException, Request

from app import state
from app.limiter import limiter
from app.models import SearchRequest
from app.services.embedder import encode_query, rerank_pairs
from app.services.llm import generate_advice
from app.services import cache

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/")
def home():
    status = "Online" if state.embedder and state.pc_index else "Maintenance Mode (Models Loading)"
    return {"message": "Anugamana API: Pinecone Search + Re-Ranking + RAG", "status": status}


@router.post("/search")
@limiter.limit("15/minute")
async def search_verses(request: Request, payload: SearchRequest):
    if not state.embedder or not state.pc_index or not state.reranker:
        raise HTTPException(status_code=503, detail="Search services are initializing. Please try again in a few seconds.")

    try:
        cache_key = cache.make_cache_key(payload.query)
        cached = cache.get_cached(cache_key)
        if cached:
            logger.info("cache_hit", query=payload.query)
            return cached

        logger.info("cache_miss", query=payload.query)

        query_embedding = await asyncio.to_thread(encode_query, payload.query)

        filter_dict = {"chapter": {"$eq": payload.chapter}} if payload.chapter else None
        pc_results = await asyncio.to_thread(
            state.pc_index.query,
            vector=query_embedding,
            top_k=payload.limit * 2,
            include_metadata=True,
            filter=filter_dict,
        )

        initial_results = [
            {
                "id": match["id"],
                "chapter": match["metadata"].get("chapter"),
                "verse": match["metadata"].get("verse"),
                "text": match["metadata"].get("text", ""),
                "translation": match["metadata"].get("translation", ""),
                "meaning": match["metadata"].get("meaning", ""),
                "score": match["score"],
            }
            for match in pc_results["matches"]
        ]

        if not initial_results:
            return {"results": []}

        rerank_texts = [f"{r['translation']} {r['meaning']}" for r in initial_results]
        cross_scores = await asyncio.to_thread(rerank_pairs, payload.query, rerank_texts)

        top_results = sorted(
            [{"score": float(s), "data": initial_results[i]} for i, s in enumerate(cross_scores)],
            key=lambda x: x["score"],
            reverse=True,
        )[:payload.limit]

        rag_advice = None
        if top_results and payload.limit == 1:
            try:
                top = top_results[0]["data"]
                rag_advice = await generate_advice(
                    payload.query,
                    f"{top['translation']} {top['meaning']}",
                )
            except Exception:
                logger.warning("rag_advice_failed_after_retries")

        final_results = []
        for i, item in enumerate(top_results):
            d = item["data"]
            entry = {
                "text": d.get("text", ""),
                "metadata": {
                    "chapter": d.get("chapter"),
                    "verse": d.get("verse"),
                    "text": d.get("text", ""),
                    "translation": d.get("translation", ""),
                    "meaning": d.get("meaning", ""),
                },
                "score": item["score"],
            }
            if i == 0 and rag_advice:
                entry["metadata"]["ai_advice"] = rag_advice
            final_results.append(entry)

        final_response = {"results": final_results}
        cache.set_cached(cache_key, final_response)
        return final_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("internal_search_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the search.")
