import asyncio
import hashlib
import json
import os
from typing import Optional, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import numpy as np
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables first
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification
from pinecone import Pinecone
from google import genai
from upstash_redis import Redis

# Security: Rate Limiting Imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------------- CONFIGURATION ---------------- #
EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# Security: Structured JSON Logger Setup
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("gemini_connected")
    else:
        client = None
        logger.warning("gemini_api_key_missing")
except Exception as e:
    logger.error("gemini_client_init_error", error=str(e))
    client = None

# Initialize Upstash Redis
redis = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
)

# ---------------- GLOBAL STATE ---------------- #
# Initialize as None. They will be populated in lifespan.
embedder: Optional[Any] = None
reranker: Optional[Any] = None
pc_index: Optional[Any] = None
tokenizer_emb: Optional[Any] = None
tokenizer_rerank: Optional[Any] = None

# ---------------- LIFESPAN MANAGER ---------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager handles startup and shutdown events.
    Models and DB connections are loaded here to prevent import-time blocking/crashes.
    """
    global embedder, reranker, pc_index, tokenizer_emb, tokenizer_rerank

    logger.info("startup_begin")

    # 1. Load Quantized Embedding Model
    try:
        logger.info("loading_embedding_model", model=EMBEDDING_MODEL)
        tokenizer_emb = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        embedder = ORTModelForFeatureExtraction.from_pretrained(
            EMBEDDING_MODEL, subfolder="onnx", file_name="model_quantized.onnx"
        )
    except Exception as e:
        logger.error("embedding_model_load_failed", error=str(e))

    # 2. Load Quantized Re-ranking Model
    try:
        logger.info("loading_reranking_model", model=RERANK_MODEL)
        tokenizer_rerank = AutoTokenizer.from_pretrained(RERANK_MODEL)
        reranker = ORTModelForSequenceClassification.from_pretrained(
            RERANK_MODEL, subfolder="onnx", file_name="model_quantized.onnx"
        )
    except Exception as e:
        logger.error("reranker_load_failed", error=str(e))

    # 3. Connect to Pinecone
    try:
        logger.info("connecting_pinecone")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pc_index = pc.Index("anugamana")
        logger.info("pinecone_connected")
    except Exception as e:
        logger.error("pinecone_connection_failed", error=str(e))

    logger.info("startup_complete")
    
    yield  # Control is yielded to the application
    
    # Shutdown logic (if any cleanup is needed)
    logger.info("shutdown")

# ---------------- INITIALIZATION ---------------- #
# Security: Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# Pass lifespan to FastAPI
app = FastAPI(lifespan=lifespan)

# Security: Register Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security: Secure CORS
origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS = origins_env.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DATA MODELS ---------------- #
class SearchRequest(BaseModel):
    # Security: Input Validation (Max length and Range Bounds)
    query: str = Field(..., max_length=500)
    limit: int = Field(default=5, ge=1, le=20) # Max 20 results to prevent OOM
    chapter: Optional[int] = Field(default=None, ge=1, le=18) # Only 18 chapters exist

# ---------------- HELPER FUNCTIONS ---------------- #

def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to token embeddings, weighted by attention mask."""
    token_embeddings = model_output[0]  # (batch, seq_len, hidden)
    mask_expanded = np.expand_dims(attention_mask, axis=-1)  # (batch, seq_len, 1)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def encode_query(text: str) -> list[float]:
    """Encode a single query string into a normalized embedding vector."""
    inputs = tokenizer_emb(text, padding=True, truncation=True, return_tensors="np")
    outputs = embedder(**inputs)
    embedding = mean_pooling(outputs, inputs["attention_mask"])
    # L2 normalize
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.clip(norm, a_min=1e-9, a_max=None)
    return embedding[0].tolist()


def rerank_pairs(query: str, texts: list[str]) -> list[float]:
    """Score query-text pairs using the quantized cross-encoder."""
    inputs = tokenizer_rerank(
        [query] * len(texts), texts,
        padding=True, truncation=True, return_tensors="np",
    )
    outputs = reranker(**inputs)
    logits = outputs.logits  # (batch, 1) or (batch, num_labels)
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0].tolist()
    return logits.tolist()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def generate_advice(query: str, verse_text: str):
    """
    Uses Gemini (New SDK) to generate personalized advice.
    Retries up to 3 times with exponential backoff on failure.
    """
    if not client:
        return None

    # Security: Use System Instructions for persona (prevents prompt injection)
    system_instruction = (
        "You are Lord Krishna, a wise and compassionate spiritual guide from the Bhagavad Gita. "
        "You speak with warmth and empathy. You always ground your advice in the verse provided. "
        "Keep your response under 100 words."
    )

    user_prompt = (
        f"The user asked the following question:\n"
        f"```\n{query}\n```\n\n"
        f"The Bhagavad Gita says:\n"
        f"\"{verse_text}\"\n\n"
        f"Explain briefly how this verse answers their question and offer one actionable piece of advice."
    )

    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config={"system_instruction": system_instruction},
        )
        return response.text
    except Exception as e:
        logger.error("llm_error", error=str(e))
        raise  # Re-raise so tenacity can retry

# ---------------- API ENDPOINTS ---------------- #

@app.get("/")
def home():
    status = "Online" if embedder and pc_index else "Maintenance Mode (Models Loading)"
    return {"message": "Anugamana API: Pinecone Search + Re-Ranking + RAG", "status": status}

@app.post("/search")
@limiter.limit("15/minute") # Security: Rate Limit applied
async def search_verses(request: Request, payload: SearchRequest):
    # Check if models are ready
    if not embedder or not pc_index or not reranker:
        raise HTTPException(status_code=503, detail="Search services are initializing. Please try again in a few seconds.")

    try:
        # 1. Normalize and hash the query to create a unique Redis key
        normalized_query = payload.query.lower().strip()
        query_hash = hashlib.sha256(normalized_query.encode('utf-8')).hexdigest()
        cache_key = f"search_cache:{query_hash}"

        # 2. Check Redis for a cached response
        logger.info("checking_cache", query=payload.query)
        cached_result = redis.get(cache_key)

        if cached_result:
            logger.info("cache_hit", query=payload.query)
            return cached_result if isinstance(cached_result, dict) else json.loads(cached_result)

        logger.info("cache_miss", query=payload.query)

        # --- VECTOR SEARCH ---
        logger.info("searching_pinecone", query=payload.query)

        # 3. Embed the query
        query_embedding = await asyncio.to_thread(encode_query, payload.query)

        # 4. Query Pinecone
        # We fetch limit * 2 to give the CrossEncoder re-ranker more options to evaluate
        filter_dict = {"chapter": {"$eq": payload.chapter}} if payload.chapter else None
        pc_results = await asyncio.to_thread(
            pc_index.query,
            vector=query_embedding,
            top_k=payload.limit * 2,
            include_metadata=True,
            filter=filter_dict
        )

        # 5. Format results for the re-ranker
        initial_results = []
        for match in pc_results['matches']:
            meta = match['metadata']
            initial_results.append({
                "id": match['id'],
                "chapter": meta.get("chapter"),
                "verse": meta.get("verse"),
                "text": meta.get("text", ""),
                "translation": meta.get("translation", ""),
                "meaning": meta.get("meaning", ""),
                "score": match['score']
            })

        if not initial_results:
            return {"results": []}

        # --- RE-RANKING ---
        rerank_texts = []
        for item in initial_results:
            rerank_texts.append(f"{item['translation']} {item['meaning']}")

        cross_scores = await asyncio.to_thread(rerank_pairs, payload.query, rerank_texts)

        scored_results = []
        for i, score in enumerate(cross_scores):
            scored_results.append({
                "score": float(score),
                "data": initial_results[i]
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)

        # Format final results
        final_results = []
        top_results = scored_results[:payload.limit]

        rag_advice = None
        if top_results and payload.limit == 1:
            top_verse = top_results[0]
            try:
                rag_advice = await generate_advice(
                    payload.query,
                    f"{top_verse['data']['translation']} {top_verse['data']['meaning']}"
                )
            except Exception:
                logger.warning("rag_advice_failed_after_retries")
                rag_advice = None

        for item in top_results:
            d = item["data"]
            res = {
                "text": d.get("text", ""),
                "metadata": {
                    "chapter": d.get("chapter"),
                    "verse": d.get("verse"),
                    "text": d.get("text", ""),
                    "translation": d.get("translation", ""),
                    "meaning": d.get("meaning", ""),
                },
                "score": item["score"]
            }
            if rag_advice and item == top_results[0]:
                res["metadata"]["ai_advice"] = rag_advice

            final_results.append(res)

        # 6. Construct final response and cache it
        final_response = {"results": final_results}

        logger.info("saving_to_cache", query=payload.query)
        redis.set(cache_key, json.dumps(final_response), ex=86400)  # TTL: 24 hours

        return final_response

    except HTTPException:
        raise
    except Exception as e:
        # Security: Prevent Information Leakage
        logger.error("internal_search_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the search.")