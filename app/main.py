import structlog
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification
from pinecone import Pinecone
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from transformers import AutoTokenizer
from upstash_redis import Redis

from app import state
from app.config import (
    ALLOWED_ORIGINS,
    ANTHROPIC_API_KEY,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    RERANK_MODEL,
    UPSTASH_REDIS_REST_TOKEN,
    UPSTASH_REDIS_REST_URL,
)
from app.limiter import limiter
from app.routes.search import router as search_router

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup_begin")

    try:
        logger.info("loading_embedding_model", model=EMBEDDING_MODEL)
        state.tokenizer_emb = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        state.embedder = ORTModelForFeatureExtraction.from_pretrained(
            EMBEDDING_MODEL, subfolder="onnx", file_name="model_quantized.onnx"
        )
    except Exception as e:
        logger.error("embedding_model_load_failed", error=str(e))

    try:
        logger.info("loading_reranking_model", model=RERANK_MODEL)
        state.tokenizer_rerank = AutoTokenizer.from_pretrained(RERANK_MODEL)
        state.reranker = ORTModelForSequenceClassification.from_pretrained(
            RERANK_MODEL, subfolder="onnx", file_name="model_quantized.onnx"
        )
    except Exception as e:
        logger.error("reranker_load_failed", error=str(e))

    try:
        logger.info("connecting_pinecone")
        state.pc_index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX)
        logger.info("pinecone_connected")
    except Exception as e:
        logger.error("pinecone_connection_failed", error=str(e))

    try:
        state.claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
        logger.info("claude_connected") if state.claude else logger.warning("anthropic_api_key_missing")
    except Exception as e:
        logger.error("claude_client_init_error", error=str(e))

    try:
        state.redis = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)
    except Exception as e:
        logger.error("redis_init_failed", error=str(e))

    logger.info("startup_complete")
    yield
    logger.info("shutdown")


app = FastAPI(lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)
