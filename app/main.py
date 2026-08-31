import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import ALLOWED_ORIGINS
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
    """Warm the heavy models so the first user request is not the slow one."""
    logger.info("startup_begin")

    try:
        from app.services.retrieval import (
            _load_chroma,
            _load_model,
            _load_sparse_default,
        )
        _load_model()
        _load_chroma()
        _load_sparse_default()
        logger.info("retrieval_models_loaded")
    except Exception as e:
        logger.error("retrieval_warmup_failed", error=str(e))

    try:
        from app.services.reranker import _load_cross_encoder
        _load_cross_encoder()
        logger.info("reranker_loaded")
    except Exception as e:
        logger.error("reranker_warmup_failed", error=str(e))

    try:
        from app.services.feedback_logger import init_db
        init_db()
        logger.info("feedback_db_ready")
    except Exception as e:
        logger.error("feedback_db_init_failed", error=str(e))

    logger.info("startup_complete")
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Anugamana",
    description="Semantic search and RAG over the Bhagavad-gita As It Is.",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(search_router)
