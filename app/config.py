"""
Runtime configuration.

Only settings the live pipeline actually reads belong here. The Pinecone /
MiniLM / Upstash settings that used to live in this file described a v1
architecture that no longer exists and were removed on 2026-08-31.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
SPARSE_FILE = DATA_DIR / "sparse_index.pkl"
ENRICHED_FILE = DATA_DIR / "gita_enriched.json"
FEEDBACK_DB = DATA_DIR / "feedback.db"

# --- models ---------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Generation and the cheap classifier stages.
LLM_MODEL = "claude-haiku-4-5-20251001"
# The judge must not be the same model that produced the text it grades —
# self-preference bias is well documented. Kept deliberately separate.
JUDGE_MODEL = "claude-sonnet-4-5-20250929"

# --- secrets --------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- api ------------------------------------------------------------------
# The web app's dev server runs on 3000 (vite.config.ts), not Vite's default
# 5173. Both are allowed so the frontend works whichever port it lands on;
# override with ALLOWED_ORIGINS in production.
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

# --- retrieval tuning -----------------------------------------------------
TOP_K = 15          # candidates fetched per individual search
RRF_K = 60          # reciprocal-rank-fusion constant
TOP_VERSES = 10     # verses handed to the reranker
TOP_RESULTS = 5     # verses surviving the MMR pass
MMR_LAMBDA = 0.7    # 0 = pure diversity, 1 = pure relevance

# --- confidence -----------------------------------------------------------
# FITTED, not guessed. eval/calibrate.py scores 4,000 judged (query, verse)
# pairs with the cross-encoder and measures how well it separates relevant
# (grade >= 2) from irrelevant verses:
#
#     ROC AUC 0.4579  — WORSE THAN RANDOM
#     relevant     n=687   median 0.0000  mean 0.0005
#     not relevant n=3313  median 0.0000  mean 0.0003
#
# ms-marco-MiniLM-L-6-v2 is trained on web-search passages and carries no usable
# signal on scripture. Its "relevance probability" is noise, so:
#   * it is switched off in the served pipeline (see pipeline.SERVED)
#   * MIN_RELEVANCE stays 0.0 — dropping results on a score with AUC below 0.5
#     is precisely what caused audit defect E-02
#   * LOW_CONFIDENCE_RELEVANCE is inert while no calibrated scorer is active
#
# Re-run eval/calibrate.py after swapping in a domain-tuned reranker; a threshold
# becomes defensible above AUC ~0.75.
MIN_RELEVANCE = 0.0
LOW_CONFIDENCE_RELEVANCE = 0.35

# --- Sarvam AI (Indic language stack) --------------------------------------
# Base URL and the auth header name are per Sarvam's published API reference.
# Model identifiers and voices are config, not constants, because they version
# independently of this codebase — scripts/verify_sarvam.py checks them against
# the live API and reports drift rather than failing silently at request time.
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_BASE_URL = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai")
SARVAM_AUTH_HEADER = "api-subscription-key"
SARVAM_TIMEOUT_S = float(os.getenv("SARVAM_TIMEOUT_S", "20"))

SARVAM_TRANSLATE_MODEL = os.getenv("SARVAM_TRANSLATE_MODEL", "mayura:v1")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "ritu")
SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-105b-conversations")
#: Chat lives under /v1; the other endpoints do not. Verified 2026-08-31.
SARVAM_CHAT_PATH = "/v1/chat/completions"

#: BCP-47 codes Sarvam accepts. English is the pivot language: the corpus,
#: the enrichment and the purports are all English, so retrieval happens there.
PIVOT_LANGUAGE = "en-IN"
SUPPORTED_LANGUAGES = {
    "en-IN": "English",   "hi-IN": "Hindi",     "bn-IN": "Bengali",
    "ta-IN": "Tamil",     "te-IN": "Telugu",    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",  "kn-IN": "Kannada",   "ml-IN": "Malayalam",
    "pa-IN": "Punjabi",   "od-IN": "Odia",
}

#: Which cross-lingual strategy the served pipeline uses. The alternatives are
#: measured as ablation conditions — see eval/conditions.py.
#:   "translate"  query -> English via Mayura, then the English pipeline
#:   "direct"     embed the Indic query as-is (BGE-M3 is multilingual)
#:   "both"       fuse both query forms through RRF
MULTILINGUAL_STRATEGY = os.getenv("MULTILINGUAL_STRATEGY", "translate")

SARVAM_ENABLED = bool(SARVAM_API_KEY)
