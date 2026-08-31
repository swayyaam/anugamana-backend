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
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

# --- retrieval tuning -----------------------------------------------------
TOP_K = 15          # candidates fetched per individual search
RRF_K = 60          # reciprocal-rank-fusion constant
TOP_VERSES = 10     # verses handed to the reranker
TOP_RESULTS = 5     # verses surviving the MMR pass
MMR_LAMBDA = 0.7    # 0 = pure diversity, 1 = pure relevance

# --- confidence -----------------------------------------------------------
# Cross-encoder logits are mapped through a sigmoid to an absolute relevance
# probability.
#
# MEASURED 2026-08-31: ms-marco-MiniLM-L-6-v2 is severely out of domain on this
# corpus. On a representative query its logits over the top-10 candidates ran
# from -2.5 to -11.3, i.e. calibrated probabilities of 0.07 down to ~0.0000 —
# it considers *every* Gita verse irrelevant, because nothing in scripture looks
# like a relevant MS MARCO web passage.
#
# Consequences, both deliberate:
#   * MIN_RELEVANCE is 0.0 — we do not silently delete results using a threshold
#     that has not been fitted. Dropping on an uncalibrated score is what caused
#     audit E-02. Fit it with eval/calibrate.py once graded judgments exist, then
#     raise this.
#   * LOW_CONFIDENCE_RELEVANCE still flags weak results to the client, which is
#     honest without being destructive.
#
# The out-of-domain reranker is itself a measurable question: conditions C8 vs C9
# in the ablation grid test whether this cross-encoder helps or hurts ranking.
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
