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
