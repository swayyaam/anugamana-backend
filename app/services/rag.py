"""
RAG generation: Claude produces guidance grounded strictly in the retrieved
verse + parent-window commentary. Faithfulness constraint is baked into system prompt.

Graceful degradation: if Claude call fails, generate() returns "" (empty guidance).
The verse is still returned to the user — only the AI commentary is missing.
"""

import json
import os
import pickle
from pathlib import Path
from functools import lru_cache

import anthropic
import structlog
from dotenv import load_dotenv

load_dotenv()

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
logger = structlog.get_logger(__name__)

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
ENRICHED_FILE = DATA_DIR / "gita_enriched.json"
SPARSE_FILE = DATA_DIR / "sparse_index.pkl"

RAG_SYSTEM = """\
You are a compassionate guide helping someone understand the Bhagavad Gita As It Is
by Srila Prabhupada.

Rules you must follow without exception:
1. Use ONLY the verse and commentary provided below — no outside knowledge
2. If the verse does not directly address the user's question, say so honestly
3. Write in warm, clear, modern English — not academic or preachy
4. Do not invent verse references or quote Sanskrit you were not given
5. Keep your response to 3-5 sentences — focused and grounded\
"""


@lru_cache(maxsize=1)
def _load_enriched() -> dict[str, dict]:
    verses = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
    return {v["verse_id"]: v for v in verses}


def _get_parent_chunk(verse_id: str, parent_start: int, parent_end: int) -> str:
    """Reconstruct the parent window from the verse's purport paragraphs."""
    from scripts.indexer import chunk_purport  # reuse chunking logic
    verse = _load_enriched().get(verse_id, {})
    purport = verse.get("purport", "")
    chunks = chunk_purport(purport)
    if not chunks:
        return purport[:1000]  # fallback: first 1000 chars
    window = chunks[parent_start: parent_end + 1]
    return "\n\n".join(window)


def generate(query: str, verse: dict) -> str:
    """
    Generate guidance for a single verse.
    verse dict must have: verse_id, translation, and optionally chunk_index,
    parent_start, parent_end (from purport collection metadata).
    """
    verse_id = verse["verse_id"]
    translation = verse["translation"]

    # Get parent chunk context
    parent_start = int(verse.get("parent_start", 0))
    parent_end = int(verse.get("parent_end", 2))
    commentary = _get_parent_chunk(verse_id, parent_start, parent_end)

    if not commentary:
        # Fallback: use the translation alone
        commentary = f"(No extended commentary available for this verse.)"

    user_message = (
        f"Question: {query}\n\n"
        f"Verse {verse_id} — {translation}\n\n"
        f"Commentary:\n{commentary}"
    )

    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=RAG_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.warning("rag_generation_failed", verse_id=verse_id, error=str(e))
        return ""  # verse still returned; only guidance is missing


def generate_batch(query: str, verses: list[dict]) -> list[str]:
    """Generate guidance for each verse sequentially."""
    return [generate(query, v) for v in verses]
