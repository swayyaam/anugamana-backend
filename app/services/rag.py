"""
RAG generation — guidance grounded strictly in the retrieved verse and the
parent window of the purport chunk that actually matched.

Parent-child retrieval (fixed 2026-08-31 — audit E-03)
------------------------------------------------------
Retrieval now carries `parent_start`/`parent_end` from the winning purport chunk
through fusion. Previously those keys were absent and every generation silently
received paragraphs 0-2 of the purport regardless of what matched.

Concurrency (fixed 2026-08-31 — audit E-04)
-------------------------------------------
Generation is concurrent across verses and uses the async client; it used to be a
sequential list comprehension over a blocking client.

Degradation: a failed call yields "" for that verse. The verse itself is still
returned — only the commentary is missing.
"""

import asyncio
import json
from functools import lru_cache

import structlog
from anthropic import AsyncAnthropic

from app.config import ANTHROPIC_API_KEY, ENRICHED_FILE, LLM_MODEL
from app.services.chunking import chunk_purport_cached

logger = structlog.get_logger(__name__)

_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

MAX_CONCURRENT_GENERATIONS = 5

RAG_SYSTEM = """\
You are a compassionate guide helping someone understand the Bhagavad Gita As It Is
by Srila Prabhupada.

Rules you must follow without exception:
1. Use ONLY the verse and commentary provided below — no outside knowledge
2. If the verse does not directly address the user's question, say so honestly
3. Write in warm, clear, modern English — not academic or preachy
4. Do not invent verse references or quote Sanskrit you were not given
5. Speak about the text and its teaching. Never write in the voice of Krishna,
   Prabhupada, or any other person — you are explaining a source, not channelling one
6. Keep your response to 3-5 sentences — focused and grounded\
"""


@lru_cache(maxsize=1)
def _load_enriched() -> dict[str, dict]:
    verses = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
    return {v["verse_id"]: v for v in verses}


def _commentary_for(verse: dict) -> tuple[str, str]:
    """
    Reconstruct the commentary window for a verse.

    Returns (commentary_text, provenance) where provenance is one of
    "parent_window" (the ±1 window around the chunk that matched),
    "purport_head" (no chunk matched — first chunks used), or
    "none" (verse has no purport).
    """
    verse_id = verse["verse_id"]
    record = _load_enriched().get(verse_id, {})
    chunks = chunk_purport_cached(record.get("purport", ""))
    if not chunks:
        return "", "none"

    if "parent_start" in verse and "parent_end" in verse:
        start = max(0, int(verse["parent_start"]))
        end = min(len(chunks) - 1, int(verse["parent_end"]))
        if start <= end:
            return "\n\n".join(chunks[start : end + 1]), "parent_window"

    # No purport chunk was retrieved for this verse — be explicit rather than
    # pretending the head of the purport is a targeted match.
    return "\n\n".join(chunks[:3]), "purport_head"


async def generate(query: str, verse: dict) -> str:
    """Generate guidance for a single verse. Returns "" on failure."""
    commentary, provenance = _commentary_for(verse)
    if not commentary:
        commentary = "(No extended commentary is available for this verse.)"

    user_message = (
        f"Question: {query}\n\n"
        f"Verse {verse['verse_id']} — {verse['translation']}\n\n"
        f"Commentary:\n{commentary}"
    )

    try:
        response = await _client.messages.create(
            model=LLM_MODEL,
            max_tokens=400,
            system=RAG_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )
        logger.debug(
            "rag_generated", verse_id=verse["verse_id"], provenance=provenance
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.warning(
            "rag_generation_failed", verse_id=verse["verse_id"], error=str(e)
        )
        return ""


async def generate_batch(query: str, verses: list[dict]) -> list[str]:
    """Generate guidance for every verse concurrently, order preserved."""
    if not verses:
        return []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)

    async def _bounded(verse: dict) -> str:
        async with semaphore:
            return await generate(query, verse)

    results = await asyncio.gather(
        *(_bounded(v) for v in verses), return_exceptions=True
    )
    return ["" if isinstance(r, BaseException) else r for r in results]
