"""
HyDE (Hypothetical Document Embeddings) + query expansion.

HyDE: generate a hypothetical Gita commentary that would answer the query,
then embed that instead of the raw query. The hypothetical lives in the same
semantic space as the indexed Prabhupada purports → much stronger retrieval.

Query expansion: generate 3 rephrasings to improve recall on short queries.
Both run as parallel Claude calls with independent fallbacks.

Graceful degradation:
  HyDE fails    → use raw query as embed text, log "hyde" in degraded_stages
  expansion fails → use [query] only, log "expansion" in degraded_stages
"""

import asyncio
import os
import anthropic
import structlog
from dotenv import load_dotenv

load_dotenv()

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
logger = structlog.get_logger(__name__)

HYDE_SYSTEM = """\
You are generating a hypothetical passage from Srila Prabhupada's Bhagavad Gita As It Is
commentary that would directly answer the user's question or address their situation.

Write exactly 4 sentences in Prabhupada's commentary style:
- Use his characteristic vocabulary: dharma, karma, the Supreme Lord, the conditioned soul,
  material existence, devotional service, transcendental knowledge, the modes of nature
- Ground the passage in the Gita's actual teachings — no generic self-help language
- Do NOT cite or reference any specific verse number
- Do NOT start with "The Bhagavad Gita says" or similar — write as flowing commentary\
"""

EXPANSION_SYSTEM = """\
Generate 3 alternative phrasings of the user's query that capture the same meaning
but use different vocabulary. Each rephrasing should be a single sentence.

Output exactly 3 lines, one rephrasing per line. No numbering, no explanation.\
"""


def _hyde_call(query: str) -> str:
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=HYDE_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text.strip()


def _expansion_call(query: str) -> list[str]:
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=EXPANSION_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )
    lines = [l.strip() for l in response.content[0].text.strip().splitlines() if l.strip()]
    return lines[:3]


async def transform(query: str) -> tuple[str, list[str], list[str]]:
    """
    Returns (hyde_text, all_queries, degraded_stages).
    HyDE and expansion fail independently — each has its own fallback.
    """
    loop = asyncio.get_event_loop()
    hyde_fut = loop.run_in_executor(None, _hyde_call, query)
    exp_fut = loop.run_in_executor(None, _expansion_call, query)

    results = await asyncio.gather(hyde_fut, exp_fut, return_exceptions=True)
    degraded: list[str] = []

    if isinstance(results[0], Exception):
        logger.warning("hyde_failed", error=str(results[0]))
        degraded.append("hyde")
        hyde_text = query  # fall back to raw query as embed text
    else:
        hyde_text = results[0]

    if isinstance(results[1], Exception):
        logger.warning("expansion_failed", error=str(results[1]))
        degraded.append("expansion")
        expansions: list[str] = []
    else:
        expansions = results[1]

    return hyde_text, [query] + expansions, degraded
