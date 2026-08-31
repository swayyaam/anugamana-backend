"""
HyDE (Hypothetical Document Embeddings) + query expansion.

HyDE: generate a hypothetical Prabhupada-style commentary that would answer the
query, then embed *that* instead of the raw query. The hypothetical lands in the
same region of vector space as the indexed purports, which is the whole point —
a casual modern query does not.

Query expansion: 3 rephrasings to widen lexical coverage for the sparse arm.

Both run concurrently with independent fallbacks:
  HyDE fails      → raw query is used as embed text, "hyde" logged as degraded
  expansion fails → [query] only, "expansion" logged as degraded

Disk cache (added 2026-08-31)
-----------------------------
HyDE generations are cached to data/cache/hyde/ keyed by (model, prompt version,
query). The evaluation harness re-runs the same queries across a dozen ablation
conditions; without a cache, "we skipped HyDE for speed" is how an evaluation
ends up not testing the mechanism the paper is about (audit F-03). Cache hits
make the expensive conditions free to repeat, and make runs reproducible.
"""

import asyncio
import hashlib
import json
from pathlib import Path

import structlog
from anthropic import AsyncAnthropic

from app.config import ANTHROPIC_API_KEY, DATA_DIR, LLM_MODEL

logger = structlog.get_logger(__name__)

_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

CACHE_DIR = DATA_DIR / "cache" / "hyde"

#: Bump when a prompt changes — cached generations from an older prompt are then
#: ignored rather than silently reused, which would corrupt an ablation.
PROMPT_VERSION = "v2"

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

#: Deliberately domain-neutral. Condition C6 in the ablation grid uses this to
#: isolate how much of HyDE's benefit comes from style calibration rather than
#: from generating a pseudo-document at all.
HYDE_SYSTEM_GENERIC = """\
Write a short hypothetical passage (4 sentences) that would answer the user's
question, as it might appear in a reference document. Do not cite sources.\
"""

EXPANSION_SYSTEM = """\
Generate 3 alternative phrasings of the user's query that capture the same meaning
but use different vocabulary. Each rephrasing should be a single sentence.

Output exactly 3 lines, one rephrasing per line. No numbering, no explanation.\
"""


def _cache_key(query: str, system_prompt: str) -> str:
    payload = f"{LLM_MODEL}|{PROMPT_VERSION}|{system_prompt}|{query.strip().lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_read(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_write(key: str, payload: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{key}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        logger.warning("hyde_cache_write_failed", error=str(e))


async def _hyde_call(query: str, system_prompt: str, use_cache: bool) -> str:
    key = _cache_key(query, system_prompt)
    if use_cache:
        cached = _cache_read(key)
        if cached and cached.get("hyde"):
            return cached["hyde"]

    response = await _client.messages.create(
        model=LLM_MODEL,
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
    )
    text = response.content[0].text.strip()
    if use_cache:
        _cache_write(key, {"query": query, "hyde": text})
    return text


async def _expansion_call(query: str, use_cache: bool = True) -> list[str]:
    key = _cache_key(query, EXPANSION_SYSTEM)
    if use_cache:
        cached = _cache_read(key)
        if cached and cached.get("expansions"):
            return cached["expansions"]

    response = await _client.messages.create(
        model=LLM_MODEL,
        max_tokens=200,
        system=EXPANSION_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )
    lines = [
        line.strip()
        for line in response.content[0].text.strip().splitlines()
        if line.strip()
    ][:3]
    if use_cache and lines:
        _cache_write(key, {"query": query, "expansions": lines})
    return lines


async def transform(
    query: str,
    *,
    use_hyde: bool = True,
    use_expansion: bool = True,
    calibrated: bool = True,
    use_cache: bool = True,
) -> tuple[str, list[str], list[str]]:
    """
    Returns (hyde_text, all_queries, degraded_stages).

    `calibrated=False` swaps in the domain-neutral HyDE prompt (ablation C6).
    """
    degraded: list[str] = []
    system_prompt = HYDE_SYSTEM if calibrated else HYDE_SYSTEM_GENERIC

    tasks = []
    tasks.append(
        _hyde_call(query, system_prompt, use_cache)
        if use_hyde
        else asyncio.sleep(0, result=query)
    )
    tasks.append(
        _expansion_call(query, use_cache) if use_expansion
        else asyncio.sleep(0, result=[])
    )

    hyde_result, expansion_result = await asyncio.gather(
        *tasks, return_exceptions=True
    )

    if isinstance(hyde_result, BaseException):
        logger.warning("hyde_failed", error=str(hyde_result))
        degraded.append("hyde")
        hyde_text = query
    else:
        hyde_text = hyde_result or query

    if isinstance(expansion_result, BaseException):
        logger.warning("expansion_failed", error=str(expansion_result))
        degraded.append("expansion")
        expansions: list[str] = []
    else:
        expansions = expansion_result

    return hyde_text, [query] + expansions, degraded
