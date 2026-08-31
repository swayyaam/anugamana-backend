"""
The two baselines that decide whether this project has a result.

**BM25 (C0, C1).** The retracted evaluation called its sparse condition
"baseline", but it ran over the *enriched* text, so it was not a baseline of
anything. This is Okapi BM25 over untouched translations — a 1994 algorithm with
no LLM anywhere near it. If the full pipeline cannot beat it convincingly, the
enrichment is not earning its cost and the paper has to say so.

**Parametric memory (P0).** The Bhagavad-gita is in every frontier model's
training data, with centuries of commentary about it. A reviewer will ask why
retrieval is needed when the model may simply know the answer, and the only
acceptable response is a measured one. The prompt is deliberately favourable to
the model: it is told the corpus and asked for ranked verse references.
"""

from __future__ import annotations

import asyncio
import json
import re

import structlog
from anthropic import AsyncAnthropic
from rank_bm25 import BM25Okapi

from app.config import ANTHROPIC_API_KEY, ENRICHED_FILE, LLM_MODEL
from eval.overlap import tokenize

logger = structlog.get_logger(__name__)

_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Retriever:
    """Okapi BM25 over raw corpus text. No enrichment, no embeddings."""

    def __init__(self, verses: list[dict], include_purport: bool = False):
        self.verse_ids: list[str] = []
        corpus: list[list[str]] = []
        for verse in verses:
            text = verse["translation"]
            if include_purport:
                text = f"{text}\n\n{verse.get('purport', '')}"
            self.verse_ids.append(verse["verse_id"])
            corpus.append(tokenize(text))
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [self.verse_ids[i] for i in ranked]


def load_corpus() -> list[dict]:
    """
    Raw corpus fields only. `text_for_embedding` and `meaning_fields` are
    deliberately not read here — a baseline that touches the enrichment is not
    a baseline.
    """
    verses = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
    return [
        {
            "verse_id": v["verse_id"],
            "chapter": v["chapter"],
            "verse": v["verse"],
            "translation": v["translation"],
            "purport": v.get("purport", ""),
        }
        for v in verses
    ]


# ---------------------------------------------------------------------------
# Parametric memory
# ---------------------------------------------------------------------------

PARAMETRIC_SYSTEM = """\
You know the Bhagavad-gita well, including Srila Prabhupada's "Bhagavad-gita As
It Is".

The user will describe a situation, a feeling, or a question. Name the verses
that most directly address it, best first.

Rules:
- Answer only with verse references in "chapter.verse" form.
- Give exactly 10, ranked, most relevant first.
- Chapter is 1-18. Use the numbering of Bhagavad-gita As It Is.
- No commentary, no explanation, no other text.

Respond with valid JSON only: {"verses": ["2.47", "3.19", ...]}\
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_REF_RE = re.compile(r"\b(\d{1,2})\.(\d{1,3})\b")


async def parametric_retrieve(query: str, top_k: int = 10) -> list[str]:
    """Ask the model directly. Returns [] on failure rather than a fake ranking."""
    try:
        response = await _client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            system=PARAMETRIC_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        text = response.content[0].text.strip()

        verses: list[str] = []
        match = _JSON_RE.search(text)
        if match:
            try:
                verses = [str(v) for v in json.loads(match.group(0)).get("verses", [])]
            except json.JSONDecodeError:
                verses = []
        if not verses:
            # The model ignored the format but the references are still usable.
            verses = [f"{c}.{v}" for c, v in _REF_RE.findall(text)]

        seen, ordered = set(), []
        for verse_id in verses:
            if verse_id not in seen:
                seen.add(verse_id)
                ordered.append(verse_id)
        return ordered[:top_k]
    except Exception as e:
        logger.warning("parametric_retrieve_failed", error=str(e))
        return []


async def parametric_retrieve_many(
    queries: list[str], top_k: int = 10, concurrency: int = 5
) -> list[list[str]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(query: str) -> list[str]:
        async with semaphore:
            return await parametric_retrieve(query, top_k)

    return await asyncio.gather(*(one(q) for q in queries))
