"""
Query routing — decide the cheapest correct path before the pipeline runs.

  direct_lookup — an explicit verse reference: skip HyDE, retrieval, reranking
  sanskrit      — Devanagari present: skip the HyDE call, embed the query as-is
  semantic      — everything else: the full pipeline

Fixed 2026-08-31 (audit E-01)
-----------------------------
The previous pattern was:

    \\b(?:bg|gita|verse|ch)?\\s*(\\d{1,2})[.\\s](\\d{1,3})\\b

The prefix group was optional *and* the separator class contained whitespace, so
any bare "N M" digit pair anywhere in a sentence hijacked the query:

    "I only sleep 4 5 hours a night"     -> direct_lookup 4.5
    "I've been meditating 2 3 months"    -> direct_lookup 2.3
    "I work 9 5 and hate it"             -> direct_lookup 9.5

Those users silently received one arbitrary verse with the entire semantic
pipeline skipped. The rules now are:

  * a whitespace-separated digit pair is NEVER a verse reference;
  * a bare dotted reference ("2.47") routes only when the query is essentially
    just that reference, or contains a scripture cue word — so "I lost 2.5 kg"
    stays semantic;
  * chapter/verse numbers are range-checked against the actual text before the
    fast path is taken.
"""

import re

#: Verses per chapter in Bhagavad-gita As It Is. Used to reject impossible
#: references before they reach the index.
CHAPTER_LENGTHS = {
    1: 46, 2: 72, 3: 43, 4: 42, 5: 29, 6: 47, 7: 30, 8: 28, 9: 34,
    10: 42, 11: 55, 12: 20, 13: 35, 14: 27, 15: 20, 16: 24, 17: 28, 18: 78,
}

_CUE_WORDS = (
    "verse", "chapter", "sloka", "shloka", "bg", "gita", "gītā", "adhyaya",
    "text", "says", "say", "mean", "means", "meaning", "explain", "translate",
    "recite", "quote",
)

# "chapter 2 verse 47", "chapter 2, verse 47", "ch 2 v 47"
_REF_LONG = re.compile(
    r"\b(?:chapter|ch|adhyaya)\.?\s*(\d{1,2})\s*[,;:]?\s*(?:verse|text|sloka|shloka|v)\.?\s*(\d{1,3})\b",
    re.IGNORECASE,
)

# "bg 2.47", "gita 2:47", "verse 2.47"
_REF_PREFIXED = re.compile(
    r"\b(?:bg|bhagavad\s*gita|gita|gītā|verse|text|sloka|shloka|chapter|ch)\.?\s*(\d{1,2})\s*[.:]\s*(\d{1,3})\b",
    re.IGNORECASE,
)

# "2.47" with a literal dot or colon. Gated by _is_reference_shaped.
_REF_BARE = re.compile(r"(?<!\d)(\d{1,2})\s*[.:]\s*(\d{1,3})(?!\d)")

# The whole query is just a reference, give or take punctuation.
_REF_ONLY = re.compile(r"^\W*(\d{1,2})\s*[.:]\s*(\d{1,3})\W*$")

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def is_valid_reference(chapter: int, verse: int) -> bool:
    return verse >= 1 and chapter in CHAPTER_LENGTHS and verse <= CHAPTER_LENGTHS[chapter]


def _is_reference_shaped(query: str) -> bool:
    """A bare dotted number is only a citation in a citation-like query."""
    if _REF_ONLY.match(query.strip()):
        return True
    lowered = query.lower()
    return any(re.search(rf"\b{re.escape(cue)}\b", lowered) for cue in _CUE_WORDS)


def has_devanagari(query: str) -> bool:
    return bool(_DEVANAGARI.search(query))


def classify_query(query: str) -> tuple[str, dict]:
    """
    Returns (route, meta).
      direct_lookup -> {"verse_id": "2.47"}
      sanskrit      -> {}
      semantic      -> {}
    """
    for pattern in (_REF_LONG, _REF_PREFIXED):
        match = pattern.search(query)
        if match:
            chapter, verse = int(match.group(1)), int(match.group(2))
            if is_valid_reference(chapter, verse):
                return "direct_lookup", {"verse_id": f"{chapter}.{verse}"}

    if _is_reference_shaped(query):
        match = _REF_BARE.search(query)
        if match:
            chapter, verse = int(match.group(1)), int(match.group(2))
            if is_valid_reference(chapter, verse):
                return "direct_lookup", {"verse_id": f"{chapter}.{verse}"}

    if has_devanagari(query):
        return "sanskrit", {}

    return "semantic", {}
