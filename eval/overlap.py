"""
Lexical overlap: the contamination gate, and the independent variable.

Two jobs, one measurement.

**Contamination gate.** The retracted benchmark was mined by splitting the
enrichment's `situations` field, and that field is embedded verbatim as each
verse's `_meaning` vector — so 80/80 queries were substrings of the text being
searched. Any future benchmark must pass `assert_clean()` before it is used.

**The independent variable.** "Vocabulary gap" is currently rhetoric in this
project's write-ups. Making it a number turns the claim from *"enrichment helps"*
into *"enrichment gain grows as query-document lexical overlap approaches zero,
and crosses over sparse retrieval below x"* — a curve, falsifiable, and the
figure worth citing. `stratify()` bins the benchmark so per-bin scores can be
reported.

Overlap is IDF-weighted deliberately: sharing "the" is not evidence of lexical
proximity, sharing "renunciation" is. IDF is computed over the corpus being
searched, so the weighting reflects what is actually rare *in this collection*.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z']+")

_STOPWORDS = frozenset("""
a an and are as at be been but by for from had has have he her his i if in is it its
me my not of on or our she that the their them there they this to too was we were
what when which who will with you your
""".split())


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 2
    ]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------

@dataclass
class ContaminationReport:
    query: str
    verse_id: str
    verbatim: bool
    max_ngram_overlap: float
    longest_shared_ngram: int

    @property
    def contaminated(self) -> bool:
        # Verbatim containment is disqualifying outright. Beyond that, sharing a
        # long exact phrase with the indexed text is the signature of a query
        # derived from that text rather than written independently.
        return self.verbatim or self.longest_shared_ngram >= 8


def _normalise(text: str) -> str:
    return " ".join(tokenize(text))


def check_query(
    query: str, indexed_texts: dict[str, str], verse_id: str
) -> ContaminationReport:
    """Compare one query against the text indexed for its gold verse."""
    indexed = indexed_texts.get(verse_id, "")
    normalised_query = _normalise(query)
    normalised_indexed = _normalise(indexed)

    verbatim = bool(normalised_query) and normalised_query in normalised_indexed

    query_tokens = tokenize(query)
    indexed_tokens = tokenize(indexed)

    longest = 0
    for n in range(min(len(query_tokens), 15), 2, -1):
        if ngrams(query_tokens, n) & ngrams(indexed_tokens, n):
            longest = n
            break

    overlap = 0.0
    query_5 = ngrams(query_tokens, 5)
    if query_5:
        overlap = len(query_5 & ngrams(indexed_tokens, 5)) / len(query_5)

    return ContaminationReport(
        query=query,
        verse_id=verse_id,
        verbatim=verbatim,
        max_ngram_overlap=round(overlap, 4),
        longest_shared_ngram=longest,
    )


def assert_clean(reports: list[ContaminationReport]) -> None:
    """Raise if any query is derived from the indexed text."""
    bad = [r for r in reports if r.contaminated]
    if bad:
        lines = "\n".join(
            f"  [{r.verse_id}] verbatim={r.verbatim} "
            f"longest_shared_ngram={r.longest_shared_ngram}: {r.query[:80]}"
            for r in bad[:20]
        )
        raise AssertionError(
            f"{len(bad)}/{len(reports)} queries are contaminated by the indexed "
            f"text. This is the defect that invalidated the 2026-05-10 "
            f"evaluation — see RETRACTION.md.\n{lines}"
        )


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def build_idf(documents: list[str]) -> dict[str, float]:
    """Smoothed IDF over the collection actually being searched."""
    total = len(documents) or 1
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(tokenize(document)))
    return {
        token: math.log((total + 1) / (count + 1)) + 1.0
        for token, count in document_frequency.items()
    }


def weighted_overlap(query: str, document: str, idf: dict[str, float]) -> float:
    """
    IDF-weighted Jaccard-style overlap in [0, 1].

    0.0 means the query and the passage that answers it share no content
    vocabulary at all — the regime this entire project exists to serve.
    """
    query_tokens = set(tokenize(query))
    document_tokens = set(tokenize(document))
    if not query_tokens:
        return 0.0

    default = max(idf.values(), default=1.0)
    shared = sum(idf.get(t, default) for t in query_tokens & document_tokens)
    total = sum(idf.get(t, default) for t in query_tokens)
    return round(shared / total, 4) if total else 0.0


#: Bin edges over IDF-weighted overlap. The lowest bin is the paper's claim.
STRATA = (
    ("none", 0.00, 0.05),
    ("low", 0.05, 0.20),
    ("medium", 0.20, 0.40),
    ("high", 0.40, 1.01),
)


def stratum_of(overlap: float) -> str:
    for name, low, high in STRATA:
        if low <= overlap < high:
            return name
    return STRATA[-1][0]


def stratify(
    queries: list[dict], texts: dict[str, str], idf: dict[str, float]
) -> dict[str, list[dict]]:
    """
    Group benchmark queries by their overlap with the gold verse's text.
    Each query dict needs "query" and "verse_id".
    """
    buckets: dict[str, list[dict]] = {name: [] for name, _, _ in STRATA}
    for item in queries:
        overlap = weighted_overlap(
            item["query"], texts.get(item["verse_id"], ""), idf
        )
        enriched = {**item, "overlap": overlap, "stratum": stratum_of(overlap)}
        buckets[enriched["stratum"]].append(enriched)
    return buckets
