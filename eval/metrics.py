"""
Retrieval metrics over *graded* relevance judgments.

The retracted evaluation used a single gold verse per query and binary
relevance. For this corpus that is close to meaningless: a life situation is
legitimately addressed by several verses, often across chapters, so a system
that returns a genuinely apt verse at rank 1 scored zero if a different apt verse
had been nominated as "the" answer. NDCG in particular is uninterpretable with
one binary gold — its ideal ranking has a single non-zero gain.

Judgments here are graded 0-3, pooled across every condition:

    3  directly addresses the query; the verse a knowledgeable person would cite
    2  clearly relevant; addresses a substantial part of the query
    1  tangentially related; shares a theme but does not address the situation
    0  not relevant

Gains are exponential (2^g - 1), which is the standard formulation and keeps a
grade-3 verse worth substantially more than two grade-1 verses.
"""

from __future__ import annotations

import math

#: A verse counts as "relevant" for set-based metrics at this grade or above.
RELEVANT_THRESHOLD = 2


def gain(grade: int) -> float:
    return (2.0**grade) - 1.0


def reciprocal_rank(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """1/rank of the first relevant verse, else 0."""
    for index, verse_id in enumerate(ranked[:k]):
        if qrels.get(verse_id, 0) >= RELEVANT_THRESHOLD:
            return 1.0 / (index + 1)
    return 0.0


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    relevant = {v for v, g in qrels.items() if g >= RELEVANT_THRESHOLD}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    if not ranked[:k]:
        return 0.0
    hits = sum(1 for v in ranked[:k] if qrels.get(v, 0) >= RELEVANT_THRESHOLD)
    return hits / len(ranked[:k])


def average_precision(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    relevant = {v for v, g in qrels.items() if g >= RELEVANT_THRESHOLD}
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for index, verse_id in enumerate(ranked[:k]):
        if verse_id in relevant:
            hits += 1
            total += hits / (index + 1)
    return total / min(len(relevant), k)


def dcg(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    return sum(
        gain(qrels.get(verse_id, 0)) / math.log2(index + 2)
        for index, verse_id in enumerate(ranked[:k])
    )


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """
    Graded NDCG. The ideal ranking is every judged verse sorted by grade, which
    is why pooled judgments matter — an unpooled qrel makes the ideal DCG too
    small and inflates every system.
    """
    ideal_grades = sorted(qrels.values(), reverse=True)[:k]
    ideal = sum(
        gain(grade) / math.log2(index + 2)
        for index, grade in enumerate(ideal_grades)
    )
    if ideal == 0:
        return 0.0
    return dcg(ranked, qrels, k) / ideal


#: name -> (function, k). Every condition reports all of these.
METRICS = {
    "mrr@10": (reciprocal_rank, 10),
    "ndcg@10": (ndcg_at_k, 10),
    "ndcg@5": (ndcg_at_k, 5),
    "recall@5": (recall_at_k, 5),
    "recall@10": (recall_at_k, 10),
    "map@10": (average_precision, 10),
    "p@1": (precision_at_k, 1),
}


def score_query(ranked: list[str], qrels: dict[str, int]) -> dict[str, float]:
    """All metrics for one query. Per-query values are kept, not just means —
    the paired bootstrap in eval/stats.py needs them."""
    return {
        name: function(ranked, qrels, k) for name, (function, k) in METRICS.items()
    }


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {name: 0.0 for name in METRICS}
    return {
        name: sum(row[name] for row in per_query) / len(per_query)
        for name in METRICS
    }
