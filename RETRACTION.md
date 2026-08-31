# Retraction — evaluation results of 2026-05-10

**Status: all retrieval numbers produced before 2026-08-31 are invalid and must not be cited.**

The retracted run is preserved at
[`data/archive/eval_results_20260510.RETRACTED.json`](data/archive/eval_results_20260510.RETRACTED.json)
for provenance only.

---

## What was wrong

### 1. Test-set contamination (fatal)

`scripts/build_dataset.py` mined evaluation queries by splitting the
`meaning_fields.situations` string on semicolons. That same string is concatenated into
`text_for_embedding`, which `scripts/indexer.py` embeds as each verse's `_meaning` vector.

The retriever was therefore asked to find documents that literally contain the query.

```
80/80 eval queries appear VERBATIM in the indexed text_for_embedding
sources: {'mined': 80, 'manual': 0}
data/golden_manual.json — never created
```

This is the reason every condition reported Recall@5 = 1.000 on a task that is supposed to
be hard, and it is the reason the *sparse* condition scored highest: verbatim overlap is
exactly the regime where lexical matching wins.

### 2. No unenriched control condition existed

All three conditions retrieved over `_meaning` vectors, and `_meaning` vectors *are* the
enrichment (`evaluate.py:104` — `if "_meaning" not in doc_id: continue`).

The condition named `baseline` was "sparse retrieval over enriched text", not a baseline.
**The effect of enrichment — the project's central claim — was never isolated or measured.**

### 3. The condition named `full` was not the full pipeline

From the source comment at `evaluate.py:148`: *"For evaluation speed, we skip live HyDE
Claude calls."* It also never invoked query expansion, the purport collection, the
cross-encoder reranker, or the MMR pass — all of which the served API uses.

| Stage | Served | Evaluated |
|---|---|---|
| HyDE | yes | **no** |
| Query expansion | yes | **no** |
| Purport collection | yes | **no** |
| Cross-encoder rerank | yes | **no** |
| MMR diversity | yes | **no** |

HyDE had never been measured a single time.

---

## The numbers that were retracted

Single run, n = 10 queries, no confidence intervals, no significance test, no seed recorded.

| Condition | MRR@5 | Recall@5 | NDCG@5 |
|---|---|---|---|
| baseline | 0.9333 | 1.0000 | 0.9500 |
| no_hyde | 0.9000 | 1.0000 | 0.9262 |
| full | 0.8000 | 1.0000 | 0.8524 |

Note that these results *contradict* the project's stated hypothesis — the full pipeline
scored worst. That inversion is an artifact of contamination, not a finding.

---

## What replaces it

Nothing, yet. A number may only be published again once all of the following hold:

1. Every query passes `scripts/check_contamination.py` with zero verbatim overlap against
   any indexed text.
2. Relevance judgments are graded 0–3, pooled across all conditions, produced by at least
   two human annotators, with Krippendorff's α or Cohen's κ reported.
3. The evaluated pipeline is the served pipeline — both call the same parameterised
   function.
4. The grid includes a genuinely unenriched condition and a retrieval-free
   LLM-from-memory baseline.
5. Every reported difference carries a 95% CI from a paired bootstrap.

Until then this repository has **no** retrieval results.

---

*Retracted 2026-08-31 following a full audit of the evaluation code.*
