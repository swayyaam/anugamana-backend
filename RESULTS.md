# Results

**Run date:** 2026-08-31 · **Benchmark:** 389 verse-blind queries, 381 scorable
· **Judgments:** silver (3 model annotators, Krippendorff α = 0.709)
· **Metric:** graded nDCG@10 unless stated

Every number here traces to a committed script and a committed data file:
`data/eval/runs/*.json` (rankings), `data/benchmark/qrels.json` (judgments),
`data/eval/results.json` (this analysis). Reproduce with:

```bash
python -m eval.run && python scripts/pool_and_judge.py && python scripts/analyze.py
```

> **These results supersede everything before 2026-08-31.** The previous
> evaluation was retracted for test-set contamination; see
> [RETRACTION.md](RETRACTION.md). No number from that run may be cited.

---

## 1. Headline

**Synthetic semantic enrichment works, and it works specifically where it was
predicted to.** Against an identically-built unenriched index it raises nDCG@10
by 67% relative; against real BM25, by 124%. Both differences are large,
significant after family correction, and consistent across queries.

**But the most important result is a negative one about method, not about the
system:** an LLM answering from parametric memory with no retrieval at all scores
higher than every retrieval condition — and analysis shows that result is very
likely an artifact of the LLM judges sharing a prior with the LLM retriever. That
finding is discussed in §5 and is, in our view, the most publishable thing in
this repository.

---

## 2. All conditions

n = 381 queries with at least one relevant verse.

| key | nDCG@10 | MRR@10 | R@5 | R@10 | P@1 | what it isolates |
|---|---|---|---|---|---|---|
| **P0** | **0.5619** | 0.7526 | 0.2497 | 0.4051 | 0.6509 | no retrieval — LLM parametric memory |
| C6 | 0.3527 | 0.5611 | 0.1539 | 0.2791 | 0.4094 | HyDE, generic prompt |
| C8 | 0.3444 | 0.5473 | 0.1531 | 0.2588 | 0.4094 | + expansion + sparse (hybrid RRF) |
| C7 | 0.3430 | 0.5250 | 0.1551 | 0.2777 | 0.3491 | HyDE, domain-calibrated |
| C12 | 0.3205 | 0.4171 | 0.1317 | 0.2735 | 0.2441 | + emotion arm |
| **C5** | **0.3088** | 0.4969 | 0.1348 | 0.2307 | 0.3438 | **enrichment alone** |
| C9 | 0.3065 | 0.4019 | 0.1220 | 0.2689 | 0.2178 | + cross-encoder rerank |
| C10 | 0.3002 | 0.3868 | 0.1204 | 0.2588 | 0.2257 | **the served system** |
| C5b | 0.2984 | 0.4984 | 0.1333 | 0.2151 | 0.3438 | enrichment + translation + purport |
| C3 | 0.2424 | 0.4492 | 0.1063 | 0.1572 | 0.3281 | raw dense + purport chunks |
| **C2** | **0.1845** | 0.3831 | 0.0849 | 0.1104 | 0.3255 | **unenriched control** |
| C3b | 0.1506 | 0.2556 | 0.0576 | 0.0920 | 0.1417 | raw hybrid |
| **C0** | **0.1381** | 0.2490 | 0.0487 | 0.0845 | 0.1417 | **BM25 over raw translations** |
| C1 | 0.1287 | 0.2286 | 0.0443 | 0.0842 | 0.1102 | BM25 + purports |

---

## 3. Planned contrasts

Each question was written in `eval/conditions.py` **before any number existed**,
and every one is reported, favourable or not. 95% CIs from a paired bootstrap
(10,000 resamples); p-values from a two-sided randomisation test, Holm-corrected
across the family.

| question | contrast | Δ nDCG@10 | 95% CI | p (Holm) | W/L/T | verdict |
|---|---|---|---|---|---|---|
| Does synthetic semantic enrichment improve retrieval? | C2→C5 | **+0.1243** | [+0.1006, +0.1479] | 0.0008 | 284/92/5 | **YES** |
| Does enrichment beat a real lexical baseline? | C0→C5 | **+0.1707** | [+0.1506, +0.1905] | 0.0008 | 307/74/0 | **YES** |
| Does retrieval beat the model's parametric memory? | P0→C10 | **−0.2617** | [−0.2885, −0.2339] | 0.0008 | 67/314/0 | **NO — worse** |
| Does domain-calibrating the HyDE prompt matter? | C6→C7 | −0.0097 | [−0.0271, +0.0078] | 0.5483 | 186/194/1 | not shown |
| Does HyDE help on top of enrichment? | C5b→C7 | **+0.0446** | [+0.0279, +0.0619] | 0.0008 | 231/149/1 | **YES** |
| Does the out-of-domain cross-encoder help or hurt? | C8→C9 | **−0.0379** | [−0.0512, −0.0246] | 0.0008 | 138/238/5 | **HURTS** |
| What does MMR cost in relevance? | C9→C10 | −0.0063 | [−0.0173, +0.0048] | 0.5483 | 171/187/23 | not shown |
| Does an explicit emotion arm beat semantic similarity alone? | C10→C12 | **+0.0203** | [+0.0083, +0.0323] | 0.0024 | 214/149/18 | **YES** |

---

## 4. The central finding: enrichment immunises against the vocabulary gap

Queries are binned by IDF-weighted lexical overlap with their best relevant
verse. "none" means the query and the passage that answers it share no content
vocabulary at all — the regime this project exists to serve.

| overlap stratum | n | C0 (BM25) | C5 (enriched) | advantage |
|---|---|---|---|---|
| **none** (< 0.05) | 108 | 0.0500 | 0.2779 | **5.56×** |
| low (0.05–0.20) | 238 | 0.1623 | 0.3217 | 1.98× |
| medium (0.20–0.40) | 31 | 0.2471 | 0.3181 | 1.29× |
| high (> 0.40) | 4 | 0.2342 | 0.3039 | 1.30× |

Read the columns, not the rows. **BM25 collapses by a factor of five as overlap
vanishes** (0.2471 → 0.0500). **Enrichment barely moves** (0.3181 → 0.2779).

This is a stronger and more useful claim than "enrichment helps on average". The
mechanism is not a uniform quality boost — it is that document expansion
*removes the dependence of retrieval quality on shared vocabulary*. The
advantage grows monotonically as the gap widens, from 1.30× to 5.56×.

The `high` stratum has n = 4 and should not be interpreted.

---

## 5. The parametric baseline, and why we do not believe it

P0 — Claude Haiku asked directly to name the ten most relevant verses, with no
retrieval whatsoever — beats the full pipeline by a wide margin (0.5619 vs
0.3002). Taken at face value this says the retrieval system is unnecessary.

We do not think it should be taken at face value, and the diagnostic is in the
data:

| condition | top-5 verses' share of all top-1 results | distinct verses used at rank 1 | most returned |
|---|---|---|---|
| **P0** | **72.8%** | **41** | **2.47 for 210 of 388 queries** |
| C10 | 28.5% | 92 | 1.33 (24), 11.41 (24) |
| C5 | 21.1% | 168 | 1.32 (41) |
| C0 (BM25) | 15.4% | 169 | 18.37 (17) |

**P0 answers 54% of all queries — about grief, parent-care costs, romantic
betrayal, money, burnout — with the same verse (2.47).** It uses 41 distinct
verses to cover 388 distinct situations. A system that returns the same famous
verse for everything is not doing retrieval; it is reciting.

And it scores highest, because the judges are also Claude models. When the
system under test and the annotators share a prior about "the verse that answers
this", the benchmark measures agreement rather than relevance. The result is
therefore best read as a **quantified demonstration of judge–system correlation
bias in LLM-as-judge evaluation**, with top-1 concentration as a cheap
diagnostic that exposes it.

This is falsifiable and the test is specified in
[docs/JUDGE_VALIDATION.md](docs/JUDGE_VALIDATION.md): if human annotators grade
2.47 far lower than the models do on non-duty queries, the bias is confirmed and
P0's ranking collapses. **Until that human pass is run, no claim about P0 —
in either direction — should be published.**

---

## 6. Negative and inconvenient results

**Domain-calibrating the HyDE prompt does nothing measurable.** A generic
"write a passage that would answer this" prompt (C6, 0.3527) performs
indistinguishably from a carefully engineered Prabhupada-style one (C7, 0.3430);
the point estimate slightly favours the generic prompt. The elaborate
corpus-derived vocabulary in `HYDE_SYSTEM` is not earning its complexity. This
contradicts one of the project's own planned paper claims.

**The cross-encoder actively hurts.** Reranking with `ms-marco-MiniLM-L-6-v2`
costs 0.0379 nDCG@10 (p < 0.001, losing on 238 queries and winning on 138). The
model is out of domain: measured directly, its logits over the top-10 candidates
run from −2.5 to −11.3, i.e. calibrated probabilities of 0.07 down to ~0.0000 —
it considers every Gita verse irrelevant. It should be removed from the served
pipeline or replaced with a domain-tuned reranker.

**MMR is not paying for itself either**, though the effect is not significant
(−0.0063, p = 0.55). It remains defensible on user-experience grounds.

**The served system is not the best system.** C10 (0.3002) ranks below C6, C7,
C8 and C12. The ranking stages that were added for product polish cost retrieval
quality. C8 + the emotion arm, without cross-encoder reranking, is the
configuration the evidence currently supports.

---

## 7. Failure analysis (C10)

| failure mode | count | share | implication |
|---|---|---|---|
| acceptable (nDCG@10 ≥ 0.5) | 55 | 14.4% | — |
| **ranked away** (relevant verse retrieved, poorly ranked) | 265 | 69.6% | ranking problem |
| never retrieved (absent from top-10) | 61 | 16.0% | recall problem |

Seven in ten failures are ranking failures, not recall failures — the relevant
verse *was* found and then buried. That is consistent with the cross-encoder
result above and says clearly where the next effort belongs.

By query register:

| register | n | nDCG@10 |
|---|---|---|
| third person ("my sister keeps saying…") | 71 | 0.1971 |
| terse ("feel guilty about mom") | 76 | 0.2676 |
| abstract | 78 | 0.3304 |
| first-person distress | 78 | 0.3408 |
| rambling | 78 | 0.3550 |

Third-person queries are the weakest by a wide margin. The enrichment's
`situations` field is written in second person about the reader's own life, so a
query about someone else's situation lands further away. That is a concrete,
addressable enrichment-design finding.

---

## 8. Threats to validity

Stated plainly, because they bound every number above.

1. **Judgments are silver, not gold.** Three Claude models, α = 0.709 —
   "tentative conclusions only" on Krippendorff's scale, and the annotators
   share training data with each other and with the systems under test. Human
   validation is required; the protocol and acceptance criteria are in
   `docs/JUDGE_VALIDATION.md`.
2. **Queries are model-generated.** They are verse-blind by construction and
   pass the contamination gate (389/389 clean, median longest shared n-gram 0,
   versus 80/80 verbatim for the retracted set), but they are not real user
   traffic. `data/feedback.db` will supply that over time.
3. **One corpus.** Generalisation to *Meditations* is built
   (`scripts/build_second_corpus.py`, 410 passages segmented) but not yet run
   through the grid.
4. **The confidence threshold is unfitted.** `MIN_RELEVANCE = 0.0` deliberately,
   because thresholding an uncalibrated score is what caused audit defect E-02.
5. **Pool depth 10 over 14 conditions** leaves relevant verses outside the pool
   unjudged, which is the standard TREC caveat and slightly favours pooled
   systems.

---

## 9. What this changes about the plan

- The paper's contribution is **not** "we built a Gita search engine", and not
  "synthetic semantic enrichment is a new method" — that is LLM document
  expansion, and the doc2query lineage must be cited (see the audit).
- The defensible contributions are: (a) the **overlap-stratified result** in §4,
  which turns "helps" into a measured curve; (b) the **judge-correlation
  finding** in §5, which is a methodological caution for the whole LLM-as-judge
  literature; (c) a **releasable benchmark** and harness.
- Two planned claims are dead: HyDE domain calibration (§6) and the served
  pipeline being the best pipeline (§6).

---

*Regenerate this document's numbers with `python scripts/analyze.py` and
`python scripts/error_analysis.py`. Do not edit the figures by hand.*
