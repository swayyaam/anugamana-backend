# Anugamana — अनुगमन

> *To be guided along the path.*

Semantic search and RAG over the **Bhagavad-gita As It Is** (Srila Prabhupada).
Someone describes a life situation in plain modern language — in English or an
Indian language — and gets the verses that actually address it, with guidance
grounded in the purports.

**[RESULTS.md](RESULTS.md)** — measured retrieval quality, ablations, and the
findings that did not go our way.
**[RETRACTION.md](RETRACTION.md)** — why every number before 2026-08-31 was
invalid.

---

## The problem

A user types *"I'm paralyzed by fear of making the wrong decision."*
The verse that answers it says *"You have a right to perform your prescribed
duties, but you are not entitled to the fruits of action."*

They share no vocabulary. Keyword search cannot bridge that, and this is not a
rhetorical flourish — it is measurable. Binning our benchmark by IDF-weighted
query–document overlap:

| overlap between query and the verse that answers it | BM25 nDCG@10 | with enrichment | advantage |
|---|---|---|---|
| **none** (n=108) | 0.0500 | 0.2779 | **5.56×** |
| low (n=238) | 0.1623 | 0.3217 | 1.98× |
| medium (n=31) | 0.2471 | 0.3181 | 1.29× |

BM25 collapses fivefold as shared vocabulary disappears. Enrichment barely
moves. That gap is what this system is for.

## The approach

For every verse, an LLM generates four fields in modern everyday language —
*situations*, *teaching*, *emotions*, *concepts* — and **those** are what gets
embedded and searched. The vocabulary gap is closed offline, once, before any
query arrives.

This is LLM **document expansion**, in the doc2query / docTTTTTquery lineage.
What is new here is the setting (a cross-register, cross-temporal gap), the
measurement (§4 of RESULTS.md), and the released benchmark — not the mechanism.

Measured effect against an identically-built unenriched index:
**+0.1243 nDCG@10, 95% CI [+0.1006, +0.1479]**, winning on 284 of 381 queries.

---

## Pipeline

```
query
  │
  ├─ crisis routing ─────────── lexical prefilter + classifier
  │                             hard-coded response with real helplines
  │                             never reaches retrieval or generation
  │
  ├─ language ──────────────── script detection (offline) → Sarvam text-lid
  │                             non-English pivots to English via Mayura
  │
  ├─ topical guardrail ─────── on-topic classifier, fails open
  │
  ├─ routing ───────────────── direct_lookup ("BG 2.47")  → skip to fetch
  │                             sanskrit (Devanagari)      → skip HyDE
  │                             semantic                   → full pipeline
  │
  ├─ query transform ───────── HyDE + 3 expansions (disk-cached)
  │   ‖ concurrent             emotion classification → probe vector
  │
  ├─ retrieval ─────────────── dense: meaning + translation + purport chunks
  │                             sparse: BGE-M3 lexical weights
  │                             extra arms: emotion probe, transliteration
  │                             RRF fusion → group by verse → top 10
  │
  ├─ ranking ───────────────── RRF fusion order
  │                             (the cross-encoder was here until it measured
  │                              ROC AUC 0.4579 — worse than random — and was
  │                              removed; +0.0455 nDCG@10, RESULTS.md §6)
  │
  ├─ generation ────────────── per verse, concurrent, grounded in the ±1
  │                             paragraph window around the matched chunk
  │
  └─ response ──────────────── translated back to the user's language
                                async: LLM judge → SQLite
```

Every stage has a fallback and reports itself in `degraded_stages`. The pipeline
never returns a 500 for an internal failure.

This was verified against a real outage rather than a simulated one — the LLM API
hit a usage limit mid-development, and the system behaved as designed:

| query | with the LLM API completely down |
|---|---|
| "I want to kill myself" | still routed to crisis, helplines served — the prefilter is lexical and offline |
| "I keep failing at work" | verses still returned (retrieval is local), guidance absent, `degraded_stages` populated |
| any | HTTP 200, honest status, no 500 |

The crisis path surviving an API outage is not incidental. A safety branch that
depends on the network is a safety branch that fails exactly when a service is
already having a bad day.

**One implementation.** `app/services/pipeline.py` is called by both the API and
the evaluation harness. An ablation condition is a `PipelineConfig`, not a second
code path — which is how the evaluated system silently stopped matching the
served one last time.

---

## API

### `POST /search`

```json
{ "query": "I keep failing and feel like giving up", "top_k": 3 }
```

```json
{
  "results": [
    {
      "verse_id": "2.47",
      "chapter": 2, "verse": 47,
      "devanagari": "कर्मण्येवाधिकारस्ते...",
      "sanskrit": "karmaṇy evādhikāras te...",
      "translation": "You have a right to perform your prescribed duties...",
      "score": 0.5,
      "ai_guidance": "The anxiety you feel about failing comes from..."
    }
  ],
  "query_meta": {
    "status": "ok",
    "score_type": "rrf",
    "query_route": "semantic",
    "low_confidence": false,
    "degraded_stages": [],
    "total_ms": 4210
  }
}
```

`status` is `ok` / `off_topic` / `crisis` / `no_results` — all HTTP 200. Crisis
and off-topic are expected outcomes, not errors, and previously returned 422,
which was indistinguishable from a schema validation failure.

`score_type` says what `score` means, and the client must respect it. `rrf` — the
current default — is **ordinal**: it ranks, it does not measure, and it must
never be thresholded. `cross_encoder` would be an absolute relevance probability
comparable across queries, and is available if a domain-tuned reranker is fitted
(see `eval/calibrate.py`).

Either way the score is deliberately *not* a within-result-set normalisation.
That was the bug that silently deleted the last result of every search and
reported 1.0 for the best of three bad matches.

Other endpoints: `GET /health`, `GET /metrics`, `POST /feedback`.

---

## Evaluation

The harness is the point of this repository as much as the search engine is.

```bash
python -m eval.run                        # 14 conditions over 389 queries
python -m eval.run --include-multilingual --include-generalisation
python -m eval.calibrate                  # fit thresholds on graded data
python scripts/pool_and_judge.py          # pooled graded judgments
python scripts/analyze.py                 # CIs, Holm correction, strata
python scripts/error_analysis.py          # failure taxonomy, bias diagnostics
python scripts/check_contamination.py --self-test
```

Properties it enforces, each of which was violated by the retracted evaluation:

- **Verse-blind queries.** No verse, translation, purport or enrichment field
  ever enters a query-generation prompt. Queries come from the information-need
  side — affective state × life domain × register — as a real IR benchmark
  elicits topics. 389/389 pass the contamination gate; the retracted set was
  80/80 verbatim.
- **Pooled graded judgments.** No gold verse is nominated in advance; the union
  of every condition's top-10 is graded 0–3 by three annotators.
- **A real control.** `scripts/build_raw_index.py` builds an unenriched index
  identically — same model, same chunking — differing in exactly one factor.
- **Dangerous baselines first.** Real BM25, and an LLM answering from memory
  with no retrieval. Both are run before anything else, because they are the
  ones that can sink the project.
- **Statistics.** Paired bootstrap CIs, randomisation tests, Holm correction,
  per-query win/loss/tie counts.

Judgments are currently **silver** (model annotators, α = 0.709). Human
validation is required before publication — protocol in
[docs/JUDGE_VALIDATION.md](docs/JUDGE_VALIDATION.md), collect with
`python scripts/annotate.py --annotator <name>`.

---

## Indic support (Sarvam AI)

Optional. Without `SARVAM_API_KEY` every entry point degrades to a defined
fallback and English search is unaffected.

| capability | model | role |
|---|---|---|
| language ID | script detection → `text-lid` | offline first, API only for Latin script |
| translation | Mayura | query → English pivot; guidance → user's language |
| transliteration | — | romanised Sanskrit → Devanagari, as an extra lexical arm |
| emotion | `sarvam-105b-conversations` | affective state → retrieval probe |
| speech | Bulbul v3 | verse and guidance audio |

Guidance is generated in English and then translated, not generated in the
target language — generating directly would make the model translate
Prabhupada's commentary on the fly, the step most likely to introduce
unsupported claims into a religious text.

```bash
python scripts/verify_sarvam.py    # checks all five endpoints against the live API
```

Run this after setting the key and whenever Sarvam version anything. It found
four wrong assumptions the first time it ran.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env          # add ANTHROPIC_API_KEY
```

Build the indexes (one-time; needs ~4GB for BGE-M3 plus the vector stores):

```bash
python scripts/scraper.py         # 700 verses
python scripts/analyze_gita.py    # 18 chapter analyses
python scripts/enrich.py          # meaning_fields  (~$1.50, ~20 min)
python scripts/indexer.py         # enriched index  (~15 min)
python scripts/build_raw_index.py # unenriched control index
```

```bash
uvicorn app.main:app --reload     # http://localhost:8000/docs
pytest                            # 223 tests, ~3s, no network
```

### Demo

```bash
uvicorn app.main:app
```

Then open **http://localhost:8000/demo** — a self-contained page served by the
API itself. No build step, no npm, no auth, no CORS: one command and a browser.
It ships example queries covering the ordinary case, grief, a Hindi query, a
direct verse lookup and an off-topic rejection, shows the pipeline stages while
it works (a semantic query takes 5-15s), and reports the route, timings and any
degraded stages underneath the results.

---

## Status

| | |
|---|---|
| Pipeline | serving, with crisis routing and graceful degradation |
| Evaluation | 18 conditions, 389 queries, silver judgments (α = 0.709) |
| Human validation | **not started** — blocks publication |
| Second corpus | *Meditations*: 410 passages enriched, indexed, replication run |
| Cross-lingual | 133 Hindi queries; translate-then-retrieve wins, equity gap measured |
| Reranker | measured at ROC AUC 0.4579 — removed from the served pipeline |

## Research

The contribution is the problem class, the measurement, and the resource — not
the mechanism. Full argument, including the contributions that did **not**
survive contact with data, in [RESULTS.md](RESULTS.md).

## License

Pipeline code is open source. The Bhagavad-gita As It Is text is copyright
Bhaktivedanta Book Trust; the enriched corpus is a derivative work and is not
redistributed here. The benchmark ships as `(query, verse_id, grade)` triples
with no BBT text, and the *Meditations* half of the artifact is public domain and
fully releasable.

---

*Built by Swayam Mishra*
