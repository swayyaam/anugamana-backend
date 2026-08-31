# Dataset card — Anugamana retrieval benchmark

**Version:** 1.0 · **Created:** 2026-08-31 · **Standard:** silver (model-judged)

A benchmark for retrieval over the Bhagavad-gita As It Is, built to measure
retrieval across a *register gap*: queries in ordinary modern language against
passages of Vedic commentary that share almost none of their vocabulary.

---

## Files

| file | contents |
|---|---|
| `queries.json` | 389 English queries with generation provenance |
| `queries_hi.json` | 133 Hindi translations, same `query_id`s |
| `qrels.json` | `{query_id: {verse_id: grade}}`, graded 0–3 |
| `qrels_meditations.json` | the same, for the *Meditations* corpus |
| `agreement.json` | inter-annotator agreement statistics |
| `judgments/*.json` | each annotator's raw grades, before consensus |

`verse_id` is a `chapter.verse` reference. **No Bhagavad-gita text is included**
— see Licensing.

---

## How the queries were made

Verse-blind generation. No verse, translation, purport or enrichment field ever
entered a generation prompt. Queries were produced from the information-need side
— a grid of 13 affective states × 12 life domains × 5 registers — the way an IR
benchmark elicits topics from users rather than from the collection.

This makes contamination structurally impossible rather than merely filtered. It
matters because the previous benchmark for this project was mined *from* the
indexed text and had to be retracted: 80/80 of its queries were verbatim
substrings of what the retriever searched.

**Registers** (78 queries each, deliberately varied because a benchmark of
uniformly well-formed sentences overstates real-world performance):
first-person distress · terse · rambling · third-person · abstract

**Verification:** `scripts/check_contamination.py` — 389/389 clean, median
longest shared n-gram with any indexed text **0**, maximum 5. The retracted set
scored 80/80 verbatim with a median of 12.

**Diversity:** 244 distinct three-word openings, 1,279-token vocabulary, 5–58
words (median 19).

---

## How relevance was judged

TREC-style pooling. No gold verse was nominated in advance. For each query the
union of the top-10 from all 14 conditions was judged — median pool 66 verses,
range 46–87.

| grade | meaning |
|---|---|
| 3 | directly addresses the query; the verse a knowledgeable person would cite |
| 2 | clearly relevant; addresses a substantial part of the situation |
| 1 | tangentially related; shares a theme but not the situation |
| 0 | not relevant |

Metrics treat **grade ≥ 2** as relevant. Gain is exponential (2^g − 1).

**Annotators:** three language models — `claude-haiku-4-5`,
`claude-sonnet-4-5`, `claude-sonnet-5`. Each graded a query's entire pool in one
call, with the pool shuffled per annotator so position could not drive
agreement. Consensus is the **median**, which is robust to one outlier and keeps
grades on the ordinal scale.

**Agreement:**

| corpus | Krippendorff α (ordinal) | pairwise weighted κ | reading |
|---|---|---|---|
| Bhagavad-gita | **0.709** | 0.698 – 0.736 | tentative conclusions only |
| Meditations | **0.655** | 0.673 – 0.692 | below the 0.667 floor — too noisy |

Grade distribution (Gita): 0 → 42,457 · 1 → 14,199 · 2 → 7,687 · 3 → 2,167.
381 of 389 queries have at least one relevant verse; mean 10.77 relevant per
query. 1,297 pairs had annotators two or more grades apart.

---

## Limitations — read before using this

1. **These are model judgments, not human ones.** α measures *consistency*, not
   correctness. All three annotators are Claude models: they share training data
   with each other and with the systems under evaluation, so their errors are
   correlated and α overstates true reliability.
2. **There is a specific, measured reason to worry.** A retrieval-free
   baseline that answers 54% of all queries with verse 2.47 — and uses just 41
   distinct verses across 388 situations — scores *highest* on this benchmark.
   That is consistent with the annotators sharing a famous-verse prior with the
   model being judged. Do not use these judgments to compare an LLM-based system
   against a non-LLM one without reading `docs/JUDGE_VALIDATION.md` first.
3. **Queries are model-generated.** Verse-blind and contamination-free, but not
   real user traffic.
4. **Pool depth 10.** Relevant verses outside every condition's top-10 are
   unjudged and count as irrelevant — the standard TREC caveat, which mildly
   favours pooled systems.
5. **The Hindi set is machine-translated** from the English, so a
   translate-then-retrieve condition translating back with the same model enjoys
   a round-trip advantage a native query would not confer.
6. **English-only judgments.** The Hindi queries reuse the English grades, which
   is valid because the information need is unchanged, but no native speaker has
   checked the translations.

**Before any published claim:** human validation on a stratified subset, with
human–human agreement and human–model correlation reported. Protocol and
acceptance criteria in `docs/JUDGE_VALIDATION.md`; collect with
`python scripts/annotate.py --annotator <name>`.

---

## Licensing

The Bhagavad-gita As It Is translations and purports are © Bhaktivedanta Book
Trust. The enriched corpus is a derivative work and is **not** redistributed.

This benchmark ships only `(query, verse_id, grade)` triples. Verse references
are not copyrightable, and the queries are original text generated for this
project, so the benchmark is releasable while the corpus is not. Anyone with
their own licensed copy of the text can reproduce every number.

The *Meditations* half of the artifact (Casaubon translation, Project Gutenberg)
is public domain end to end — corpus, enrichment, index and judgments — and is
fully releasable.

---

## Reproduce

```bash
python scripts/build_benchmark.py          # regenerate queries (verse-blind)
python scripts/check_contamination.py      # gate — must pass before use
python -m eval.run                         # all conditions
python scripts/pool_and_judge.py           # pooled graded judgments
python scripts/analyze.py                  # CIs, Holm correction, strata
```

Generation is sampled at temperature 1.0, so regenerated queries will differ.
The committed `queries.json` is the canonical set for the numbers in
[RESULTS.md](../../RESULTS.md).
