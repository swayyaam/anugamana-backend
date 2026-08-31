#!/usr/bin/env python3
"""
Build pooled, graded relevance judgments (qrels) for the benchmark.

Pooling
-------
Nobody nominates a gold verse in advance. For each query we take the union of the
top-K results returned by every condition, and judge that pool. This is the
standard TREC construction, and it matters for two reasons:

* A benchmark whose gold verses were chosen by one system encodes that system's
  idea of the answer and flatters it.
* Graded NDCG needs an ideal ranking. Judging only one verse per query makes the
  ideal DCG too small and inflates every system's score.

Judging
-------
Each annotator grades a query's entire pool in a single call, so the grades for
one query are made in the same context and are mutually comparable. Annotators
are different models, and the pool order is shuffled per annotator so that
position cannot drive agreement.

**These are model-generated judgments — a silver standard.** Krippendorff's alpha
across annotators measures their consistency, which is a necessary but not
sufficient condition: consistent models can be consistently wrong. A human-judged
subset produced with scripts/annotate.py is required before any of this supports
a published claim. See data/benchmark/DATASET_CARD.md.

Usage:
    python scripts/pool_and_judge.py
    python scripts/pool_and_judge.py --pool-depth 10 --limit 50
"""

import argparse
import asyncio
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import AsyncAnthropic  # noqa: E402

from app.config import ANTHROPIC_API_KEY, DATA_DIR, ENRICHED_FILE  # noqa: E402
from eval.agreement import analyse  # noqa: E402

BENCHMARK_DIR = DATA_DIR / "benchmark"
RUNS_DIR = DATA_DIR / "eval" / "runs"
QRELS_FILE = BENCHMARK_DIR / "qrels.json"
AGREEMENT_FILE = BENCHMARK_DIR / "agreement.json"

client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

#: Independent annotators. Different model families/tiers, because a model
#: agreeing with itself is not evidence of anything.
ANNOTATORS = [
    ("haiku", "claude-haiku-4-5-20251001"),
    ("sonnet", "claude-sonnet-4-5-20250929"),
]

JUDGE_SYSTEM = """\
You are grading how well individual Bhagavad-gita verses address a person's query,
for a retrieval benchmark.

Grade each verse 0-3:
  3  directly addresses the query — the verse a knowledgeable person would cite
  2  clearly relevant — addresses a substantial part of the situation
  1  tangentially related — shares a theme but does not address the situation
  0  not relevant

Judge the verse as shown. Do not reward a verse for being famous, or for being
generally wise. Ask only: would this verse help *this* person with *this*
situation?

Most verses in a candidate pool are not relevant. Grades of 3 should be rare.
Do not spread grades evenly.

Respond with valid JSON only, mapping every verse id you were given to its grade:
{"2.47": 3, "3.19": 1, "18.66": 0}\
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_verses() -> dict[str, dict]:
    verses = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
    return {v["verse_id"]: v for v in verses}


def load_runs() -> dict[str, dict[str, list[str]]]:
    runs = {}
    for path in sorted(RUNS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[payload["condition"]] = payload["results"]
    return runs


def build_pools(
    runs: dict[str, dict[str, list[str]]], depth: int
) -> dict[str, list[str]]:
    pools: dict[str, set[str]] = defaultdict(set)
    for results in runs.values():
        for query_id, ranked in results.items():
            pools[query_id].update(ranked[:depth])
    return {query_id: sorted(verses) for query_id, verses in pools.items()}


async def judge_pool(
    model: str,
    query: str,
    pool: list[str],
    verses: dict[str, dict],
    seed: int,
) -> dict[str, int]:
    """Grade one query's whole pool in a single call."""
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)

    listing = "\n\n".join(
        f"[{vid}] {verses[vid]['translation']}"
        for vid in shuffled
        if vid in verses
    )
    message = f"Query: {query}\n\nCandidate verses:\n\n{listing}"

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=1200,
            temperature=0.0,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": message}],
        )
        match = _JSON_RE.search(response.content[0].text.strip())
        if not match:
            return {}
        raw = json.loads(match.group(0))
        return {
            str(vid): max(0, min(3, int(grade)))
            for vid, grade in raw.items()
            if str(vid) in set(pool) and isinstance(grade, (int, float))
        }
    except Exception as e:
        print(f"    judge failed ({model}): {e}")
        return {}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-depth", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    queries = json.loads((BENCHMARK_DIR / "queries.json").read_text(encoding="utf-8"))
    if args.limit:
        queries = queries[: args.limit]
    query_text = {q["query_id"]: q["query"] for q in queries}

    runs = load_runs()
    if not runs:
        print(f"No cached runs in {RUNS_DIR}. Run: python -m eval.run")
        return 2

    verses = load_verses()
    pools = build_pools(runs, args.pool_depth)
    pools = {qid: pool for qid, pool in pools.items() if qid in query_text}

    sizes = sorted(len(p) for p in pools.values())
    print(f"Pooled {len(runs)} conditions over {len(pools)} queries")
    print(f"  pool size: min={sizes[0]} median={sizes[len(sizes) // 2]} max={sizes[-1]}")
    print(f"  total judgments: {sum(sizes) * len(ANNOTATORS)} "
          f"({len(ANNOTATORS)} annotators)")

    judgments: dict[str, dict[str, int]] = {name: {} for name, _ in ANNOTATORS}
    semaphore = asyncio.Semaphore(args.concurrency)
    done = 0
    total = len(pools) * len(ANNOTATORS)

    async def one(name: str, model: str, query_id: str, seed: int) -> None:
        nonlocal done
        async with semaphore:
            graded = await judge_pool(
                model, query_text[query_id], pools[query_id], verses, seed
            )
        for vid, grade in graded.items():
            judgments[name][f"{query_id}|{vid}"] = grade
        done += 1
        if done % 25 == 0 or done == total:
            print(f"\r  judged {done}/{total}", end="", flush=True)

    tasks = [
        one(name, model, query_id, index)
        for index, (name, model) in enumerate(ANNOTATORS)
        for query_id in pools
    ]
    await asyncio.gather(*tasks)
    print()

    report = analyse(judgments)
    print("\n" + report.format())

    # Consensus grade = mean of annotators, rounded. Ties round toward the
    # lower grade: a benchmark should not credit a system for a verse the
    # annotators could not agree was relevant.
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    disagreements = 0
    for query_id, pool in pools.items():
        for vid in pool:
            key = f"{query_id}|{vid}"
            grades = [
                judgments[name][key] for name, _ in ANNOTATORS if key in judgments[name]
            ]
            if not grades:
                continue
            if max(grades) - min(grades) >= 2:
                disagreements += 1
            qrels[query_id][vid] = int(sum(grades) / len(grades))

    relevant_counts = [
        sum(1 for g in grades.values() if g >= 2) for grades in qrels.values()
    ]
    no_relevant = sum(1 for c in relevant_counts if c == 0)

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    QRELS_FILE.write_text(json.dumps(qrels, indent=2), encoding="utf-8")
    AGREEMENT_FILE.write_text(json.dumps({
        "krippendorff_alpha_ordinal": report.krippendorff_alpha,
        "interpretation": report.interpretation,
        "n_units": report.n_units,
        "annotators": [name for name, _ in ANNOTATORS],
        "pairwise_weighted_kappa": report.pairwise_kappa,
        "grade_distribution": report.label_distribution,
        "hard_disagreements": disagreements,
        "queries_with_no_relevant_verse": no_relevant,
        "standard": "SILVER — model-generated judgments. A human-judged subset "
                    "is required before publication. See DATASET_CARD.md.",
    }, indent=2), encoding="utf-8")

    print(f"\n  queries with >=1 relevant verse: {len(qrels) - no_relevant}/{len(qrels)}")
    print(f"  mean relevant per query: "
          f"{sum(relevant_counts) / max(1, len(relevant_counts)):.2f}")
    print(f"  hard disagreements (>=2 grades apart): {disagreements}")
    print(f"\n  qrels     -> {QRELS_FILE}")
    print(f"  agreement -> {AGREEMENT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
