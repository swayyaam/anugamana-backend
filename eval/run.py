#!/usr/bin/env python3
"""
Execute conditions over the benchmark and cache the ranked results.

Runs are cached per condition to data/eval/runs/{key}.json. Retrieval over 389
queries is expensive, judging is expensive, and the analysis will be re-run many
times — so execution and analysis are separated. A cached run is keyed by the
condition and the benchmark file, and `--force` recomputes.

Nothing here computes a metric. Ranked lists only. Analysis lives in
scripts/analyze.py so that changing how results are scored never silently
changes what was retrieved.

Usage:
    python -m eval.run                          # every runnable condition
    python -m eval.run --conditions C0,C2,C5,C10
    python -m eval.run --limit 20               # smoke test
    python -m eval.run --list                   # what can run right now
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_DIR, SARVAM_ENABLED  # noqa: E402
from app.services.pipeline import run as run_pipeline  # noqa: E402
from app.services.retrieval import RAW_CHROMA_DIR  # noqa: E402
from eval.conditions import (  # noqa: E402
    GENERALISATION_GRID,
    GRID,
    MULTILINGUAL_GRID,
    Condition,
)
from eval.lexical import BM25Retriever, load_corpus, parametric_retrieve_many  # noqa: E402

BENCHMARK_FILE = DATA_DIR / "benchmark" / "queries.json"
RUNS_DIR = DATA_DIR / "eval" / "runs"
DEPTH = 10

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
)


def availability() -> dict[str, bool]:
    """What the machine can actually run right now."""
    from app.config import ANTHROPIC_API_KEY, CHROMA_DIR, ENRICHED_FILE
    from app.services.retrieval import MEDITATIONS_DIR

    return {
        "meditations_corpus": (MEDITATIONS_DIR / "enriched.json").exists(),
        "meditations_index": (MEDITATIONS_DIR / "chroma").exists(),
        "corpus": ENRICHED_FILE.exists(),
        "enriched_index": CHROMA_DIR.exists(),
        "raw_index": RAW_CHROMA_DIR.exists(),
        "anthropic_api": bool(ANTHROPIC_API_KEY),
        "sarvam": SARVAM_ENABLED,
    }


def load_benchmark(limit: int | None = None) -> list[dict]:
    queries = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    return queries[:limit] if limit else queries


# ---------------------------------------------------------------------------
# Per-kind execution
# ---------------------------------------------------------------------------

async def run_pipeline_condition(
    condition: Condition, queries: list[dict], concurrency: int = 4
) -> dict[str, list[str]]:
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, list[str]] = {}
    done = 0
    started = time.time()

    async def one(item: dict) -> None:
        nonlocal done
        async with semaphore:
            try:
                outcome = await run_pipeline(item["query"], condition.config)
                results[item["query_id"]] = outcome.verse_ids[:DEPTH]
            except Exception as e:
                print(f"\n  {RED}query {item['query_id']} failed: {e}{RESET}")
                results[item["query_id"]] = []
        done += 1
        if done % 20 == 0 or done == len(queries):
            rate = done / max(1e-9, time.time() - started)
            print(f"\r  {done}/{len(queries)}  ({rate:.1f} q/s)", end="", flush=True)

    await asyncio.gather(*(one(item) for item in queries))
    print()
    return results


def run_bm25_condition(
    condition: Condition, queries: list[dict]
) -> dict[str, list[str]]:
    retriever = BM25Retriever(load_corpus(), include_purport=(condition.key == "C1"))
    return {
        item["query_id"]: retriever.retrieve(item["query"], DEPTH)
        for item in queries
    }


async def run_parametric_condition(queries: list[dict]) -> dict[str, list[str]]:
    rankings = await parametric_retrieve_many([q["query"] for q in queries], DEPTH)
    return {item["query_id"]: ranking for item, ranking in zip(queries, rankings)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def execute(condition: Condition, queries: list[dict]) -> dict[str, list[str]]:
    if condition.kind == "bm25":
        return run_bm25_condition(condition, queries)
    if condition.kind == "bm25_meditations":
        from app.services.retrieval import MEDITATIONS_DIR

        retriever = BM25Retriever(
            load_corpus(MEDITATIONS_DIR / "enriched.json"), include_purport=False
        )
        return {
            item["query_id"]: retriever.retrieve(item["query"], DEPTH)
            for item in queries
        }
    if condition.kind == "parametric":
        return await run_parametric_condition(queries)
    return await run_pipeline_condition(condition, queries)


def cache_path(condition: Condition, suffix: str = "") -> Path:
    return RUNS_DIR / f"{condition.key}{suffix}.json"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", help="comma-separated keys, default all")
    parser.add_argument("--limit", type=int, help="only the first N queries")
    parser.add_argument("--force", action="store_true", help="ignore cached runs")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--include-multilingual", action="store_true")
    parser.add_argument("--include-generalisation", action="store_true",
                        help="run the Meditations replication conditions")
    parser.add_argument("--benchmark", help="alternate benchmark file "
                        "(e.g. the translated Indic query set)")
    parser.add_argument("--suffix", default="", help="suffix for cached run files, "
                        "so a second benchmark does not overwrite the first")
    args = parser.parse_args()

    global BENCHMARK_FILE
    if args.benchmark:
        BENCHMARK_FILE = Path(args.benchmark)

    have = availability()
    grid = (
        GRID
        + (MULTILINGUAL_GRID if args.include_multilingual else [])
        + (GENERALISATION_GRID if args.include_generalisation else [])
    )

    if args.list:
        print(f"\n{'key':<6} {'runnable':<10} {'requires':<16} isolates")
        print("-" * 90)
        for condition in grid:
            ok = have.get(condition.requires, False)
            mark = f"{GREEN}yes{RESET}" if ok else f"{RED}no{RESET}"
            print(f"{condition.key:<6} {mark:<19} {condition.requires:<16} "
                  f"{condition.isolates}")
        print(f"\n{DIM}availability: "
              + ", ".join(f"{k}={v}" for k, v in have.items()) + RESET)
        return 0

    if not BENCHMARK_FILE.exists():
        print(f"{RED}No benchmark at {BENCHMARK_FILE}{RESET}")
        print("Build one:  python scripts/build_benchmark.py")
        return 2

    queries = load_benchmark(args.limit)
    print(f"Benchmark: {len(queries)} queries")

    wanted = set(args.conditions.split(",")) if args.conditions else None
    selected = [c for c in grid if wanted is None or c.key in wanted]

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    skipped: list[tuple[str, str]] = []

    for condition in selected:
        if not have.get(condition.requires, False):
            skipped.append((condition.key, condition.requires))
            continue

        path = cache_path(condition, args.suffix)
        if path.exists() and not args.force:
            cached = json.loads(path.read_text())
            if len(cached.get("results", {})) >= len(queries):
                print(f"{DIM}{condition.key:<6} cached{RESET}")
                continue

        print(f"\n{YELLOW}{condition.key}{RESET}  {condition.label}")
        started = time.time()
        results = await execute(condition, queries)
        elapsed = time.time() - started

        empty = sum(1 for r in results.values() if not r)
        path.write_text(json.dumps({
            "condition": condition.key,
            "label": condition.label,
            "isolates": condition.isolates,
            "kind": condition.kind,
            "depth": DEPTH,
            "n_queries": len(results),
            "empty_results": empty,
            "elapsed_s": round(elapsed, 1),
            "results": results,
        }, indent=2), encoding="utf-8")
        print(f"  {GREEN}done{RESET} in {elapsed:.0f}s "
              f"({elapsed / max(1, len(queries)):.2f}s/query), "
              f"{empty} empty -> {path.name}")

    # A grid that silently drops what it cannot run is how "we measured
    # everything" becomes false.
    if skipped:
        print(f"\n{YELLOW}SKIPPED — missing prerequisites:{RESET}")
        for key, requires in skipped:
            print(f"  {key:<6} needs {requires}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
