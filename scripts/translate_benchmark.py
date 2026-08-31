#!/usr/bin/env python3
"""
Produce an Indic-language version of the benchmark for the cross-lingual grid.

Design
------
Query ids are preserved, so the *same* relevance judgments apply. This is the
standard CLIR construction: the information need is held constant and only its
surface language changes, which is what makes L1/L2/L3 a comparison of retrieval
strategies rather than a comparison of two different benchmarks.

The queries are translated with Mayura — the same component the pipeline uses at
request time under the "translate" strategy. That is deliberate but it must be
stated: condition L1 translates back to English with the same model that produced
the Hindi, so L1 enjoys a round-trip advantage that a genuine Hindi speaker's
query would not give it. L1 is therefore an *optimistic* bound on
translate-then-retrieve, and the paper must say so. L2 (embed the Indic query
directly) carries no such advantage.

Usage:
    python scripts/translate_benchmark.py --language hi-IN --limit 150
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_DIR, PIVOT_LANGUAGE, SARVAM_ENABLED  # noqa: E402
from app.services.sarvam.client import client  # noqa: E402
from app.services.sarvam.text import translate  # noqa: E402

BENCHMARK_DIR = DATA_DIR / "benchmark"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="hi-IN")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    if not SARVAM_ENABLED:
        print("SARVAM_API_KEY is not set — cross-lingual conditions need it.")
        return 2

    source = BENCHMARK_DIR / "queries.json"
    queries = json.loads(source.read_text(encoding="utf-8"))[: args.limit]
    out_path = BENCHMARK_DIR / f"queries_{args.language.split('-')[0]}.json"

    existing = {}
    if out_path.exists():
        existing = {q["query_id"]: q for q in json.loads(out_path.read_text())}
        print(f"resuming — {len(existing)} already translated")

    todo = [q for q in queries if q["query_id"] not in existing]
    print(f"translating {len(todo)} queries -> {args.language}")

    semaphore = asyncio.Semaphore(args.concurrency)
    failures = 0

    async def one(item: dict) -> dict | None:
        nonlocal failures
        async with semaphore:
            translated = await translate(item["query"], PIVOT_LANGUAGE, args.language)
        if not translated or translated == item["query"]:
            failures += 1
            return None
        return {
            **item,
            "query": translated,
            "query_en": item["query"],
            "language": args.language,
            "source": "mayura_translation_of_verse_blind_query",
        }

    results = list(existing.values())
    for start in range(0, len(todo), 40):
        batch = await asyncio.gather(*(one(q) for q in todo[start : start + 40]))
        results.extend(q for q in batch if q)
        results.sort(key=lambda q: q["query_id"])
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\r  {min(start + 40, len(todo))}/{len(todo)}", end="", flush=True)

    await client.aclose()

    print(f"\n\nwrote {len(results)} queries -> {out_path}")
    if failures:
        print(f"  {failures} translations failed and were dropped")
    if results:
        print(f"\n  sample:")
        for item in results[:2]:
            print(f"    en: {item['query_en'][:70]}")
            print(f"    {args.language}: {item['query'][:70]}")
    print(f"\nNext: python -m eval.run --benchmark {out_path} "
          f"--conditions L1,L2,L3 --include-multilingual --suffix _hi")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
