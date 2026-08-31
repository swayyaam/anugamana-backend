#!/usr/bin/env python3
"""
The contamination gate. No benchmark may be used before it passes this.

The 2026-05-10 evaluation was invalidated because every one of its 80 queries was
a verbatim substring of the text being searched — queries were mined by splitting
the enrichment's `situations` field, and that field is embedded as each verse's
`_meaning` vector. This script is the check that would have caught it.

Exit codes:
    0  clean
    1  contamination found
    2  could not run

Usage:
    python scripts/check_contamination.py data/benchmark/queries.json
    python scripts/check_contamination.py --self-test   # prove the gate works
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_DIR, ENRICHED_FILE  # noqa: E402
from eval.overlap import check_query, check_query_against_corpus  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
)

DEFAULT_BENCHMARK = DATA_DIR / "benchmark" / "queries.json"
CONTAMINATED_ARCHIVE = DATA_DIR / "archive" / "golden_dataset_20260510.CONTAMINATED.json"


def indexed_texts() -> dict[str, str]:
    """
    Everything that is embedded or lexically indexed, per verse. A query must
    not be derived from any of it.
    """
    verses = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
    texts = {}
    for verse in verses:
        parts = [
            verse.get("text_for_embedding", ""),
            verse.get("translation", ""),
            verse.get("purport", ""),
        ]
        fields = verse.get("meaning_fields", {}) or {}
        parts.extend(str(v) for v in fields.values())
        texts[verse["verse_id"]] = "\n".join(p for p in parts if p)
    return texts


def run(pairs: list[dict], label: str) -> int:
    texts = indexed_texts()

    # Verse-blind benchmarks carry no nominated gold verse, so each query is
    # checked against the entire corpus — a strictly stronger test, since a
    # query lifted from the text is contaminated whichever verse it came from.
    if pairs and "verse_id" not in pairs[0]:
        print(f"{DIM}no gold verses declared — checking every query against all "
              f"{len(texts)} verses{RESET}")
        reports = [check_query_against_corpus(p["query"], texts) for p in pairs]
    else:
        reports = [check_query(p["query"], texts, p["verse_id"]) for p in pairs]

    contaminated = [r for r in reports if r.contaminated]
    verbatim = [r for r in reports if r.verbatim]
    clean = len(reports) - len(contaminated)

    print(f"\n{'=' * 66}")
    print(f"Contamination check — {label}")
    print(f"{'=' * 66}")
    print(f"  queries checked      {len(reports)}")
    print(f"  clean                {GREEN if clean == len(reports) else ''}{clean}{RESET}")
    print(f"  verbatim in index    {RED if verbatim else GREEN}{len(verbatim)}{RESET}")
    print(f"  contaminated (any)   {RED if contaminated else GREEN}{len(contaminated)}{RESET}")

    if reports:
        shared = sorted(r.longest_shared_ngram for r in reports)
        print(f"  longest shared n-gram  median={shared[len(shared) // 2]}  "
              f"max={shared[-1]}")

    if contaminated:
        print(f"\n{RED}Contaminated examples:{RESET}")
        for report in contaminated[:8]:
            flag = "VERBATIM" if report.verbatim else f"{report.longest_shared_ngram}-gram"
            print(f"  {DIM}[{report.verse_id}] {flag}{RESET} {report.query[:78]}")

    print()
    if contaminated:
        print(f"{RED}FAIL{RESET} — this benchmark cannot be used. Queries must be "
              f"written without sight of the enrichment.")
        print(f"{DIM}See RETRACTION.md for why this check exists.{RESET}")
        return 1

    print(f"{GREEN}PASS{RESET} — no query is derived from the indexed text.")
    return 0


def self_test() -> int:
    """
    Prove the gate works by running it against the dataset that broke the
    original evaluation. A gate that has never caught anything is not evidence.
    """
    if not CONTAMINATED_ARCHIVE.exists():
        print(f"{YELLOW}Archive not found: {CONTAMINATED_ARCHIVE}{RESET}")
        return 2

    pairs = json.loads(CONTAMINATED_ARCHIVE.read_text(encoding="utf-8"))
    print(f"{YELLOW}SELF-TEST{RESET} — running the gate against the retracted "
          f"2026-05-10 benchmark.\nIt must FAIL. If it passes, the gate is broken.")
    status = run(pairs, "RETRACTED 2026-05-10 benchmark (expected: FAIL)")

    if status == 1:
        print(f"{GREEN}Self-test passed:{RESET} the gate correctly rejects the "
              f"contaminated benchmark.")
        return 0
    print(f"{RED}Self-test FAILED:{RESET} the gate did not detect known "
          f"contamination.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", nargs="?", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--self-test", action="store_true",
                        help="verify the gate against the known-bad archive")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    path = Path(args.benchmark)
    if not path.exists():
        print(f"{RED}Benchmark not found: {path}{RESET}")
        print("Build one with: python scripts/build_benchmark.py")
        return 2

    pairs = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(pairs, dict):
        pairs = pairs.get("queries", [])
    return run(pairs, str(path))


if __name__ == "__main__":
    raise SystemExit(main())
