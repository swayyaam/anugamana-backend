#!/usr/bin/env python3
"""
Human annotation CLI — produces the gold subset the benchmark needs.

Why this exists
---------------
scripts/pool_and_judge.py produces *silver* judgments: two language models
grading a pool. Their agreement measures consistency, not correctness — models
can be consistently wrong, and they share training data with each other and with
the system under evaluation. A published claim needs human judgments on a
stratified subset, and a reported correlation between the human and model grades.

This tool collects them. It samples across overlap strata and disagreement so the
subset is informative rather than merely small, records who judged what and when,
and saves after every item so a session can be abandoned and resumed.

Usage:
    python scripts/annotate.py --annotator swayam
    python scripts/annotate.py --annotator friend2 --sample 60
    python scripts/annotate.py --report          # human vs model correlation
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_DIR, ENRICHED_FILE  # noqa: E402
from eval.agreement import analyse, weighted_kappa  # noqa: E402

BENCHMARK_DIR = DATA_DIR / "benchmark"
GOLD_DIR = BENCHMARK_DIR / "human"

BOLD, DIM, GREEN, YELLOW, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"
)

RUBRIC = """
  3  directly addresses the query — the verse a knowledgeable person would cite
  2  clearly relevant — addresses a substantial part of the situation
  1  tangentially related — shares a theme but not the situation
  0  not relevant

  s  skip this item      u  undo previous      q  save and quit
"""


def load(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def sample_items(qrels: dict, queries: list[dict], size: int, seed: int) -> list[tuple[str, str]]:
    """
    Sample (query_id, verse_id) pairs, stratified by model grade so the subset
    spans the scale. Judging only the model's 3s would measure precision alone
    and tell us nothing about what it wrongly rejected.
    """
    by_grade: dict[int, list[tuple[str, str]]] = {}
    valid = {q["query_id"] for q in queries}
    for query_id, grades in qrels.items():
        if query_id not in valid:
            continue
        for verse_id, grade in grades.items():
            by_grade.setdefault(int(grade), []).append((query_id, verse_id))

    rng = random.Random(seed)
    per_grade = max(1, size // max(1, len(by_grade)))
    chosen: list[tuple[str, str]] = []
    for grade in sorted(by_grade):
        pool = by_grade[grade]
        rng.shuffle(pool)
        chosen.extend(pool[:per_grade])
    rng.shuffle(chosen)
    return chosen[:size]


def annotate(args) -> int:
    qrels = load(BENCHMARK_DIR / "qrels.json", {})
    queries = load(BENCHMARK_DIR / "queries.json", [])
    if not qrels or not queries:
        print("Need data/benchmark/qrels.json and queries.json first.")
        return 2

    query_text = {q["query_id"]: q["query"] for q in queries}
    verses = {v["verse_id"]: v for v in json.loads(ENRICHED_FILE.read_text())}

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLD_DIR / f"{args.annotator}.json"
    existing = load(out_path, {})

    items = [
        item for item in sample_items(qrels, queries, args.sample, args.seed)
        if f"{item[0]}|{item[1]}" not in existing
    ]

    if not items:
        print(f"{GREEN}Nothing left to judge for '{args.annotator}'.{RESET} "
              f"({len(existing)} already done)")
        return 0

    print(f"\n{BOLD}Relevance annotation — {args.annotator}{RESET}")
    print(f"{len(items)} items to judge, {len(existing)} already saved.")
    print(f"{DIM}You are NOT shown the model's grade. That is deliberate.{RESET}")
    print(RUBRIC)

    history: list[str] = []
    for index, (query_id, verse_id) in enumerate(items, 1):
        verse = verses.get(verse_id)
        if not verse:
            continue

        print(f"\n{'─' * 74}")
        print(f"{DIM}{index}/{len(items)}{RESET}")
        print(f"{BOLD}Query:{RESET} {query_text[query_id]}")
        print(f"\n{CYAN}Verse {verse_id}{RESET}")
        print(f"  {verse['translation']}")

        while True:
            answer = input(f"\n  grade [0-3/s/u/q] > ").strip().lower()
            if answer in {"0", "1", "2", "3"}:
                existing[f"{query_id}|{verse_id}"] = int(answer)
                history.append(f"{query_id}|{verse_id}")
                break
            if answer == "s":
                break
            if answer == "u" and history:
                existing.pop(history.pop(), None)
                print(f"  {YELLOW}undone{RESET}")
                continue
            if answer == "q":
                out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
                print(f"\n{GREEN}Saved {len(existing)} judgments -> {out_path}{RESET}")
                return 0
            print("  enter 0, 1, 2, 3, s, u or q")

        out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    (GOLD_DIR / f"{args.annotator}.meta.json").write_text(json.dumps({
        "annotator": args.annotator,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_judgments": len(existing),
        "sample_seed": args.seed,
    }, indent=2), encoding="utf-8")

    print(f"\n{GREEN}Done. {len(existing)} judgments -> {out_path}{RESET}")
    print("Report human vs model correlation:  python scripts/annotate.py --report")
    return 0


def report() -> int:
    human_files = sorted(GOLD_DIR.glob("*.json"))
    human_files = [p for p in human_files if not p.name.endswith(".meta.json")]
    if not human_files:
        print("No human judgments yet. Run: python scripts/annotate.py --annotator NAME")
        return 2

    humans = {p.stem: load(p, {}) for p in human_files}
    model_qrels = load(BENCHMARK_DIR / "qrels.json", {})
    model_flat = {
        f"{qid}|{vid}": int(grade)
        for qid, grades in model_qrels.items()
        for vid, grade in grades.items()
    }

    print(f"\n{BOLD}Human annotation report{RESET}")
    if len(humans) >= 2:
        print("\nHuman-human agreement:")
        print(analyse(humans).format())
    else:
        print(f"\n{YELLOW}Only one human annotator — inter-human agreement "
              f"cannot be computed. A second is required for publication.{RESET}")

    print("\nHuman vs model (silver) agreement:")
    for name, grades in humans.items():
        shared = [k for k in grades if k in model_flat]
        if not shared:
            continue
        kappa = weighted_kappa(
            [grades[k] for k in shared], [model_flat[k] for k in shared]
        )
        exact = sum(1 for k in shared if grades[k] == model_flat[k]) / len(shared)
        within_one = sum(
            1 for k in shared if abs(grades[k] - model_flat[k]) <= 1
        ) / len(shared)
        print(f"  {name:<12} n={len(shared):<5} weighted kappa={kappa:.4f}  "
              f"exact={exact:.1%}  within-1={within_one:.1%}")

    print(f"\n{DIM}A weighted kappa below ~0.6 against humans means the silver "
          f"judgments cannot carry a published claim on their own.{RESET}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", help="your name — one file per annotator")
    parser.add_argument("--sample", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.report:
        return report()
    if not args.annotator:
        parser.error("--annotator is required (or use --report)")
    return annotate(args)


if __name__ == "__main__":
    raise SystemExit(main())
