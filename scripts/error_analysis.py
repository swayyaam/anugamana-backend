#!/usr/bin/env python3
"""
Where the system fails, and what kind of failure it is.

A results table says whether a system is better. It does not say what is wrong
with it, and the failures are where the interesting writing is. This script
produces the qualitative half of the evaluation:

  * the worst queries for the served system, with what it returned instead
  * failure taxonomy — was the right verse never retrieved, or retrieved and
    then ranked away? Those have completely different fixes
  * famous-verse bias — do conditions differ in how much they lean on the
    handful of verses everyone quotes?
  * where each condition uniquely wins, which is what an ablation is actually for

Usage:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --condition C10 --worst 20
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_DIR, ENRICHED_FILE  # noqa: E402
from eval import metrics as M  # noqa: E402

BENCHMARK_DIR = DATA_DIR / "benchmark"
RUNS_DIR = DATA_DIR / "eval" / "runs"
OUT_FILE = DATA_DIR / "eval" / "error_analysis.json"

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"
)


def load_runs() -> dict[str, dict[str, list[str]]]:
    runs = {}
    for path in sorted(RUNS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[payload["condition"]] = payload["results"]
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", default="C10")
    parser.add_argument("--worst", type=int, default=20)
    parser.add_argument("--metric", default="ndcg@10", choices=list(M.METRICS))
    args = parser.parse_args()

    qrels_path = BENCHMARK_DIR / "qrels.json"
    if not qrels_path.exists():
        print(f"{RED}No qrels — run scripts/pool_and_judge.py first{RESET}")
        return 2

    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    queries = json.loads((BENCHMARK_DIR / "queries.json").read_text(encoding="utf-8"))
    query_text = {q["query_id"]: q["query"] for q in queries}
    query_meta = {q["query_id"]: q for q in queries}
    runs = load_runs()
    verses = {v["verse_id"]: v for v in json.loads(ENRICHED_FILE.read_text())}

    scorable = [
        qid for qid in query_text
        if any(g >= M.RELEVANT_THRESHOLD for g in qrels.get(qid, {}).values())
    ]

    if args.condition not in runs:
        print(f"{RED}No run for {args.condition}{RESET}")
        return 2
    target = runs[args.condition]

    metric_fn, k = M.METRICS[args.metric]
    scores = {
        qid: metric_fn(target.get(qid, []), qrels.get(qid, {}), k) for qid in scorable
    }

    report: dict = {"condition": args.condition, "metric": args.metric}

    # --- failure taxonomy --------------------------------------------------
    # Retrieval and ranking failures need different fixes, so separating them is
    # the single most actionable thing this analysis produces.
    never_retrieved = ranked_away = fine = 0
    for qid in scorable:
        ranked = target.get(qid, [])
        relevant = {v for v, g in qrels[qid].items() if g >= M.RELEVANT_THRESHOLD}
        if scores[qid] >= 0.5:
            fine += 1
        elif relevant & set(ranked):
            ranked_away += 1
        else:
            never_retrieved += 1

    total = len(scorable)
    print(f"\n{BOLD}Failure taxonomy — {args.condition}{RESET}  "
          f"({total} scorable queries)")
    print(f"  {GREEN}acceptable{RESET} ({args.metric} >= 0.5)      {fine:>4}  "
          f"{fine / total:>6.1%}")
    print(f"  {YELLOW}ranked away{RESET} (retrieved, poorly ranked) {ranked_away:>4}  "
          f"{ranked_away / total:>6.1%}   {DIM}-> reranking problem{RESET}")
    print(f"  {RED}never retrieved{RESET} (absent from top-10)  {never_retrieved:>4}  "
          f"{never_retrieved / total:>6.1%}   {DIM}-> recall problem{RESET}")
    report["taxonomy"] = {
        "acceptable": fine, "ranked_away": ranked_away,
        "never_retrieved": never_retrieved, "total": total,
    }

    # --- failure by register ----------------------------------------------
    print(f"\n{BOLD}Failure rate by query register{RESET}")
    by_register: dict[str, list[float]] = {}
    for qid in scorable:
        by_register.setdefault(query_meta[qid]["register"], []).append(scores[qid])
    for register, values in sorted(by_register.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {register:<24} n={len(values):<4} mean {args.metric} = "
              f"{sum(values) / len(values):.4f}")
    report["by_register"] = {
        r: {"n": len(v), "mean": sum(v) / len(v)} for r, v in by_register.items()
    }

    # --- famous-verse concentration ---------------------------------------
    print(f"\n{BOLD}Famous-verse concentration{RESET}")
    print(f"{DIM}share of a condition's top-1 results taken by its 5 most-returned "
          f"verses — a high value means the system answers everything with the "
          f"same few verses{RESET}")
    for key in sorted(runs):
        top1 = Counter(r[0] for r in runs[key].values() if r)
        if not top1:
            continue
        concentration = sum(c for _, c in top1.most_common(5)) / sum(top1.values())
        common = ", ".join(f"{v}({c})" for v, c in top1.most_common(3))
        flag = f"  {RED}<-{RESET}" if concentration > 0.5 else ""
        print(f"  {key:<6} {concentration:>6.1%}  distinct top-1: "
              f"{len(top1):>3}   {DIM}{common}{RESET}{flag}")
    report["concentration"] = {
        key: {
            "top5_share": sum(c for _, c in Counter(
                r[0] for r in runs[key].values() if r).most_common(5)
            ) / max(1, sum(Counter(r[0] for r in runs[key].values() if r).values())),
            "distinct_top1": len(Counter(r[0] for r in runs[key].values() if r)),
        }
        for key in runs
    }

    # --- worst queries -----------------------------------------------------
    worst = sorted(scorable, key=lambda q: scores[q])[: args.worst]
    print(f"\n{BOLD}Worst {len(worst)} queries for {args.condition}{RESET}")
    worst_report = []
    for qid in worst:
        ranked = target.get(qid, [])
        relevant = sorted(
            ((v, g) for v, g in qrels[qid].items() if g >= M.RELEVANT_THRESHOLD),
            key=lambda x: -x[1],
        )[:2]
        print(f"\n  {DIM}{qid}  {args.metric}={scores[qid]:.3f}  "
              f"register={query_meta[qid]['register']}{RESET}")
        print(f"  Q: {query_text[qid][:96]}")
        if ranked:
            top = ranked[0]
            print(f"  {RED}returned{RESET} {top}: {verses[top]['translation'][:72]}"
                  if top in verses else f"  returned {top}")
        for vid, grade in relevant:
            print(f"  {GREEN}wanted{RESET}   {vid} (grade {grade}): "
                  f"{verses[vid]['translation'][:72]}")
        worst_report.append({
            "query_id": qid, "query": query_text[qid], "score": scores[qid],
            "register": query_meta[qid]["register"],
            "returned": ranked[:3], "relevant": [v for v, _ in relevant],
        })
    report["worst_queries"] = worst_report

    # --- unique wins -------------------------------------------------------
    print(f"\n{BOLD}Unique contribution per condition{RESET}")
    print(f"{DIM}queries where this condition finds a relevant verse in its top-3 "
          f"that no other condition does{RESET}")
    unique_counts = {}
    for key in sorted(runs):
        unique = 0
        for qid in scorable:
            relevant = {v for v, g in qrels[qid].items() if g >= M.RELEVANT_THRESHOLD}
            mine = set(runs[key].get(qid, [])[:3]) & relevant
            if not mine:
                continue
            others = set()
            for other_key, other in runs.items():
                if other_key != key:
                    others |= set(other.get(qid, [])[:3]) & relevant
            if mine - others:
                unique += 1
        unique_counts[key] = unique
        print(f"  {key:<6} {unique:>4} queries")
    report["unique_wins"] = unique_counts

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{DIM}written -> {OUT_FILE}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
