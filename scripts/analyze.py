#!/usr/bin/env python3
"""
Score the cached runs and report the result.

This is the only script that turns retrieved rankings into numbers, and it
enforces the conditions listed in RETRACTION.md:

  * graded, pooled judgments (never a single binary gold verse)
  * 95% CIs from a paired bootstrap on every reported difference
  * a two-sided randomisation test, Holm-corrected across the comparison family
  * per-query win/loss/tie counts alongside every mean
  * the planned contrasts, stated in eval/conditions.py before any number
    existed, reported whether or not they came out favourably
  * overlap-stratified results, because "enrichment helps" is a weaker and less
    useful claim than "enrichment helps *here*, by this much"

Usage:
    python scripts/analyze.py
    python scripts/analyze.py --metric ndcg@10
    python scripts/analyze.py --no-strata
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_DIR, ENRICHED_FILE  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval import stats  # noqa: E402
from eval.conditions import BY_KEY, PLANNED_CONTRASTS  # noqa: E402
from eval.overlap import build_idf, stratum_of, weighted_overlap  # noqa: E402

BENCHMARK_DIR = DATA_DIR / "benchmark"
RUNS_DIR = DATA_DIR / "eval" / "runs"
REPORT_FILE = DATA_DIR / "eval" / "results.json"

GREEN, RED, YELLOW, BOLD, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[2m", "\033[0m"
)


def load_runs() -> dict[str, dict[str, list[str]]]:
    runs = {}
    for path in sorted(RUNS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[payload["condition"]] = payload["results"]
    return runs


def per_query_scores(
    ranked_by_query: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    query_ids: list[str],
) -> dict[str, list[float]]:
    """metric -> per-query values, aligned to `query_ids` so runs stay paired."""
    rows = [
        M.score_query(ranked_by_query.get(qid, []), qrels.get(qid, {}))
        for qid in query_ids
    ]
    return {name: [row[name] for row in rows] for name in M.METRICS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="ndcg@10", choices=list(M.METRICS))
    parser.add_argument("--no-strata", action="store_true")
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()

    qrels_path = BENCHMARK_DIR / "qrels.json"
    if not qrels_path.exists():
        print(f"{RED}No qrels at {qrels_path}{RESET}")
        print("Run: python scripts/pool_and_judge.py")
        return 2

    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    queries = json.loads((BENCHMARK_DIR / "queries.json").read_text(encoding="utf-8"))
    runs = load_runs()
    if not runs:
        print(f"{RED}No runs in {RUNS_DIR}{RESET}. Run: python -m eval.run")
        return 2

    # Only queries with at least one relevant verse can discriminate systems.
    query_ids = [
        q["query_id"] for q in queries
        if any(g >= M.RELEVANT_THRESHOLD for g in qrels.get(q["query_id"], {}).values())
    ]
    query_text = {q["query_id"]: q["query"] for q in queries}
    dropped = len(queries) - len(query_ids)

    print(f"\n{BOLD}Anugamana — retrieval evaluation{RESET}")
    print(f"{DIM}benchmark: {len(queries)} queries, {len(query_ids)} with at least "
          f"one relevant verse ({dropped} dropped){RESET}")

    agreement_path = BENCHMARK_DIR / "agreement.json"
    if agreement_path.exists():
        agreement = json.loads(agreement_path.read_text())
        print(f"{DIM}judgments: {agreement['standard'].split('.')[0]}; "
              f"Krippendorff alpha = {agreement['krippendorff_alpha_ordinal']:.3f} "
              f"({agreement['interpretation']}){RESET}")

    scored = {key: per_query_scores(runs[key], qrels, query_ids) for key in runs}

    # --- main table --------------------------------------------------------
    print(f"\n{BOLD}All conditions ({len(query_ids)} queries){RESET}")
    header = f"{'key':<6} {'ndcg@10':>8} {'mrr@10':>8} {'r@5':>7} {'r@10':>7} {'p@1':>7}  isolates"
    print(header)
    print("-" * len(header))
    for key in sorted(scored, key=lambda k: -sum(scored[k][args.metric]) / len(query_ids)):
        values = scored[key]
        mean = lambda name: sum(values[name]) / len(query_ids)
        label = BY_KEY[key].isolates if key in BY_KEY else ""
        print(f"{key:<6} {mean('ndcg@10'):>8.4f} {mean('mrr@10'):>8.4f} "
              f"{mean('recall@5'):>7.4f} {mean('recall@10'):>7.4f} "
              f"{mean('p@1'):>7.4f}  {DIM}{label}{RESET}")

    # --- planned contrasts -------------------------------------------------
    print(f"\n{BOLD}Planned contrasts — metric: {args.metric}{RESET}")
    print(f"{DIM}stated in eval/conditions.py before any number existed{RESET}\n")

    comparisons = []
    questions = {}
    for baseline, system, question in PLANNED_CONTRASTS:
        if baseline not in scored or system not in scored:
            print(f"  {YELLOW}skipped{RESET} {baseline} vs {system} "
                  f"{DIM}(condition not run){RESET}")
            continue
        comparison = stats.compare(
            args.metric, baseline, system,
            scored[baseline][args.metric], scored[system][args.metric],
            iterations=args.iterations,
        )
        comparisons.append(comparison)
        questions[(baseline, system)] = question

    stats.holm_correction(comparisons)

    for comparison in comparisons:
        question = questions[(comparison.baseline, comparison.system)]
        verdict = (
            f"{GREEN}YES{RESET}" if comparison.significant and comparison.delta > 0
            else f"{RED}NO — worse{RESET}" if comparison.significant and comparison.delta < 0
            else f"{YELLOW}NOT SHOWN{RESET}"
        )
        print(f"  {question}")
        print(f"    {comparison.baseline} -> {comparison.system}   {verdict}")
        print(f"    {comparison.baseline}={comparison.baseline_mean:.4f}  "
              f"{comparison.system}={comparison.system_mean:.4f}  "
              f"delta={comparison.delta:+.4f} "
              f"95% CI [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}]")
        print(f"    p={comparison.p_value:.4f} (Holm {comparison.p_adjusted:.4f})  "
              f"W/L/T={comparison.wins}/{comparison.losses}/{comparison.ties}\n")

    # --- overlap stratification -------------------------------------------
    strata_report = {}
    if not args.no_strata:
        print(f"{BOLD}Stratified by query-document lexical overlap{RESET}")
        print(f"{DIM}the project's central claim is that gain concentrates in the "
              f"low-overlap regime{RESET}")

        verses = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
        texts = {v["verse_id"]: v["translation"] for v in verses}
        idf = build_idf(list(texts.values()))

        # A query's overlap is measured against its best relevant verse.
        buckets: dict[str, list[str]] = {}
        for qid in query_ids:
            relevant = [v for v, g in qrels[qid].items() if g >= M.RELEVANT_THRESHOLD]
            best = max(
                (weighted_overlap(query_text[qid], texts.get(v, ""), idf)
                 for v in relevant),
                default=0.0,
            )
            buckets.setdefault(stratum_of(best), []).append(qid)

        keys = sorted(scored, key=lambda k: -sum(scored[k][args.metric]) / len(query_ids))
        shown = [k for k in ("C0", "C2", "C5", "C10", "P0") if k in scored] or keys[:5]

        print(f"\n{'stratum':<10} {'n':>5}  " + "  ".join(f"{k:>8}" for k in shown))
        print("-" * (17 + 10 * len(shown)))
        for name in ("none", "low", "medium", "high"):
            ids = buckets.get(name, [])
            if not ids:
                continue
            index = {qid: i for i, qid in enumerate(query_ids)}
            row = []
            for key in shown:
                values = [scored[key][args.metric][index[qid]] for qid in ids]
                row.append(sum(values) / len(values))
            strata_report[name] = {"n": len(ids), **dict(zip(shown, row))}
            print(f"{name:<10} {len(ids):>5}  " + "  ".join(f"{v:>8.4f}" for v in row))

    # --- persist -----------------------------------------------------------
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps({
        "metric": args.metric,
        "n_queries": len(query_ids),
        "n_queries_dropped_no_relevant": dropped,
        "conditions": {
            key: {name: sum(values[name]) / len(query_ids) for name in M.METRICS}
            for key, values in scored.items()
        },
        "planned_contrasts": [
            {
                "question": questions[(c.baseline, c.system)],
                "baseline": c.baseline, "system": c.system,
                "baseline_mean": c.baseline_mean, "system_mean": c.system_mean,
                "delta": c.delta, "ci_low": c.ci_low, "ci_high": c.ci_high,
                "p_value": c.p_value, "p_holm": c.p_adjusted,
                "significant": c.significant,
                "wins": c.wins, "losses": c.losses, "ties": c.ties,
            }
            for c in comparisons
        ],
        "strata": strata_report,
    }, indent=2), encoding="utf-8")

    print(f"\n{DIM}written -> {REPORT_FILE}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
