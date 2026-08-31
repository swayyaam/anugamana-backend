#!/usr/bin/env python3
"""
Compare rerankers by reordering candidates that were already retrieved.

Why offline
-----------
69.6% of the served system's failures are queries where a relevant verse *was*
retrieved and then ranked too low. That is a ranking problem, and it deserves a
ranking experiment — not a full pipeline run, which confounds retrieval and
ranking and costs an API call per query.

So: take a cached run's top-10 for each query, hold that candidate set fixed, and
let each reranker reorder it. nDCG@10 over a fixed set of ten can only change by
reordering, so any difference is attributable to the reranker alone. It also runs
in minutes with no network.

Variants tested per model, because what you show a reranker matters as much as
which reranker it is:
    translation  — the raw verse text (what the pipeline used)
    enriched     — the generated meaning fields
    both         — translation followed by the meaning fields

Usage:
    python -m eval.rerank_offline
    python -m eval.rerank_offline --run C13 --models bge,msmarco
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_DIR, ENRICHED_FILE, RERANK_MODEL  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval import stats  # noqa: E402

BENCHMARK_DIR = DATA_DIR / "benchmark"
RUNS_DIR = DATA_DIR / "eval" / "runs"
OUT_FILE = DATA_DIR / "eval" / "rerank_comparison.json"

MODELS = {
    "msmarco": RERANK_MODEL,
    "bge": "BAAI/bge-reranker-v2-m3",
}

BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"
)


def document_for(verse: dict, variant: str) -> str:
    translation = verse["translation"]
    enriched = verse.get("text_for_embedding", "")
    if variant == "translation":
        return translation
    if variant == "enriched":
        return enriched or translation
    return f"{translation}\n\n{enriched}"[:2000]


def score_run(order_by_query: dict[str, list[str]], qrels: dict, ids: list[str]):
    return [
        M.ndcg_at_k(order_by_query.get(qid, []), qrels.get(qid, {}), 10)
        for qid in ids
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="C13", help="cached run supplying candidates")
    parser.add_argument("--models", default="msmarco,bge")
    parser.add_argument("--variants", default="translation,enriched,both")
    args = parser.parse_args()

    run_path = RUNS_DIR / f"{args.run}.json"
    if not run_path.exists():
        print(f"No cached run at {run_path}")
        return 2

    qrels = json.loads((BENCHMARK_DIR / "qrels.json").read_text(encoding="utf-8"))
    queries = {
        q["query_id"]: q["query"]
        for q in json.loads((BENCHMARK_DIR / "queries.json").read_text())
    }
    verses = {v["verse_id"]: v for v in json.loads(ENRICHED_FILE.read_text())}
    candidates = json.loads(run_path.read_text(encoding="utf-8"))["results"]

    ids = [
        qid for qid in candidates
        if qid in queries
        and len(candidates[qid]) > 1
        and any(g >= M.RELEVANT_THRESHOLD for g in qrels.get(qid, {}).values())
    ]

    print(f"\n{BOLD}Reranking {args.run}'s candidates{RESET}")
    print(f"{DIM}  {len(ids)} queries · candidate set held fixed, so any change "
          f"is the reranker alone{RESET}\n")

    baseline_scores = score_run(candidates, qrels, ids)
    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    print(f"{'model':<10} {'shown':<12} {'ndcg@10':>9} {'delta':>9} "
          f"{'95% CI':>20} {'p':>8}")
    print("-" * 74)
    print(f"{'(none)':<10} {'RRF order':<12} {baseline_mean:>9.4f} "
          f"{'—':>9} {'—':>20} {'—':>8}")

    from app.services.reranker import _load_cross_encoder

    results = []
    for model_key in args.models.split(","):
        model_name = MODELS.get(model_key, model_key)
        try:
            model = _load_cross_encoder(model_name)
        except Exception as e:
            print(f"{model_key:<10} {RED}unavailable: {str(e)[:40]}{RESET}")
            continue

        for variant in args.variants.split(","):
            pairs, index = [], []
            for qid in ids:
                for vid in candidates[qid]:
                    if vid in verses:
                        pairs.append((queries[qid], document_for(verses[vid], variant)))
                        index.append((qid, vid))

            logits = model.predict(pairs, show_progress_bar=False)

            by_query: dict[str, list[tuple[float, str]]] = {}
            for (qid, vid), score in zip(index, logits):
                by_query.setdefault(qid, []).append((float(score), vid))
            reordered = {
                qid: [vid for _, vid in sorted(items, reverse=True)]
                for qid, items in by_query.items()
            }

            scores = score_run(reordered, qrels, ids)
            comparison = stats.compare(
                "ndcg@10", "rrf", f"{model_key}:{variant}",
                baseline_scores, scores, iterations=5000,
            )
            results.append(comparison)
            mark = (
                f"{GREEN}+{RESET}" if comparison.significant and comparison.delta > 0
                else f"{RED}-{RESET}" if comparison.significant
                else " "
            )
            print(f"{model_key:<10} {variant:<12} {comparison.system_mean:>9.4f} "
                  f"{comparison.delta:>+9.4f} "
                  f"[{comparison.ci_low:+.4f},{comparison.ci_high:+.4f}] "
                  f"{comparison.p_value:>8.4f}{mark}")

    stats.holm_correction(results)
    winners = [c for c in results if c.significant and c.delta > 0]

    print()
    if winners:
        best = max(winners, key=lambda c: c.delta)
        print(f"{GREEN}Best: {best.system}{RESET} — {best.delta:+.4f} nDCG@10 "
              f"over RRF order (Holm p={best.p_adjusted:.4f}, "
              f"W/L/T={best.wins}/{best.losses}/{best.ties})")
    else:
        print(f"{YELLOW}No reranker beats plain RRF ordering.{RESET} That is a "
              f"result, not a gap: it says off-the-shelf cross-encoders do not "
              f"transfer to this corpus, and that a domain-trained reranker is "
              f"the only route worth taking.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "candidates_from": args.run,
        "n_queries": len(ids),
        "rrf_baseline_ndcg": baseline_mean,
        "results": [
            {
                "system": c.system, "ndcg": c.system_mean, "delta": c.delta,
                "ci_low": c.ci_low, "ci_high": c.ci_high,
                "p_value": c.p_value, "p_holm": c.p_adjusted,
                "significant": c.significant,
                "wins": c.wins, "losses": c.losses, "ties": c.ties,
            } for c in results
        ],
    }, indent=2), encoding="utf-8")
    print(f"{DIM}written -> {OUT_FILE}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
