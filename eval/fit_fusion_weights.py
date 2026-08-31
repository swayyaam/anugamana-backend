#!/usr/bin/env python3
"""
Fit per-arm fusion weights on the benchmark, with cross-validation.

Classic RRF gives every retrieval arm the same weight. That is a convenience, not
a finding: the dense meaning arm and the sparse lexical arm are demonstrably not
equally good at this task, and the emotion arm is a different kind of signal
again. Fitting the weights is a cheap, well-established IR win that needs no new
model.

Method
------
1. Compute each query's arm lists once (`retrieval.arm_lists`) and cache them.
   Fusion is then pure arithmetic, so thousands of weight settings can be
   evaluated without re-running retrieval.
2. Coordinate ascent on mean nDCG@10 over a fixed grid per arm.
3. **5-fold cross-validation.** Six free parameters on 381 queries will overfit
   if fitted and reported on the same data — the held-out score is the only one
   that means anything, and it is what this reports.

The weights are written to data/eval/fusion_weights.json. They are only adopted
if the held-out gain is positive; a fit that only helps in-sample is reported as
such and discarded.

Usage:
    python -m eval.fit_fusion_weights
    python -m eval.fit_fusion_weights --limit 100      # quick pass
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_DIR  # noqa: E402
from app.services.retrieval import ARMS, RRF_K, _verse_id_of  # noqa: E402
from eval.metrics import ndcg_at_k  # noqa: E402

BENCHMARK_DIR = DATA_DIR / "benchmark"
ARMS_CACHE = DATA_DIR / "eval" / "arm_lists.json"
OUT_FILE = DATA_DIR / "eval" / "fusion_weights.json"

#: Weight values tried per arm. 0.0 lets the optimiser switch an arm off.
GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
DEPTH = 10
FOLDS = 5

BOLD, DIM, GREEN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"
)


def fuse(arm_lists: list[tuple[str, list[str]]], weights: dict[str, float]) -> list[str]:
    """Weighted RRF, grouped to verses — the same arithmetic as retrieval."""
    scores: dict[str, float] = {}
    for arm, ranked in arm_lists:
        weight = weights.get(arm, 1.0)
        if weight == 0.0:
            continue
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (RRF_K + rank + 1)

    verse_scores: dict[str, float] = {}
    for doc_id, score in scores.items():
        verse_id = _verse_id_of(doc_id)
        if score > verse_scores.get(verse_id, float("-inf")):
            verse_scores[verse_id] = score

    return sorted(verse_scores, key=lambda v: verse_scores[v], reverse=True)[:DEPTH]


def evaluate(cache: dict, qrels: dict, ids: list[str], weights: dict) -> float:
    total = 0.0
    for query_id in ids:
        ranked = fuse(cache[query_id], weights)
        total += ndcg_at_k(ranked, qrels.get(query_id, {}), DEPTH)
    return total / max(1, len(ids))


def coordinate_ascent(cache, qrels, ids, arms, rounds=3) -> dict[str, float]:
    weights = {arm: 1.0 for arm in arms}
    best = evaluate(cache, qrels, ids, weights)
    for _ in range(rounds):
        improved = False
        for arm in arms:
            original = weights[arm]
            for value in GRID:
                if value == original:
                    continue
                weights[arm] = value
                score = evaluate(cache, qrels, ids, weights)
                if score > best + 1e-9:
                    best, original, improved = score, value, True
            weights[arm] = original
        if not improved:
            break
    return weights


async def build_cache(queries: list[dict], limit: int | None) -> dict:
    """Compute and cache each query's arm lists. The expensive part, done once."""
    from app.services import hyde, retrieval
    from app.services.pipeline import SERVED

    if ARMS_CACHE.exists():
        cached = json.loads(ARMS_CACHE.read_text(encoding="utf-8"))
        if len(cached) >= len(queries):
            print(f"{DIM}reusing cached arm lists for {len(cached)} queries{RESET}")
            return {k: [(a, d) for a, d in v] for k, v in cached.items()}

    cache: dict[str, list] = {}
    for index, item in enumerate(queries[:limit] if limit else queries, 1):
        query = item["query"]
        hyde_text, all_queries, _ = await hyde.transform(query)
        arms = await asyncio.to_thread(
            retrieval.arm_lists, hyde_text, all_queries, SERVED.retrieval
        )
        cache[item["query_id"]] = arms
        if index % 25 == 0:
            print(f"\r  arm lists {index}/{len(queries)}", end="", flush=True)
    print()

    ARMS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ARMS_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--folds", type=int, default=FOLDS)
    args = parser.parse_args()

    qrels = json.loads((BENCHMARK_DIR / "qrels.json").read_text(encoding="utf-8"))
    queries = json.loads((BENCHMARK_DIR / "queries.json").read_text(encoding="utf-8"))
    queries = [
        q for q in queries
        if any(g >= 2 for g in qrels.get(q["query_id"], {}).values())
    ]
    if args.limit:
        queries = queries[: args.limit]

    cache = await build_cache(queries, args.limit)
    ids = [q["query_id"] for q in queries if q["query_id"] in cache]
    present = sorted({arm for lists in cache.values() for arm, _ in lists})

    print(f"\n{BOLD}Fitting fusion weights{RESET}")
    print(f"  {len(ids)} queries · arms present: {', '.join(present)}")

    uniform = {arm: 1.0 for arm in present}
    baseline = evaluate(cache, qrels, ids, uniform)
    print(f"  uniform RRF (current):  nDCG@10 = {baseline:.4f}")

    # --- cross-validated estimate -----------------------------------------
    folds = [ids[i::args.folds] for i in range(args.folds)]
    held_out_fitted, held_out_uniform = [], []
    for index, test in enumerate(folds):
        train = [q for q in ids if q not in set(test)]
        fitted = coordinate_ascent(cache, qrels, train, present)
        held_out_fitted.append(evaluate(cache, qrels, test, fitted))
        held_out_uniform.append(evaluate(cache, qrels, test, uniform))

    cv_fitted = sum(held_out_fitted) / len(held_out_fitted)
    cv_uniform = sum(held_out_uniform) / len(held_out_uniform)
    gain = cv_fitted - cv_uniform

    print(f"\n{BOLD}{args.folds}-fold cross-validation (the honest number){RESET}")
    print(f"  uniform, held out:      {cv_uniform:.4f}")
    print(f"  fitted,  held out:      {cv_fitted:.4f}")
    print(f"  gain:                   {gain:+.4f}")

    # --- final fit on everything, for inspection --------------------------
    final = coordinate_ascent(cache, qrels, ids, present)
    in_sample = evaluate(cache, qrels, ids, final)
    print(f"\n{BOLD}Weights fitted on all data{RESET} "
          f"{DIM}(in-sample {in_sample:.4f} — expect optimism){RESET}")
    for arm in ARMS:
        if arm in final:
            bar = "#" * int(final[arm] * 8)
            print(f"  {arm:<18} {final[arm]:>5.2f}  {DIM}{bar}{RESET}")

    adopt = gain > 0
    print()
    if adopt:
        print(f"  {GREEN}ADOPT{RESET} — held-out gain is positive.")
    else:
        print(f"  {YELLOW}DO NOT ADOPT{RESET} — the fit does not generalise; "
              f"uniform RRF stands.")

    OUT_FILE.write_text(json.dumps({
        "arms": present,
        "uniform_ndcg": baseline,
        "cv_folds": args.folds,
        "cv_uniform_ndcg": cv_uniform,
        "cv_fitted_ndcg": cv_fitted,
        "cv_gain": gain,
        "adopt": adopt,
        "weights": final,
        "in_sample_ndcg": in_sample,
    }, indent=2), encoding="utf-8")
    print(f"{DIM}written -> {OUT_FILE}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
