#!/usr/bin/env python3
"""
Fit the confidence thresholds on graded data instead of guessing them.

`config.MIN_RELEVANCE` and `config.LOW_CONFIDENCE_RELEVANCE` were left at
provisional values with an explicit note that they must be fitted before
anything rests on them. Audit defect E-02 was caused by exactly this kind of
unfitted threshold silently deleting results, so the fix is not to pick a nicer
number by intuition — it is to measure.

What this does
--------------
Takes judged (query, verse) pairs from the benchmark, scores each with the same
cross-encoder the pipeline uses, and asks how well that score separates relevant
verses (grade >= 2) from irrelevant ones. Then reports:

  * ROC AUC — whether the score carries usable signal at all
  * the threshold maximising F1, as a candidate MIN_RELEVANCE
  * the threshold at high precision, as a candidate LOW_CONFIDENCE_RELEVANCE
  * the score distributions, so a degenerate separation is visible rather than
    hidden behind a single number

If AUC is near 0.5 the honest conclusion is that this cross-encoder cannot
support a confidence threshold on this corpus, and the flag should stay off.

Usage:
    python -m eval.calibrate
    python -m eval.calibrate --sample 4000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_DIR, ENRICHED_FILE  # noqa: E402
from app.services.reranker import _load_cross_encoder, _sigmoid  # noqa: E402
from eval.metrics import RELEVANT_THRESHOLD  # noqa: E402

BENCHMARK_DIR = DATA_DIR / "benchmark"
OUT_FILE = DATA_DIR / "eval" / "calibration.json"


def roc_auc(positives: list[float], negatives: list[float]) -> float:
    """
    Probability that a random relevant verse outscores a random irrelevant one.
    Computed by rank sum (Mann-Whitney U), so ties are handled correctly.
    """
    if not positives or not negatives:
        return float("nan")
    labelled = [(s, 1) for s in positives] + [(s, 0) for s in negatives]
    labelled.sort(key=lambda x: x[0])

    ranks, index = {}, 0
    while index < len(labelled):
        stop = index
        while stop + 1 < len(labelled) and labelled[stop + 1][0] == labelled[index][0]:
            stop += 1
        average_rank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = average_rank
        index = stop + 1

    positive_rank_sum = sum(
        ranks[position] for position, (_, label) in enumerate(labelled) if label == 1
    )
    n_pos, n_neg = len(positives), len(negatives)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def sweep(positives: list[float], negatives: list[float]) -> list[dict]:
    rows = []
    candidates = sorted({round(s, 4) for s in positives + negatives})
    # Evaluate at most 400 thresholds; more adds nothing but runtime.
    step = max(1, len(candidates) // 400)
    for threshold in candidates[::step]:
        true_pos = sum(1 for s in positives if s >= threshold)
        false_pos = sum(1 for s in negatives if s >= threshold)
        false_neg = len(positives) - true_pos
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        rows.append({
            "threshold": threshold, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--model", default=None,
                        help="cross-encoder to calibrate (default: config)")
    args = parser.parse_args()

    qrels_path = BENCHMARK_DIR / "qrels.json"
    if not qrels_path.exists():
        print("No qrels — run scripts/pool_and_judge.py first")
        return 2

    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    queries = {
        q["query_id"]: q["query"]
        for q in json.loads((BENCHMARK_DIR / "queries.json").read_text())
    }
    verses = {v["verse_id"]: v for v in json.loads(ENRICHED_FILE.read_text())}

    pairs = [
        (queries[qid], verses[vid]["translation"], int(grade))
        for qid, grades in qrels.items()
        if qid in queries
        for vid, grade in grades.items()
        if vid in verses
    ]
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.sample]
    print(f"scoring {len(pairs)} judged pairs with "
          f"{args.model or 'the configured cross-encoder'}...")

    model = _load_cross_encoder(args.model) if args.model else _load_cross_encoder()
    logits = model.predict([(q, t) for q, t, _ in pairs])
    scores = [_sigmoid(float(x)) for x in logits]

    positives = [s for s, (_, _, g) in zip(scores, pairs) if g >= RELEVANT_THRESHOLD]
    negatives = [s for s, (_, _, g) in zip(scores, pairs) if g < RELEVANT_THRESHOLD]

    auc = roc_auc(positives, negatives)
    rows = sweep(positives, negatives)
    best_f1 = max(rows, key=lambda r: r["f1"]) if rows else None
    high_precision = next(
        (r for r in sorted(rows, key=lambda r: -r["threshold"]) if r["precision"] >= 0.5),
        None,
    )

    def describe(values: list[float], label: str) -> None:
        if not values:
            print(f"  {label}: none")
            return
        ordered = sorted(values)
        pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        print(f"  {label:<14} n={len(values):<6} "
              f"p10={pick(0.10):.4f} median={pick(0.50):.4f} p90={pick(0.90):.4f} "
              f"mean={sum(values) / len(values):.4f}")

    print(f"\nCross-encoder score distribution (calibrated probability)")
    describe(positives, "relevant")
    describe(negatives, "not relevant")

    print(f"\nROC AUC: {auc:.4f}")
    if auc < 0.60:
        verdict = ("WEAK — this cross-encoder barely separates relevant from "
                   "irrelevant on this corpus. Do not threshold on it.")
    elif auc < 0.75:
        verdict = "MODERATE — usable for flagging, not for dropping results."
    else:
        verdict = "GOOD — a drop threshold is defensible."
    print(f"  {verdict}")

    if best_f1:
        print(f"\nBest-F1 threshold:      {best_f1['threshold']:.4f}  "
              f"(P={best_f1['precision']:.3f} R={best_f1['recall']:.3f} "
              f"F1={best_f1['f1']:.3f})")
    if high_precision:
        print(f"Precision>=0.5 at:      {high_precision['threshold']:.4f}  "
              f"(R={high_precision['recall']:.3f})")

    recommendation = {
        "MIN_RELEVANCE": 0.0,
        "LOW_CONFIDENCE_RELEVANCE": (
            round(best_f1["threshold"], 4) if best_f1 else 0.0
        ),
    }
    if auc >= 0.75 and best_f1:
        recommendation["MIN_RELEVANCE"] = round(best_f1["threshold"] / 2, 4)

    print(f"\nRecommended config values:")
    for key, value in recommendation.items():
        print(f"  {key} = {value}")
    if recommendation["MIN_RELEVANCE"] == 0.0:
        print("  (MIN_RELEVANCE stays 0.0 — dropping results on a score this "
              "weak is what caused audit defect E-02.)")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "model": args.model or "config default",
        "n_pairs": len(pairs),
        "n_relevant": len(positives),
        "n_not_relevant": len(negatives),
        "roc_auc": auc,
        "verdict": verdict,
        "best_f1": best_f1,
        "recommendation": recommendation,
        "sweep": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten -> {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
