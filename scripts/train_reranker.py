#!/usr/bin/env python3
"""
Fine-tune a reranker on this corpus's own relevance judgments.

Why
---
69.6% of the served system's failures are ranking failures — the relevant verse
was retrieved and then buried. The obvious fix is a reranker, but the
off-the-shelf one measured ROC AUC 0.4579 on this corpus: worse than random, so
it was removed. The question this script answers is whether that is a property of
*that model* or of the *task*.

We are unusually well placed to answer it: `data/benchmark/qrels.json` holds
25,545 graded (query, verse) pairs, 4,189 of them relevant. That is a real
training set for exactly the thing that is failing.

Honest limitations, stated up front because they bound every number this produces
-------------------------------------------------------------------------------
1. The labels are **silver** — three Claude models, Krippendorff alpha 0.709. A
   reranker trained on them learns to agree with Claude, not necessarily to be
   right. The held-out test set is silver too, so a gain here means "agrees
   better with the annotators". That is genuinely useful and genuinely limited,
   and it must be revalidated once human judgments exist
   (docs/JUDGE_VALIDATION.md).
2. Splits are **by query, never by pair**. Splitting pairs would put the same
   query in train and test and inflate everything.
3. The test split is touched exactly once, at the end.

Usage:
    python scripts/train_reranker.py --dry-run          # data report only
    python scripts/train_reranker.py --epochs 2
    python scripts/train_reranker.py --base BAAI/bge-reranker-v2-m3
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_DIR, ENRICHED_FILE  # noqa: E402

BENCHMARK_DIR = DATA_DIR / "benchmark"
RUNS_DIR = DATA_DIR / "eval" / "runs"
OUTPUT_DIR = DATA_DIR / "models" / "reranker-gita"
SPLIT_FILE = DATA_DIR / "eval" / "reranker_splits.json"
REPORT_FILE = DATA_DIR / "eval" / "reranker_training.json"

DEFAULT_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SEED = 20260831

BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"
)


def load_data():
    qrels = json.loads((BENCHMARK_DIR / "qrels.json").read_text(encoding="utf-8"))
    queries = {
        q["query_id"]: q["query"]
        for q in json.loads((BENCHMARK_DIR / "queries.json").read_text())
    }
    verses = {v["verse_id"]: v for v in json.loads(ENRICHED_FILE.read_text())}
    return qrels, queries, verses


def split_by_query(query_ids: list[str], seed: int = SEED):
    """60/20/20 by query. Never by pair — that would leak."""
    shuffled = sorted(query_ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end, val_end = int(0.6 * n), int(0.8 * n)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def build_pairs(ids, qrels, queries, verses):
    """(query, verse text, label in [0,1]) — graded relevance as a soft target."""
    rows = []
    for qid in ids:
        for vid, grade in qrels.get(qid, {}).items():
            if vid not in verses:
                continue
            rows.append({
                "query": queries[qid],
                "answer": verses[vid]["translation"],
                "label": float(grade) / 3.0,
            })
    return rows


def evaluate_ranking(model, ids, qrels, queries, verses, candidates):
    """nDCG@10 by reordering the served system's own candidate lists."""
    from eval import metrics as M

    baseline, reranked = [], []
    for qid in ids:
        pool = [v for v in candidates.get(qid, []) if v in verses]
        if len(pool) < 2:
            continue
        baseline.append(M.ndcg_at_k(pool, qrels.get(qid, {}), 10))
        scores = model.predict(
            [(queries[qid], verses[v]["translation"]) for v in pool],
            show_progress_bar=False,
        )
        order = [v for _, v in sorted(zip(scores, pool), key=lambda x: -x[0])]
        reranked.append(M.ndcg_at_k(order, qrels.get(qid, {}), 10))
    return baseline, reranked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qrels, queries, verses = load_data()
    usable = [q for q in qrels if q in queries]
    train_ids, val_ids, test_ids = split_by_query(usable)

    train = build_pairs(train_ids, qrels, queries, verses)
    val = build_pairs(val_ids, qrels, queries, verses)
    test = build_pairs(test_ids, qrels, queries, verses)

    def positives(rows):
        return sum(1 for r in rows if r["label"] >= 2 / 3)

    print(f"\n{BOLD}Training data — split by query, never by pair{RESET}")
    for name, ids, rows in (
        ("train", train_ids, train), ("val", val_ids, val), ("test", test_ids, test)
    ):
        print(f"  {name:<6} {len(ids):>4} queries  {len(rows):>7,} pairs  "
              f"{positives(rows):>5,} relevant "
              f"({positives(rows) / max(1, len(rows)):.1%})")
    print(f"  {DIM}labels are silver (3 Claude annotators, alpha 0.709) — a model "
          f"trained here learns to agree with them{RESET}")

    if args.dry_run:
        return 0

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(
        {"train": train_ids, "val": val_ids, "test": test_ids, "seed": SEED},
        indent=2,
    ), encoding="utf-8")

    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n{BOLD}Fine-tuning{RESET} {args.base} on {device}")

    model = CrossEncoder(args.base, num_labels=1, device=device)

    # Baseline before training, on the held-out test queries only.
    run_path = RUNS_DIR / "C13.json"
    candidates = (
        json.loads(run_path.read_text(encoding="utf-8"))["results"]
        if run_path.exists() else {}
    )
    before_base, before_rerank = evaluate_ranking(
        model, test_ids, qrels, queries, verses, candidates
    )
    print(f"  before: RRF {sum(before_base) / len(before_base):.4f} · "
          f"base reranker {sum(before_rerank) / len(before_rerank):.4f}")

    trainer = CrossEncoderTrainer(
        model=model,
        args=CrossEncoderTrainingArguments(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=100,
            report_to=[],
            seed=SEED,
            use_mps_device=(device == "mps"),
        ),
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(val),
        loss=BinaryCrossEntropyLoss(model),
    )
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    print(f"  saved -> {OUTPUT_DIR}")

    # --- held-out evaluation, touched once --------------------------------
    from eval import stats

    after_base, after_rerank = evaluate_ranking(
        model, test_ids, qrels, queries, verses, candidates
    )
    comparison = stats.compare(
        "ndcg@10", "rrf", "finetuned", after_base, after_rerank, iterations=5000
    )

    print(f"\n{BOLD}Held-out test ({len(after_base)} queries, seen once){RESET}")
    print(f"  RRF order            {comparison.baseline_mean:.4f}")
    print(f"  base reranker        {sum(before_rerank) / len(before_rerank):.4f}")
    print(f"  fine-tuned reranker  {comparison.system_mean:.4f}")
    print(f"  delta vs RRF         {comparison.delta:+.4f} "
          f"95% CI [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}]")
    print(f"  p={comparison.p_value:.4f}  "
          f"W/L/T={comparison.wins}/{comparison.losses}/{comparison.ties}")

    verdict = (
        f"{GREEN}ADOPT — domain training fixes the reranker{RESET}"
        if comparison.significant and comparison.delta > 0
        else f"{YELLOW}DO NOT ADOPT — ranking is not recoverable this way{RESET}"
    )
    print(f"\n  {verdict}")

    REPORT_FILE.write_text(json.dumps({
        "base_model": args.base,
        "device": device,
        "epochs": args.epochs,
        "n_train_pairs": len(train), "n_test_queries": len(after_base),
        "rrf_ndcg": comparison.baseline_mean,
        "base_reranker_ndcg": sum(before_rerank) / len(before_rerank),
        "finetuned_ndcg": comparison.system_mean,
        "delta": comparison.delta,
        "ci_low": comparison.ci_low, "ci_high": comparison.ci_high,
        "p_value": comparison.p_value, "significant": comparison.significant,
        "wins": comparison.wins, "losses": comparison.losses,
        "ties": comparison.ties,
        "labels": "silver — 3 Claude annotators, Krippendorff alpha 0.709",
    }, indent=2), encoding="utf-8")
    print(f"{DIM}written -> {REPORT_FILE}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
