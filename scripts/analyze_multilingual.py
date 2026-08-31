#!/usr/bin/env python3
"""
Cross-lingual results: what an Indic-language user actually gets.

Compares three strategies for a Hindi query against an English corpus, all
sharing the English relevance judgments because the information need is
identical and only its surface language changes:

    L1  translate to an English pivot (Mayura), then retrieve
    L2  embed the Hindi query directly (BGE-M3 is multilingual)
    L3  fuse both query forms

It also reports the *cross-lingual penalty*: the same system on the same
information needs, asked in English versus asked in Hindi.

Stated limitation: the Hindi queries were produced by translating English ones
with Mayura, so L1 translates back with the model that produced them and enjoys a
round-trip advantage a real Hindi speaker would not confer. L1 is an optimistic
bound; L2 carries no such advantage.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_DIR  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval import stats  # noqa: E402

BENCHMARK_DIR = DATA_DIR / "benchmark"
RUNS_DIR = DATA_DIR / "eval" / "runs"
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    hi_path = BENCHMARK_DIR / "queries_hi.json"
    if not hi_path.exists():
        print("No Hindi query set — run scripts/translate_benchmark.py")
        return 2

    qrels = load(BENCHMARK_DIR / "qrels.json")
    hindi = load(hi_path)

    ids = [
        q["query_id"] for q in hindi
        if any(g >= M.RELEVANT_THRESHOLD for g in qrels.get(q["query_id"], {}).values())
    ]
    print(f"\n{BOLD}Cross-lingual retrieval — Hindi queries{RESET}")
    print(f"{DIM}{len(ids)} queries with at least one relevant verse; English "
          f"judgments reused because the information need is unchanged{RESET}\n")

    runs = {}
    for path in RUNS_DIR.glob("*.json"):
        payload = load(path)
        runs[path.stem] = payload["results"]

    def scores_for(run_key, metric="ndcg@10"):
        results = runs.get(run_key, {})
        fn, k = M.METRICS[metric]
        return [fn(results.get(qid, []), qrels.get(qid, {}), k) for qid in ids]

    print(f"{'run':<10} {'language':<10} {'ndcg@10':>9} {'mrr@10':>9} {'r@10':>8}  strategy")
    print("-" * 78)
    rows = [
        ("L1_hi", "Hindi", "translate to English pivot, then retrieve"),
        ("L2_hi", "Hindi", "embed the Hindi query directly"),
        ("L3_hi", "Hindi", "fuse both query forms"),
        ("C13", "English", "same needs, asked in English (reference)"),
        ("C10", "English", "previous served system, English"),
    ]
    available = []
    for key, language, strategy in rows:
        if key not in runs:
            print(f"{key:<10} {DIM}not run{RESET}")
            continue
        available.append(key)
        n = [scores_for(key, m) for m in ("ndcg@10", "mrr@10", "recall@10")]
        means = [sum(v) / len(v) for v in n]
        print(f"{key:<10} {language:<10} {means[0]:>9.4f} {means[1]:>9.4f} "
              f"{means[2]:>8.4f}  {DIM}{strategy}{RESET}")

    print(f"\n{BOLD}Contrasts{RESET}")
    contrasts = [
        ("L2_hi", "L1_hi", "Is translate-then-retrieve better than direct multilingual?"),
        ("L1_hi", "L3_hi", "Does fusing both query forms beat translating alone?"),
        ("L1_hi", "C13", "What does asking in Hindi cost, versus English?"),
    ]
    comparisons, questions = [], {}
    for baseline, system, question in contrasts:
        if baseline not in available or system not in available:
            continue
        comparison = stats.compare(
            "ndcg@10", baseline, system,
            scores_for(baseline), scores_for(system), iterations=10_000,
        )
        comparisons.append(comparison)
        questions[(baseline, system)] = question
    stats.holm_correction(comparisons)

    for comparison in comparisons:
        verdict = ("significant" if comparison.significant else "not shown")
        print(f"\n  {questions[(comparison.baseline, comparison.system)]}")
        print(f"    {comparison.baseline} -> {comparison.system}: "
              f"{comparison.baseline_mean:.4f} -> {comparison.system_mean:.4f}  "
              f"delta={comparison.delta:+.4f} "
              f"95% CI [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}]")
        print(f"    p={comparison.p_value:.4f} (Holm {comparison.p_adjusted:.4f}) "
              f"-> {verdict}   W/L/T={comparison.wins}/{comparison.losses}/{comparison.ties}")

    out = DATA_DIR / "eval" / "results_multilingual.json"
    out.write_text(json.dumps({
        "n_queries": len(ids),
        "conditions": {
            key: {m: sum(scores_for(key, m)) / len(ids) for m in M.METRICS}
            for key in available
        },
        "contrasts": [
            {
                "question": questions[(c.baseline, c.system)],
                "baseline": c.baseline, "system": c.system,
                "delta": c.delta, "ci_low": c.ci_low, "ci_high": c.ci_high,
                "p_value": c.p_value, "p_holm": c.p_adjusted,
                "significant": c.significant,
            } for c in comparisons
        ],
    }, indent=2), encoding="utf-8")
    print(f"\n{DIM}written -> {out}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
