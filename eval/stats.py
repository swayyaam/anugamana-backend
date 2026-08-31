"""
Significance testing for paired retrieval runs.

The retracted evaluation reported three point estimates on ten queries with no
intervals and no test, then drew a conclusion from their ordering. Differences of
that size on that sample are indistinguishable from noise.

What is implemented here is the standard apparatus for IR comparisons:

* **Paired bootstrap** over queries. Conditions are run on the same query set, so
  the pairing must be preserved when resampling — resampling conditions
  independently discards the variance reduction that makes small IR effects
  detectable at all.
* **Two-sided randomisation (permutation) test.** Makes no distributional
  assumption, which matters because per-query metric values are bounded, highly
  skewed, and often zero-inflated.
* **Holm-Bonferroni correction.** A twelve-condition grid produces many pairwise
  comparisons; uncorrected p-values across that family will manufacture a
  "significant" result from noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

DEFAULT_ITERATIONS = 10_000
DEFAULT_SEED = 20260831


@dataclass
class Comparison:
    metric: str
    baseline: str
    system: str
    baseline_mean: float
    system_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    wins: int
    losses: int
    ties: int
    n: int
    p_adjusted: float | None = None

    @property
    def significant(self) -> bool:
        alpha = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return alpha < 0.05

    def format(self) -> str:
        star = "*" if self.significant else " "
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return (
            f"{self.system:<10} {self.system_mean:.4f}  "
            f"Δ={self.delta:+.4f} [{self.ci_low:+.4f}, {self.ci_high:+.4f}]  "
            f"p={p:.4f}{star}  W/L/T={self.wins}/{self.losses}/{self.ties}"
        )


def bootstrap_ci(
    baseline: list[float],
    system: list[float],
    iterations: int = DEFAULT_ITERATIONS,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile CI for the paired mean difference."""
    if len(baseline) != len(system):
        raise ValueError("paired runs must cover the same queries")
    n = len(baseline)
    if n == 0:
        return (0.0, 0.0)

    rng = random.Random(seed)
    deltas = [s - b for b, s in zip(baseline, system)]
    means = []
    for _ in range(iterations):
        # Resample query indices, keeping each query's paired difference intact.
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    tail = (1.0 - confidence) / 2.0
    return (
        means[int(tail * iterations)],
        means[min(int((1.0 - tail) * iterations), iterations - 1)],
    )


def randomization_test(
    baseline: list[float],
    system: list[float],
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> float:
    """
    Two-sided paired permutation test. Under the null, swapping a query's two
    scores is equally likely, so we flip the sign of each paired difference at
    random and count how often we see a mean difference at least as extreme.
    """
    if len(baseline) != len(system):
        raise ValueError("paired runs must cover the same queries")
    n = len(baseline)
    if n == 0:
        return 1.0

    rng = random.Random(seed)
    deltas = [s - b for b, s in zip(baseline, system)]
    observed = abs(sum(deltas) / n)
    if observed == 0.0:
        return 1.0

    at_least_as_extreme = 0
    for _ in range(iterations):
        total = sum(d if rng.random() < 0.5 else -d for d in deltas)
        if abs(total / n) >= observed - 1e-12:
            at_least_as_extreme += 1

    # Add-one smoothing: a permutation test can never justify p = 0.
    return (at_least_as_extreme + 1) / (iterations + 1)


def compare(
    metric: str,
    baseline_name: str,
    system_name: str,
    baseline: list[float],
    system: list[float],
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> Comparison:
    n = len(baseline)
    ci_low, ci_high = bootstrap_ci(baseline, system, iterations, seed=seed)
    return Comparison(
        metric=metric,
        baseline=baseline_name,
        system=system_name,
        baseline_mean=sum(baseline) / n if n else 0.0,
        system_mean=sum(system) / n if n else 0.0,
        delta=(sum(system) - sum(baseline)) / n if n else 0.0,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=randomization_test(baseline, system, iterations, seed=seed),
        wins=sum(1 for b, s in zip(baseline, system) if s > b),
        losses=sum(1 for b, s in zip(baseline, system) if s < b),
        ties=sum(1 for b, s in zip(baseline, system) if s == b),
        n=n,
    )


def holm_correction(comparisons: list[Comparison]) -> list[Comparison]:
    """
    Holm-Bonferroni, applied across a family of comparisons. Uniformly more
    powerful than plain Bonferroni and makes no independence assumption.
    Mutates and returns the same objects, ordered as given.
    """
    ordered = sorted(comparisons, key=lambda c: c.p_value)
    total = len(ordered)
    previous = 0.0
    for index, comparison in enumerate(ordered):
        adjusted = min(1.0, (total - index) * comparison.p_value)
        # Enforce monotonicity: an adjusted p may never decrease down the list.
        adjusted = max(adjusted, previous)
        comparison.p_adjusted = adjusted
        previous = adjusted
    return comparisons
