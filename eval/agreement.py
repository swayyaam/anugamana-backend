"""
Inter-annotator agreement for graded relevance judgments.

A benchmark without a reported agreement figure is an assertion, not a resource.
Two coefficients, because they answer different questions:

* **Krippendorff's alpha (ordinal)** — the right general-purpose coefficient
  here. It handles any number of annotators, tolerates missing judgments, and
  respects the ordering of the 0-3 scale: confusing 0 with 3 is a worse error
  than confusing 2 with 3, and an ordinal coefficient charges accordingly.
  Nominal kappa would treat those as equally wrong.

* **Quadratically-weighted Cohen's kappa** — reported per annotator pair,
  because it is the figure most readers of an IR paper will expect to see, and a
  per-pair number localises disagreement that a single pooled alpha hides.

Interpretation, following Krippendorff: alpha >= 0.800 supports firm conclusions,
0.667-0.800 supports tentative ones, below 0.667 means the judgments are too
noisy to carry a claim.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class AgreementReport:
    krippendorff_alpha: float
    n_units: int
    n_annotators: int
    pairwise_kappa: dict[str, float]
    label_distribution: dict[int, int]

    @property
    def interpretation(self) -> str:
        alpha = self.krippendorff_alpha
        if alpha >= 0.800:
            return "firm conclusions supportable"
        if alpha >= 0.667:
            return "tentative conclusions only"
        return "TOO NOISY — judgments cannot carry a claim"

    def format(self) -> str:
        lines = [
            f"Krippendorff's alpha (ordinal): {self.krippendorff_alpha:.4f}",
            f"  -> {self.interpretation}",
            f"  units: {self.n_units}   annotators: {self.n_annotators}",
            "  grade distribution: "
            + ", ".join(f"{g}:{c}" for g, c in sorted(self.label_distribution.items())),
        ]
        if self.pairwise_kappa:
            lines.append("  pairwise quadratic-weighted kappa:")
            for pair, value in sorted(self.pairwise_kappa.items()):
                lines.append(f"    {pair:<24} {value:.4f}")
        return "\n".join(lines)


def _ordinal_delta_squared(c: int, k: int, counts: dict[int, float], grades: list[int]) -> float:
    """
    Krippendorff's ordinal difference function.

    The distance between two grades depends on how much probability mass lies
    between them, so the scale's spacing is derived from the data rather than
    assumed uniform.
    """
    if c == k:
        return 0.0
    low, high = (c, k) if c < k else (k, c)
    between = sum(counts.get(g, 0.0) for g in grades if low <= g <= high)
    return (between - (counts.get(c, 0.0) + counts.get(k, 0.0)) / 2.0) ** 2


def krippendorff_alpha(units: list[list[int | None]]) -> float:
    """
    `units` is one list per item, holding each annotator's grade or None.

    Returns 1.0 for perfect agreement, 0.0 for chance, negative for systematic
    disagreement. Units with fewer than two judgments contribute nothing.
    """
    usable = [[v for v in unit if v is not None] for unit in units]
    usable = [unit for unit in usable if len(unit) >= 2]
    if not usable:
        return float("nan")

    grades = sorted({v for unit in usable for v in unit})
    if len(grades) < 2:
        return 1.0  # everyone used a single grade throughout

    # Coincidence matrix: every ordered pair of judgments within a unit,
    # weighted by 1/(m_u - 1) so units with more annotators do not dominate.
    coincidence: dict[tuple[int, int], float] = {}
    for unit in usable:
        m = len(unit)
        weight = 1.0 / (m - 1)
        for i, first in enumerate(unit):
            for j, second in enumerate(unit):
                if i != j:
                    key = (first, second)
                    coincidence[key] = coincidence.get(key, 0.0) + weight

    marginals: dict[int, float] = {}
    for (c, _), value in coincidence.items():
        marginals[c] = marginals.get(c, 0.0) + value
    total = sum(marginals.values())
    if total <= 1:
        return float("nan")

    observed = sum(
        value * _ordinal_delta_squared(c, k, marginals, grades)
        for (c, k), value in coincidence.items()
    )
    expected = sum(
        marginals[c] * marginals[k] * _ordinal_delta_squared(c, k, marginals, grades)
        for c in grades
        for k in grades
    ) / (total - 1)

    if expected == 0:
        return 1.0
    return 1.0 - (observed / expected)


def weighted_kappa(first: list[int], second: list[int], max_grade: int = 3) -> float:
    """Cohen's kappa with quadratic weights, for one annotator pair."""
    if len(first) != len(second) or not first:
        return float("nan")

    n = len(first)
    grades = list(range(max_grade + 1))

    observed = [[0.0] * len(grades) for _ in grades]
    for a, b in zip(first, second):
        observed[a][b] += 1.0 / n

    first_marginal = Counter(first)
    second_marginal = Counter(second)

    denominator = max_grade**2
    numerator_observed = 0.0
    numerator_expected = 0.0
    for a in grades:
        for b in grades:
            weight = ((a - b) ** 2) / denominator
            numerator_observed += weight * observed[a][b]
            numerator_expected += weight * (
                (first_marginal.get(a, 0) / n) * (second_marginal.get(b, 0) / n)
            )

    if numerator_expected == 0:
        return 1.0
    return 1.0 - (numerator_observed / numerator_expected)


def analyse(
    judgments: dict[str, dict[str, int]], max_grade: int = 3
) -> AgreementReport:
    """
    `judgments` maps annotator name -> {item_key: grade}.
    Items judged by fewer than two annotators are excluded from alpha.
    """
    annotators = sorted(judgments)
    all_items = sorted({item for grades in judgments.values() for item in grades})

    units = [
        [judgments[annotator].get(item) for annotator in annotators]
        for item in all_items
    ]

    pairwise: dict[str, float] = {}
    for i, first in enumerate(annotators):
        for second in annotators[i + 1 :]:
            shared = [
                item
                for item in all_items
                if item in judgments[first] and item in judgments[second]
            ]
            if shared:
                pairwise[f"{first} vs {second}"] = weighted_kappa(
                    [judgments[first][item] for item in shared],
                    [judgments[second][item] for item in shared],
                    max_grade,
                )

    distribution = Counter(
        grade for grades in judgments.values() for grade in grades.values()
    )

    return AgreementReport(
        krippendorff_alpha=krippendorff_alpha(units),
        n_units=len(all_items),
        n_annotators=len(annotators),
        pairwise_kappa=pairwise,
        label_distribution=dict(distribution),
    )
