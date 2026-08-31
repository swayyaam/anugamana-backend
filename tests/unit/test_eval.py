"""
The evaluation harness itself.

These are the tests that matter most for the research claim: a bug in a metric or
a significance test produces a wrong *result*, not a wrong *response*, and a
wrong result is far harder to notice and far more damaging.
"""

import math

import pytest

from eval import metrics, stats
from eval.conditions import BY_KEY, GRID, PLANNED_CONTRASTS
from eval.overlap import (
    ContaminationReport,
    assert_clean,
    build_idf,
    check_query,
    stratum_of,
    tokenize,
    weighted_overlap,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestGradedMetrics:
    def test_reciprocal_rank_finds_first_relevant(self):
        qrels = {"2.47": 3, "3.19": 2}
        assert metrics.reciprocal_rank(["2.47", "1.1"], qrels) == 1.0
        assert metrics.reciprocal_rank(["1.1", "2.47"], qrels) == 0.5
        assert metrics.reciprocal_rank(["1.1", "1.2"], qrels) == 0.0

    def test_grade_one_is_not_relevant_enough(self):
        """A tangential verse must not count as a hit."""
        assert metrics.reciprocal_rank(["9.9"], {"9.9": 1}) == 0.0
        assert metrics.reciprocal_rank(["9.9"], {"9.9": 2}) == 1.0

    def test_ndcg_rewards_grade_and_position(self):
        qrels = {"a": 3, "b": 1}
        best = metrics.ndcg_at_k(["a", "b"], qrels, k=2)
        worse = metrics.ndcg_at_k(["b", "a"], qrels, k=2)
        assert best == pytest.approx(1.0)
        assert worse < best

    def test_ndcg_uses_exponential_gain(self):
        """2^3-1 = 7 vs 2^1-1 = 1: a grade-3 verse is worth seven grade-1s."""
        assert metrics.gain(3) == 7.0
        assert metrics.gain(1) == 1.0
        assert metrics.gain(0) == 0.0

    def test_ndcg_is_bounded(self):
        qrels = {"a": 3, "b": 2, "c": 1}
        for ranking in (["a", "b", "c"], ["c", "b", "a"], ["x", "y", "z"]):
            assert 0.0 <= metrics.ndcg_at_k(ranking, qrels, k=3) <= 1.0

    def test_ndcg_of_empty_qrels_is_zero_not_nan(self):
        assert metrics.ndcg_at_k(["a"], {}, k=5) == 0.0

    def test_recall_counts_only_relevant_grades(self):
        qrels = {"a": 3, "b": 2, "c": 1}   # two relevant, one tangential
        assert metrics.recall_at_k(["a", "b"], qrels, k=5) == 1.0
        assert metrics.recall_at_k(["a"], qrels, k=5) == 0.5
        assert metrics.recall_at_k(["c"], qrels, k=5) == 0.0

    def test_average_precision_rewards_early_hits(self):
        qrels = {"a": 3, "b": 3}
        early = metrics.average_precision(["a", "b", "x"], qrels)
        late = metrics.average_precision(["x", "a", "b"], qrels)
        assert early == pytest.approx(1.0)
        assert late < early

    def test_score_query_returns_every_metric(self):
        scored = metrics.score_query(["a"], {"a": 3})
        assert set(scored) == set(metrics.METRICS)

    def test_aggregate_of_nothing_is_zero(self):
        assert all(v == 0.0 for v in metrics.aggregate([]).values())


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestSignificance:
    def test_identical_runs_are_not_significant(self):
        scores = [0.5, 0.3, 0.9, 0.2, 0.7] * 8
        comparison = stats.compare("mrr@10", "A", "B", scores, scores, iterations=2000)
        assert comparison.delta == 0.0
        assert comparison.p_value == 1.0
        assert not comparison.significant

    def test_consistent_improvement_is_significant(self):
        baseline = [0.2] * 40
        system = [0.6] * 40
        comparison = stats.compare("mrr@10", "A", "B", baseline, system, iterations=2000)
        assert comparison.delta == pytest.approx(0.4)
        assert comparison.p_value < 0.05
        assert comparison.wins == 40 and comparison.losses == 0

    def test_tiny_noisy_difference_on_ten_queries_is_not_significant(self):
        """
        The retracted evaluation drew a conclusion from exactly this shape:
        three point estimates on ten queries with no interval.
        """
        baseline = [0.9, 0.5, 1.0, 0.3, 0.8, 0.2, 1.0, 0.6, 0.4, 0.7]
        system = [0.8, 0.6, 1.0, 0.4, 0.7, 0.3, 0.9, 0.5, 0.5, 0.6]
        comparison = stats.compare("mrr@5", "base", "full", baseline, system,
                                   iterations=3000)
        assert not comparison.significant
        assert comparison.ci_low < 0 < comparison.ci_high

    def test_confidence_interval_brackets_the_delta(self):
        baseline = [0.1, 0.2, 0.3, 0.4, 0.5] * 6
        system = [0.3, 0.4, 0.5, 0.6, 0.7] * 6
        comparison = stats.compare("m", "A", "B", baseline, system, iterations=3000)
        assert comparison.ci_low <= comparison.delta <= comparison.ci_high

    def test_p_value_is_never_zero(self):
        """A permutation test cannot justify p = 0; add-one smoothing enforces it."""
        comparison = stats.compare(
            "m", "A", "B", [0.0] * 50, [1.0] * 50, iterations=1000
        )
        assert comparison.p_value > 0

    def test_results_are_deterministic_under_a_fixed_seed(self):
        baseline = [0.3, 0.7, 0.2, 0.9, 0.5] * 6
        system = [0.4, 0.6, 0.4, 0.8, 0.6] * 6
        first = stats.compare("m", "A", "B", baseline, system, iterations=1500, seed=7)
        second = stats.compare("m", "A", "B", baseline, system, iterations=1500, seed=7)
        assert first.p_value == second.p_value
        assert first.ci_low == second.ci_low

    def test_mismatched_run_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same queries"):
            stats.bootstrap_ci([0.1, 0.2], [0.1])

    def test_holm_is_monotonic_and_conservative(self):
        comparisons = [
            stats.Comparison("m", "A", f"B{i}", 0, 0, 0, 0, 0, p, 0, 0, 0, 30)
            for i, p in enumerate([0.001, 0.01, 0.04, 0.3])
        ]
        stats.holm_correction(comparisons)
        adjusted = [c.p_adjusted for c in sorted(comparisons, key=lambda c: c.p_value)]
        assert adjusted == sorted(adjusted), "adjusted p-values must not decrease"
        assert all(a >= c.p_value for a, c in zip(adjusted, sorted(comparisons, key=lambda c: c.p_value)))

    def test_holm_can_overturn_a_marginal_result(self):
        comparisons = [
            stats.Comparison("m", "A", f"B{i}", 0, 0, 0, 0, 0, 0.04, 0, 0, 0, 30)
            for i in range(4)
        ]
        stats.holm_correction(comparisons)
        assert not any(c.significant for c in comparisons)


# ---------------------------------------------------------------------------
# Contamination and stratification
# ---------------------------------------------------------------------------

class TestContaminationGate:
    def test_detects_verbatim_containment(self):
        """The exact failure mode that invalidated the 2026-05-10 benchmark."""
        texts = {"2.47": "stuck in a job you hate but too scared to quit; agonizing "
                         "over a decision you cannot undo and losing sleep over it"}
        report = check_query(
            "agonizing over a decision you cannot undo and losing sleep over it",
            texts, "2.47",
        )
        assert report.verbatim
        assert report.contaminated

    def test_short_generic_phrase_is_coincidence_not_leakage(self):
        """
        A handful of common words will appear somewhere in a 700-verse corpus by
        chance. Treating that as contamination would reject terse queries — the
        register real users actually type. Measured on the live benchmark: this
        rule was the difference between 388/389 and 389/389 clean.
        """
        texts = {"1.29": "my whole body is trembling and i am not able to know "
                         "whether i am doing the right thing here"}
        report = check_query("am i doing the right thing", texts, "1.29")
        assert report.verbatim, "the phrase really is present"
        assert report.query_tokens < 6
        assert not report.contaminated, "but it is not evidence of derivation"

    def test_detects_long_shared_phrase_without_full_containment(self):
        texts = {"2.47": "agonizing over a decision you cannot undo and losing sleep "
                         "over the outcome every single night"}
        report = check_query(
            "why am I agonizing over a decision you cannot undo and losing sleep "
            "over the outcome",
            texts, "2.47",
        )
        assert report.longest_shared_ngram >= 8
        assert report.contaminated

    def test_independent_query_is_clean(self):
        texts = {"2.47": "You have a right to perform your prescribed duties, but "
                         "you are not entitled to the fruits of action."}
        report = check_query(
            "my manager takes credit for everything I build and I am done trying",
            texts, "2.47",
        )
        assert not report.verbatim
        assert not report.contaminated

    def test_shared_topic_vocabulary_is_not_contamination(self):
        """Sharing subject words is normal retrieval, not leakage."""
        texts = {"2.47": "You have a right to perform your prescribed duties, but "
                         "you are not entitled to the fruits of action."}
        report = check_query("what are my duties and their fruits", texts, "2.47")
        assert not report.contaminated

    def test_assert_clean_raises_and_names_the_retraction(self):
        bad = [ContaminationReport("q", "2.47", True, 1.0, 12)]
        with pytest.raises(AssertionError, match="RETRACTION.md"):
            assert_clean(bad)

    def test_assert_clean_passes_on_clean_reports(self):
        assert_clean([ContaminationReport("q", "2.47", False, 0.0, 3)])

    def test_missing_verse_text_does_not_crash(self):
        assert not check_query("anything", {}, "9.99").contaminated


class TestStratification:
    def test_idf_ranks_rare_terms_above_common_ones(self):
        idf = build_idf([
            "duty and action", "duty and knowledge", "duty and devotion",
            "renunciation of ephemeral attachment",
        ])
        assert idf["renunciation"] > idf["duty"]

    def test_overlap_is_one_when_query_terms_all_appear(self):
        idf = build_idf(["duty action fruits", "soul eternal"])
        assert weighted_overlap("duty action", "duty action fruits", idf) == 1.0

    def test_overlap_is_zero_with_no_shared_content_words(self):
        idf = build_idf(["duty action fruits"])
        assert weighted_overlap("manager credit promotion", "duty action fruits", idf) == 0.0

    def test_stopwords_do_not_create_overlap(self):
        idf = build_idf(["the duty of the action"])
        assert weighted_overlap("the and of it", "the duty of the action", idf) == 0.0

    @pytest.mark.parametrize("value,expected", [
        (0.0, "none"), (0.04, "none"), (0.05, "low"),
        (0.19, "low"), (0.2, "medium"), (0.5, "high"), (1.0, "high"),
    ])
    def test_bins(self, value, expected):
        assert stratum_of(value) == expected

    def test_tokenizer_drops_stopwords_and_short_tokens(self):
        assert tokenize("The cat is on a mat") == ["cat", "mat"]


# ---------------------------------------------------------------------------
# Grid integrity
# ---------------------------------------------------------------------------

class TestGrid:
    def test_keys_are_unique(self):
        keys = [c.key for c in GRID]
        assert len(keys) == len(set(keys))

    def test_contains_the_two_dangerous_baselines(self):
        assert BY_KEY["P0"].kind == "parametric"
        assert BY_KEY["C0"].kind == "bm25"

    def test_contains_a_genuinely_unenriched_control(self):
        """The condition whose absence invalidated the original evaluation."""
        c2 = BY_KEY["C2"]
        assert c2.config.retrieval.index.name == "raw"
        assert "meaning" not in c2.config.retrieval.doc_types

    def test_enrichment_contrast_differs_by_exactly_one_factor(self):
        """C2 vs C5 must isolate the enrichment and nothing else."""
        c2, c5 = BY_KEY["C2"].config, BY_KEY["C5"].config
        for flag in ("use_hyde", "use_expansion", "use_cross_encoder", "use_mmr",
                     "use_emotion_arm", "use_transliteration"):
            assert getattr(c2, flag) == getattr(c5, flag), flag
        assert c2.retrieval.index.name == "raw"
        assert c5.retrieval.index.name == "enriched"

    def test_hyde_calibration_contrast_differs_only_in_calibration(self):
        c6, c7 = BY_KEY["C6"].config, BY_KEY["C7"].config
        assert c6.use_hyde and c7.use_hyde
        assert c6.hyde_calibrated is False and c7.hyde_calibrated is True
        assert c6.retrieval == c7.retrieval

    def test_c10_is_the_served_system(self):
        from app.services.pipeline import SERVED
        c10 = BY_KEY["C10"].config
        for flag in ("use_hyde", "use_expansion", "use_cross_encoder", "use_mmr"):
            assert getattr(c10, flag) == getattr(SERVED, flag), flag

    def test_grid_conditions_never_generate_or_gate(self):
        """Evaluation must measure retrieval, not the user-facing wrapper."""
        for condition in GRID:
            if condition.config is None:
                continue
            assert condition.config.use_generation is False
            assert condition.config.use_guardrail is False
            assert condition.config.use_safety is False

    def test_planned_contrasts_reference_real_conditions(self):
        for baseline, system, question in PLANNED_CONTRASTS:
            assert baseline in BY_KEY, baseline
            assert system in BY_KEY, system
            assert question.endswith("?")
