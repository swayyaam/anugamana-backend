"""
Score calibration and confidence filtering — regression tests for audit E-02.

The bug: scores were min-max normalised *within the result set*, which forced the
lowest-ranked result to exactly 0.0 — always below the drop threshold. A request
for three verses could never return three, and the top result was forced to 1.0
whether it was an excellent match or noise.
"""

import pytest

from app.services import reranker


class TestSigmoidCalibration:
    def test_maps_logits_to_probabilities(self):
        assert reranker._sigmoid(0.0) == pytest.approx(0.5)
        assert reranker._sigmoid(10.0) > 0.99
        assert reranker._sigmoid(-10.0) < 0.01

    def test_is_monotonic(self):
        values = [reranker._sigmoid(x) for x in (-11.0, -5.0, -2.0, 0.0, 3.0, 8.0)]
        assert values == sorted(values)

    def test_extreme_logits_do_not_overflow(self):
        assert 0.0 <= reranker._sigmoid(-1e9) <= 1.0
        assert 0.0 <= reranker._sigmoid(1e9) <= 1.0

    def test_score_is_absolute_not_positional(self):
        """
        The core of E-02: a set of uniformly weak candidates must produce
        uniformly weak scores. Under min-max normalisation the best of three
        terrible results was reported as 1.0.
        """
        weak = [reranker._sigmoid(x) for x in (-9.0, -9.5, -10.0)]
        assert all(score < 0.01 for score in weak)

        strong = [reranker._sigmoid(x) for x in (9.0, 8.9, 8.8)]
        assert all(score > 0.99 for score in strong)


class TestRerankContract:
    def test_reports_cross_encoder_score_type(self, verses, monkeypatch):
        monkeypatch.setattr(
            reranker, "_load_cross_encoder",
            lambda: type("CE", (), {"predict": lambda self, pairs: [3.0, 1.0, -2.0]})(),
        )
        monkeypatch.setattr(reranker, "_mmr", lambda c, n, l: c[:n])
        ranked, degraded, score_type = reranker.rerank("q", verses, top_n=3)
        assert score_type == "cross_encoder"
        assert degraded == []
        assert [v["verse_id"] for v in ranked] == ["2.47", "3.19", "18.48"]
        assert ranked[0]["relevance"] > ranked[-1]["relevance"]

    def test_falls_back_to_rrf_and_says_so(self, verses, monkeypatch):
        def boom():
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(reranker, "_load_cross_encoder", boom)
        monkeypatch.setattr(reranker, "_mmr", lambda c, n, l: c[:n])
        ranked, degraded, score_type = reranker.rerank("q", verses, top_n=3)
        assert score_type == "rrf"
        assert "reranker" in degraded
        # ordering preserved by rrf_score
        assert [v["verse_id"] for v in ranked] == ["2.47", "3.19", "18.48"]

    def test_disabled_cross_encoder_is_rrf(self, verses):
        _, _, score_type = reranker.rerank(
            "q", verses, use_cross_encoder=False, use_mmr=False, top_n=3
        )
        assert score_type == "rrf"

    def test_empty_input(self):
        assert reranker.rerank("q", []) == ([], [], "none")

    def test_mmr_failure_degrades_gracefully(self, verses, monkeypatch):
        monkeypatch.setattr(
            reranker, "_load_cross_encoder",
            lambda: type("CE", (), {"predict": lambda self, pairs: [3.0, 1.0, -2.0]})(),
        )

        def boom(*args, **kwargs):
            raise RuntimeError("no embedder")

        monkeypatch.setattr(reranker, "_mmr", boom)
        ranked, degraded, _ = reranker.rerank("q", verses, top_n=2)
        assert "mmr" in degraded
        assert len(ranked) == 2


class TestConfidenceFilterDoesNotEatResults:
    @pytest.mark.asyncio
    async def test_requested_count_is_returned(self, stub_llms, stub_retrieval):
        """A request for three verses returns three. This was impossible before."""
        from dataclasses import replace

        from app.services import pipeline

        config = replace(pipeline.SERVED, top_k=3, rerank_top_n=3)
        result = await pipeline.run("I keep failing at work", config)
        assert len(result.verses) == 3
        assert result.confidence_filtered == 0

    @pytest.mark.asyncio
    async def test_never_returns_empty_when_candidates_exist(
        self, stub_llms, stub_retrieval, monkeypatch
    ):
        """All-weak candidates yield the best one, flagged — not nothing."""
        from dataclasses import replace

        from app.services import pipeline

        # Above every achievable sigmoid, so nothing clears the bar. The
        # cross-encoder is enabled explicitly: SERVED no longer runs it (it
        # measured ROC AUC 0.4579), and the confidence filter only applies when
        # a calibrated scorer is active.
        monkeypatch.setattr(pipeline, "MIN_RELEVANCE", 0.9999)
        config = replace(pipeline.SERVED, top_k=3, rerank_top_n=3,
                         use_cross_encoder=True)
        result = await pipeline.run("something obscure", config)
        assert len(result.verses) == 1
        assert result.low_confidence is True

    @pytest.mark.asyncio
    async def test_weak_tail_is_dropped_but_strong_head_is_kept(
        self, stub_llms, stub_retrieval, monkeypatch
    ):
        """Partial filtering: a confident top result is not flagged low."""
        from dataclasses import replace

        from app.services import pipeline

        # Clears the top candidate (0.9933) but not the others.
        monkeypatch.setattr(pipeline, "MIN_RELEVANCE", 0.99)
        config = replace(pipeline.SERVED, top_k=3, rerank_top_n=3,
                         use_cross_encoder=True)
        result = await pipeline.run("I keep failing at work", config)
        assert len(result.verses) == 1
        assert result.confidence_filtered == 2
        assert result.low_confidence is False

    @pytest.mark.asyncio
    async def test_ordinal_scores_are_never_thresholded(
        self, stub_llms, stub_retrieval, monkeypatch
    ):
        """Under RRF fallback the score is a rank, so dropping on it is invalid."""
        from dataclasses import replace

        from app.services import pipeline

        monkeypatch.setattr(pipeline, "MIN_RELEVANCE", 0.99)
        config = replace(
            pipeline.SERVED, top_k=3, rerank_top_n=3, use_cross_encoder=False
        )
        result = await pipeline.run("I keep failing at work", config)
        assert result.score_type == "rrf"
        assert len(result.verses) == 3
        assert result.confidence_filtered == 0
        # No calibrated scorer is active, so there is no confidence signal.
        # Flagging every result would make the flag meaningless; score_type
        # already tells the client the value is ordinal.
        assert result.low_confidence is False
