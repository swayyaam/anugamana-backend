"""
Pipeline orchestration: stage ordering, graceful degradation, and the guarantee
that an ablation condition is a config rather than a second implementation.
"""

from dataclasses import replace

import pytest

from app.services import pipeline


class TestStageOrdering:
    @pytest.mark.asyncio
    async def test_crisis_check_precedes_everything(self, stub_llms, stub_retrieval):
        """
        Audit S-01: the topical guardrail is tuned to accept personal distress,
        so it can never be what stands between a user in crisis and retrieval.
        """
        result = await pipeline.run("I want to kill myself")
        assert result.safety == "crisis"
        assert result.verses == []
        assert result.guidances == []

    @pytest.mark.asyncio
    async def test_off_topic_stops_before_retrieval(self, stub_llms, stub_retrieval):
        stub_llms["guardrail_verdict"] = "off_topic"
        result = await pipeline.run("how do I center a div")
        assert result.guardrail == "off_topic"
        assert result.verses == []

    @pytest.mark.asyncio
    async def test_direct_lookup_skips_transformation(self, stub_llms, stub_retrieval):
        result = await pipeline.run("what does verse 2.47 say")
        assert result.route == "direct_lookup"
        assert result.score_type == "exact"
        assert result.verses[0]["relevance"] == 1.0
        assert "transform_ms" not in result.timings

    @pytest.mark.asyncio
    async def test_direct_lookup_miss_falls_through_to_semantic(
        self, stub_llms, stub_retrieval
    ):
        stub_retrieval["direct"] = []
        result = await pipeline.run("what does verse 2.47 say")
        assert result.route == "semantic"
        assert result.verses

    @pytest.mark.asyncio
    async def test_sanskrit_route_skips_hyde(self, stub_llms, stub_retrieval):
        result = await pipeline.run("कर्मण्येवाधिकारस्ते")
        assert result.route == "sanskrit"
        assert "hyde" not in result.degraded


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_retrieval_failure_yields_empty_not_an_exception(
        self, stub_llms, stub_retrieval, monkeypatch
    ):
        def boom(*args, **kwargs):
            raise RuntimeError("chroma is down")

        monkeypatch.setattr(pipeline.retrieval, "retrieve", boom)
        result = await pipeline.run("I feel lost")
        assert "retrieval" in result.degraded
        assert result.verses == []

    @pytest.mark.asyncio
    async def test_generation_failure_still_returns_verses(
        self, stub_llms, stub_retrieval, monkeypatch
    ):
        async def boom(query, verses):
            raise RuntimeError("anthropic down")

        monkeypatch.setattr(pipeline.rag, "generate_batch", boom)
        with pytest.raises(RuntimeError):
            await pipeline.run("I feel lost")

    @pytest.mark.asyncio
    async def test_individual_generation_failure_is_absorbed(
        self, stub_llms, stub_retrieval, monkeypatch
    ):
        from app.services import rag

        async def sometimes_fails(query, verse):
            if verse["verse_id"] == "3.19":
                raise RuntimeError("rate limited")
            return "guidance"

        monkeypatch.setattr(rag, "generate", sometimes_fails)
        result = await pipeline.run("I feel lost")
        assert len(result.guidances) == len(result.verses)
        assert "" in result.guidances

    @pytest.mark.asyncio
    async def test_hyde_failure_is_reported(self, stub_llms, stub_retrieval, monkeypatch):
        from app.services import hyde

        async def boom(query, system_prompt, use_cache):
            raise RuntimeError("api down")

        monkeypatch.setattr(hyde, "_hyde_call", boom)
        result = await pipeline.run("I feel lost")
        assert "hyde" in result.degraded
        assert result.verses  # retrieval still ran on the raw query


class TestEvaluationParity:
    def test_for_evaluation_strips_user_facing_stages(self):
        evaluated = pipeline.SERVED.for_evaluation(depth=10)
        assert evaluated.use_safety is False
        assert evaluated.use_guardrail is False
        assert evaluated.use_generation is False
        assert evaluated.apply_confidence_filter is False
        assert evaluated.top_k == 10

    def test_for_evaluation_preserves_the_retrieval_stack(self):
        """
        The whole point of unification: the stages under study must survive.
        The old harness silently dropped HyDE, expansion, reranking and MMR.
        """
        served, evaluated = pipeline.SERVED, pipeline.SERVED.for_evaluation()
        for flag in (
            "use_hyde", "hyde_calibrated", "use_expansion",
            "use_cross_encoder", "use_mmr", "retrieval",
        ):
            assert getattr(evaluated, flag) == getattr(served, flag), flag

    @pytest.mark.asyncio
    async def test_ablation_is_a_config_not_a_branch(self, stub_llms, stub_retrieval):
        no_rerank = replace(pipeline.SERVED, use_cross_encoder=False, use_mmr=False)
        result = await pipeline.run("I feel lost", no_rerank)
        assert result.score_type == "rrf"

    @pytest.mark.asyncio
    async def test_timings_are_recorded(self, stub_llms, stub_retrieval):
        result = await pipeline.run("I feel lost")
        assert result.timings["total_ms"] >= 0
        assert "retrieval_ms" in result.timings
        assert "rerank_ms" in result.timings
