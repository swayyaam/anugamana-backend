"""
The pipeline. One parameterised implementation, two callers.

Before 2026-08-31 the API and the evaluation harness were separate code. The
harness skipped HyDE ("for evaluation speed"), the reranker, MMR, query expansion
and the purport collection — so the system that was measured was not the system
that shipped, and the mechanism the research is about had never been measured at
all (audit F-03).

Everything now runs through `run()`. An ablation condition is a `PipelineConfig`,
not a second implementation, so a condition cannot silently drift from production.
`SERVED` is the configuration the API uses; the grid in eval/conditions.py builds
the rest by toggling flags on the same dataclass.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace

import structlog

from app.config import MIN_RELEVANCE, LOW_CONFIDENCE_RELEVANCE
from app.services import guardrail, hyde, rag, reranker, retrieval, routing, safety
from app.services.retrieval import (
    DEFAULT_CONFIG as DEFAULT_RETRIEVAL,
    RetrievalConfig,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """One ablation condition, or the served system."""

    name: str = "served"
    description: str = ""

    retrieval: RetrievalConfig = DEFAULT_RETRIEVAL

    # pre-retrieval stages
    use_safety: bool = True
    use_guardrail: bool = True
    use_routing: bool = True

    # query transformation
    use_hyde: bool = True
    hyde_calibrated: bool = True
    use_expansion: bool = True

    # ranking
    use_cross_encoder: bool = True
    use_mmr: bool = True
    rerank_top_n: int = 5
    apply_confidence_filter: bool = True

    # output
    use_generation: bool = True
    top_k: int = 3

    def for_evaluation(self, depth: int = 10) -> "PipelineConfig":
        """Strip user-facing stages; keep pure retrieval quality, ranked to `depth`."""
        return replace(
            self,
            use_safety=False,
            use_guardrail=False,
            use_generation=False,
            apply_confidence_filter=False,
            rerank_top_n=depth,
            top_k=depth,
        )


#: The configuration the API serves. The grid measures this as condition C10.
SERVED = PipelineConfig(
    name="served",
    description="Full pipeline: enriched hybrid retrieval + HyDE + expansion "
                "+ cross-encoder + MMR",
)


@dataclass
class PipelineResult:
    verses: list[dict] = field(default_factory=list)
    guidances: list[str] = field(default_factory=list)
    route: str = "semantic"
    guardrail: str = "relevant"
    safety: str = "safe"
    score_type: str = "none"
    degraded: list[str] = field(default_factory=list)
    low_confidence: bool = False
    confidence_filtered: int = 0
    timings: dict[str, int] = field(default_factory=dict)

    @property
    def verse_ids(self) -> list[str]:
        return [v["verse_id"] for v in self.verses]


async def run(query: str, config: PipelineConfig = SERVED) -> PipelineResult:
    """Execute the pipeline. Never raises for pipeline-internal failures."""
    started = time.perf_counter()
    result = PipelineResult()

    def _elapsed_ms(since: float) -> int:
        return int((time.perf_counter() - since) * 1000)

    # --- 1. crisis routing -------------------------------------------------
    # Deliberately ahead of the topical guardrail, which is tuned to let
    # personal distress through (audit S-01).
    if config.use_safety:
        result.safety = await safety.classify(query)
        if result.safety == "crisis":
            result.timings["total_ms"] = _elapsed_ms(started)
            return result

    # --- 2. topical guardrail ---------------------------------------------
    if config.use_guardrail:
        result.guardrail = await guardrail.classify(query)
        if result.guardrail == "off_topic":
            result.timings["total_ms"] = _elapsed_ms(started)
            return result

    # --- 3. routing --------------------------------------------------------
    route_meta: dict = {}
    if config.use_routing:
        result.route, route_meta = routing.classify_query(query)

    # --- 4. direct lookup fast path ---------------------------------------
    if result.route == "direct_lookup":
        t0 = time.perf_counter()
        verses = await asyncio.to_thread(
            retrieval.retrieve_by_verse_id, route_meta["verse_id"], config.retrieval
        )
        result.timings["retrieval_ms"] = _elapsed_ms(t0)
        if verses:
            for verse in verses:
                verse["relevance"] = 1.0
            result.verses = verses[: config.top_k]
            result.score_type = "exact"
            return await _finish(query, result, config, started)
        logger.info("direct_lookup_miss", verse_id=route_meta["verse_id"])
        result.route = "semantic"  # fall through

    # --- 5. query transformation ------------------------------------------
    t0 = time.perf_counter()
    if result.route == "sanskrit":
        # Devanagari embeds well directly; a Prabhupada-style English
        # hypothetical would move it away from the query's own script.
        hyde_text, all_queries = query, [query]
    elif config.use_hyde or config.use_expansion:
        hyde_text, all_queries, degraded = await hyde.transform(
            query,
            use_hyde=config.use_hyde,
            use_expansion=config.use_expansion,
            calibrated=config.hyde_calibrated,
        )
        result.degraded.extend(degraded)
    else:
        hyde_text, all_queries = query, [query]
    result.timings["transform_ms"] = _elapsed_ms(t0)

    # --- 6. retrieval ------------------------------------------------------
    t0 = time.perf_counter()
    try:
        candidates = await asyncio.to_thread(
            retrieval.retrieve, hyde_text, all_queries, config.retrieval
        )
    except Exception as e:
        logger.error("retrieval_failed", error=str(e))
        result.degraded.append("retrieval")
        candidates = []
    result.timings["retrieval_ms"] = _elapsed_ms(t0)

    if not candidates:
        return await _finish(query, result, config, started)

    # --- 7. rerank + diversify --------------------------------------------
    t0 = time.perf_counter()
    ranked, degraded, score_type = await asyncio.to_thread(
        reranker.rerank,
        query,
        candidates,
        use_cross_encoder=config.use_cross_encoder,
        use_mmr=config.use_mmr,
        top_n=config.rerank_top_n,
    )
    result.degraded.extend(degraded)
    result.score_type = score_type
    result.timings["rerank_ms"] = _elapsed_ms(t0)

    ranked = ranked[: config.top_k]

    # --- 8. absolute confidence filtering ---------------------------------
    # Thresholding is only meaningful on calibrated cross-encoder probabilities.
    # Under RRF fallback the score is ordinal, so we never drop on it (audit E-02).
    if ranked and config.apply_confidence_filter and score_type == "cross_encoder":
        top_relevance = max(v["relevance"] for v in ranked)
        result.low_confidence = top_relevance < LOW_CONFIDENCE_RELEVANCE

        kept = [v for v in ranked if v["relevance"] >= MIN_RELEVANCE]
        if not kept:
            # Everything is weak. Return the single best, flagged, rather than
            # an empty result the user cannot interpret.
            kept = ranked[:1]
            result.low_confidence = True
        result.confidence_filtered = len(ranked) - len(kept)
        ranked = kept
    elif ranked:
        result.low_confidence = score_type != "cross_encoder"

    result.verses = ranked
    return await _finish(query, result, config, started)


async def _finish(
    query: str,
    result: PipelineResult,
    config: PipelineConfig,
    started: float,
) -> PipelineResult:
    if config.use_generation and result.verses:
        t0 = time.perf_counter()
        result.guidances = await rag.generate_batch(query, result.verses)
        result.timings["generation_ms"] = int((time.perf_counter() - t0) * 1000)
    else:
        result.guidances = [""] * len(result.verses)

    result.timings["total_ms"] = int((time.perf_counter() - started) * 1000)
    return result


async def retrieve_ranked(query: str, config: PipelineConfig, depth: int = 10) -> list[str]:
    """Convenience for the evaluation harness: ranked verse ids only."""
    result = await run(query, config.for_evaluation(depth))
    return result.verse_ids
