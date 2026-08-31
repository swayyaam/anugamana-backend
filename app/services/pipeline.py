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

from app.config import (
    LOW_CONFIDENCE_RELEVANCE,
    MIN_RELEVANCE,
    MULTILINGUAL_STRATEGY,
    PIVOT_LANGUAGE,
)
from app.services import (
    emotion as emotion_service,
    guardrail,
    hyde,
    rag,
    reranker,
    retrieval,
    routing,
    safety,
)
from app.services import sarvam
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
    #: Let the cross-encoder decide order, not just calibrate scores. Measured
    #: as harmful on this corpus (RESULTS.md section 6), so the served system
    #: sets this False while keeping calibrated scores.
    cross_encoder_reorders: bool = True
    use_mmr: bool = True
    rerank_top_n: int = 5
    apply_confidence_filter: bool = True

    # indic / affective arms
    #: "translate" (Mayura -> English pivot), "direct" (embed the Indic query
    #: as-is), or "both" (fuse both forms). Measured as conditions L1-L3.
    multilingual_strategy: str = MULTILINGUAL_STRATEGY
    #: Romanised Sanskrit -> Devanagari as an extra lexical arm (condition C11).
    use_transliteration: bool = True
    #: Explicit emotion-matching arm in RRF fusion (condition C12).
    use_emotion_arm: bool = True
    #: Translate generated guidance back into the user's language.
    translate_response: bool = True

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
            translate_response=False,
            rerank_top_n=depth,
            top_k=depth,
        )


#: The configuration the API serves. The grid measures this as condition C10.
#: The configuration the API serves.
#:
#: Changed 2026-08-31 in response to our own measurements, in two steps.
#:
#: 1. The grid showed cross-encoder *reordering* costs 0.0379 nDCG@10
#:    (p < 0.001), and that the previously served configuration ranked below
#:    four simpler ones. So it was demoted to scoring only.
#: 2. eval/calibrate.py then showed its scores have ROC AUC 0.4579 on this
#:    corpus — worse than random. A score that cannot separate relevant from
#:    irrelevant is not a score, so the stage is off entirely.
#:
#: The emotion arm stays: measured +0.0203 nDCG@10, p = 0.0024.
#: Ordering is therefore RRF fusion, and `score` is reported as ordinal.
SERVED = PipelineConfig(
    name="served",
    description="Enriched hybrid retrieval + HyDE + expansion + emotion arm, "
                "ranked by RRF fusion",
    use_cross_encoder=False,
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
    language: str = PIVOT_LANGUAGE
    language_method: str = "default"
    translated_query: str | None = None
    emotion: dict | None = None

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

    # --- 2. language detection and pivot translation ----------------------
    # The corpus, the enrichment and the purports are all English, so retrieval
    # happens in English and everything downstream works on `working_query`.
    # Which cross-lingual strategy is correct is an empirical question, not an
    # assumption: conditions L1-L3 in the grid measure translate vs. direct vs.
    # fusing both.
    working_query = query
    t0 = time.perf_counter()
    result.language, result.language_method = await sarvam.identify_language(query)

    needs_pivot = (
        result.language != PIVOT_LANGUAGE
        and config.multilingual_strategy in ("translate", "both")
    )
    if needs_pivot:
        translated = await sarvam.translate(query, result.language, PIVOT_LANGUAGE)
        if translated and translated != query:
            result.translated_query = translated
            working_query = translated
        else:
            result.degraded.append("translation")
    if result.language != PIVOT_LANGUAGE:
        result.timings["language_ms"] = _elapsed_ms(t0)

    # --- 3. topical guardrail ---------------------------------------------
    if config.use_guardrail:
        result.guardrail = await guardrail.classify(working_query)
        if result.guardrail == "off_topic":
            result.timings["total_ms"] = _elapsed_ms(started)
            return result

    # --- 4. routing --------------------------------------------------------
    route_meta: dict = {}
    if config.use_routing:
        # Route on the *working* query, i.e. after any pivot translation.
        # Routing on the original made every Hindi query take the "sanskrit"
        # fast path simply because it contained Devanagari — skipping HyDE and
        # the emotion arm for exactly the users Indic support exists to serve.
        # The sanskrit path is for Sanskrit verse text, not for any Indic script.
        result.route, route_meta = routing.classify_query(working_query)

        # A citation in the original still counts: translation can mangle
        # "verse 2.47" into prose, so fall back to the untranslated form.
        if result.route != "direct_lookup" and working_query != query:
            original_route, original_meta = routing.classify_query(query)
            if original_route == "direct_lookup":
                result.route, route_meta = original_route, original_meta

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
            return await _finish(working_query, result, config, started)
        logger.info("direct_lookup_miss", verse_id=route_meta["verse_id"])
        result.route = "semantic"  # fall through

    # --- 5. query transformation and affective classification --------------
    # HyDE, expansion and emotion classification are independent — run them
    # concurrently rather than paying for three sequential round trips.
    t0 = time.perf_counter()

    async def _transform():
        if result.route == "sanskrit":
            # Devanagari embeds well directly; a Prabhupada-style English
            # hypothetical would move it away from the query's own script.
            return working_query, [working_query], []
        if config.use_hyde or config.use_expansion:
            return await hyde.transform(
                working_query,
                use_hyde=config.use_hyde,
                use_expansion=config.use_expansion,
                calibrated=config.hyde_calibrated,
            )
        return working_query, [working_query], []

    async def _emotion():
        if not config.use_emotion_arm or result.route == "sanskrit":
            return emotion_service.EmotionResult()
        return await emotion_service.classify(working_query)

    (hyde_text, all_queries, hyde_degraded), emotion_result = await asyncio.gather(
        _transform(), _emotion()
    )
    result.degraded.extend(hyde_degraded)
    if emotion_result.detected:
        result.emotion = emotion_result.as_dict()
    result.timings["transform_ms"] = _elapsed_ms(t0)

    # --- 6. extra retrieval arms ------------------------------------------
    extra_probes: list[str] = []
    if emotion_result.detected:
        extra_probes.append(emotion_result.probe_text())

    # Romanised Sanskrit ("karmanye vadhikaraste") shares almost no surface form
    # with the Devanagari and IAST in the index, so the lexical arm is blind to
    # it without a script conversion.
    if (
        config.use_transliteration
        and result.route != "sanskrit"
        and sarvam.looks_romanised_indic(working_query)
    ):
        devanagari = await sarvam.to_devanagari(working_query)
        if devanagari and devanagari != working_query:
            extra_probes.append(devanagari)

    # Under the "both" strategy the untranslated query is fused alongside its
    # translation instead of being discarded.
    if (
        config.multilingual_strategy == "both"
        and result.translated_query
        and query not in all_queries
    ):
        all_queries = all_queries + [query]

    # --- 7. retrieval ------------------------------------------------------
    t0 = time.perf_counter()
    try:
        candidates = await asyncio.to_thread(
            retrieval.retrieve, hyde_text, all_queries, config.retrieval, extra_probes
        )
    except Exception as e:
        logger.error("retrieval_failed", error=str(e))
        result.degraded.append("retrieval")
        candidates = []
    result.timings["retrieval_ms"] = _elapsed_ms(t0)

    if not candidates:
        return await _finish(working_query, result, config, started)

    # --- 7. rerank + diversify --------------------------------------------
    t0 = time.perf_counter()
    ranked, degraded, score_type = await asyncio.to_thread(
        reranker.rerank,
        working_query,
        candidates,
        use_cross_encoder=config.use_cross_encoder,
        cross_encoder_reorders=config.cross_encoder_reorders,
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
        # No calibrated scorer is active, so there is no confidence signal to
        # report. Flagging every result low-confidence would make the flag
        # meaningless; `score_type` already tells the client the score is
        # ordinal and must not be thresholded.
        result.low_confidence = False

    result.verses = ranked
    return await _finish(working_query, result, config, started)


async def _finish(
    query: str,
    result: PipelineResult,
    config: PipelineConfig,
    started: float,
) -> PipelineResult:
    """
    Generate guidance, then return it in the user's language.

    Generation happens in English because the verse translations and purports
    are English — generating directly in the target language would force the
    model to translate Prabhupada's commentary on the fly, which is exactly the
    step most likely to introduce unsupported claims into a religious text.
    Translating finished, grounded English output is the more faithful order.
    """
    if config.use_generation and result.verses:
        t0 = time.perf_counter()
        result.guidances = await rag.generate_batch(query, result.verses)
        result.timings["generation_ms"] = int((time.perf_counter() - t0) * 1000)
    else:
        result.guidances = [""] * len(result.verses)

    if (
        config.translate_response
        and result.language != PIVOT_LANGUAGE
        and any(result.guidances)
    ):
        t0 = time.perf_counter()
        translated = await asyncio.gather(
            *(
                sarvam.translate(text, PIVOT_LANGUAGE, result.language)
                if text else asyncio.sleep(0, result="")
                for text in result.guidances
            ),
            return_exceptions=True,
        )
        result.guidances = [
            original if isinstance(new, BaseException) else new
            for original, new in zip(result.guidances, translated)
        ]
        result.timings["response_translation_ms"] = int((time.perf_counter() - t0) * 1000)

    result.timings["total_ms"] = int((time.perf_counter() - started) * 1000)
    return result


async def retrieve_ranked(query: str, config: PipelineConfig, depth: int = 10) -> list[str]:
    """Convenience for the evaluation harness: ranked verse ids only."""
    result = await run(query, config.for_evaluation(depth))
    return result.verse_ids
