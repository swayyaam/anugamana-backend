"""
The ablation grid.

Each condition differs from the one above it by exactly one factor, so a
difference in score is attributable. Every condition is a `PipelineConfig` — the
same dataclass the API serves — so a condition cannot drift from production the
way the retracted harness did.

Two conditions are deliberately dangerous to the project's thesis and are run
first, because they determine whether the paper survives at all:

  P0  the LLM answering from parametric memory, with no retrieval. The Gita is
      in every frontier model's training data. If Claude can name the verse as
      well as the pipeline can retrieve it, the framing has to change — and it
      is far better to learn that here than from a reviewer.
  C0  a real BM25 implementation over raw translations. If a 1994 algorithm over
      untouched text is within noise of the full pipeline, the enrichment is not
      earning its cost.

`requires` names the artifact a condition needs. The runner reports skipped
conditions explicitly — a grid that silently drops the conditions it cannot run
is how "we measured everything" becomes false.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.services.pipeline import PipelineConfig
from app.services.retrieval import (
    ENRICHED_INDEX,
    MEDITATIONS_INDEX,
    MEDITATIONS_RAW_INDEX,
    RAW_INDEX,
    RetrievalConfig,
)


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    isolates: str
    #: None for conditions the runner implements specially (BM25, parametric).
    config: PipelineConfig | None = None
    kind: str = "pipeline"          # pipeline | bm25 | parametric
    requires: str = "enriched_index"


def _pipeline(
    key: str,
    label: str,
    isolates: str,
    retrieval: RetrievalConfig,
    requires: str = "enriched_index",
    **flags,
) -> Condition:
    base = dict(
        use_safety=False,
        use_guardrail=False,
        use_generation=False,
        apply_confidence_filter=False,
        translate_response=False,
        use_routing=False,          # the grid measures retrieval, not fast paths
        use_hyde=False,
        use_expansion=False,
        use_cross_encoder=False,
        use_mmr=False,
        use_emotion_arm=False,
        use_transliteration=False,
        multilingual_strategy="direct",
        rerank_top_n=10,
        top_k=10,
    )
    base.update(flags)
    return Condition(
        key=key,
        label=label,
        isolates=isolates,
        config=PipelineConfig(name=key, description=label, retrieval=retrieval, **base),
        requires=requires,
    )


# --- retrieval configurations ---------------------------------------------

RAW_TRANSLATION = RetrievalConfig(
    index=RAW_INDEX, dense=True, sparse=False,
    doc_types=("translation",), use_purport=False,
)
RAW_TRANSLATION_PURPORT = replace(RAW_TRANSLATION, use_purport=True)
RAW_LEXICAL = replace(RAW_TRANSLATION, dense=False, sparse=True, use_purport=True)

ENRICHED_MEANING = RetrievalConfig(
    index=ENRICHED_INDEX, dense=True, sparse=False,
    doc_types=("meaning",), use_purport=False,
)
ENRICHED_FULL_DENSE = RetrievalConfig(
    index=ENRICHED_INDEX, dense=True, sparse=False,
    doc_types=("meaning", "translation"), use_purport=True,
)
ENRICHED_HYBRID = RetrievalConfig(
    index=ENRICHED_INDEX, dense=True, sparse=True,
    doc_types=("meaning", "translation"), use_purport=True,
)


# --- the grid --------------------------------------------------------------

GRID: list[Condition] = [
    # ---- baselines that could sink the project ----------------------------
    Condition(
        key="P0",
        label="LLM from parametric memory, no retrieval",
        isolates="whether retrieval is necessary at all",
        kind="parametric",
        requires="anthropic_api",
    ),
    Condition(
        key="C0",
        label="BM25 over raw translations",
        isolates="the true lexical floor",
        kind="bm25",
        requires="corpus",
    ),
    Condition(
        key="C1",
        label="BM25 over raw translations + purports",
        isolates="value of commentary text to a lexical system",
        kind="bm25",
        requires="corpus",
    ),

    # ---- dense, unenriched: the control the retracted eval lacked ---------
    _pipeline("C2", "Dense over raw translations",
              "semantic floor without enrichment",
              RAW_TRANSLATION, requires="raw_index"),
    _pipeline("C3", "Dense over raw translations + purport chunks",
              "value of parent-child purport indexing",
              RAW_TRANSLATION_PURPORT, requires="raw_index"),
    _pipeline("C3b", "Hybrid over raw text",
              "sparse+dense fusion without enrichment",
              RAW_LEXICAL, requires="raw_index"),

    # ---- the contribution -------------------------------------------------
    _pipeline("C5", "Dense over chapter-aware enrichment",
              "THE ENRICHMENT ITSELF (vs C2)",
              ENRICHED_MEANING),
    _pipeline("C5b", "Dense over enrichment + translation + purport",
              "enrichment combined with the raw text",
              ENRICHED_FULL_DENSE),

    # ---- query-side transformation ---------------------------------------
    _pipeline("C6", "C5b + generic-prompt HyDE",
              "HyDE as a mechanism, uncalibrated",
              ENRICHED_FULL_DENSE, use_hyde=True, hyde_calibrated=False),
    _pipeline("C7", "C5b + domain-calibrated HyDE",
              "value of Prabhupada-style calibration (vs C6)",
              ENRICHED_FULL_DENSE, use_hyde=True, hyde_calibrated=True),
    _pipeline("C8", "C7 + query expansion + sparse arm (hybrid RRF)",
              "multi-arm fusion",
              ENRICHED_HYBRID, use_hyde=True, use_expansion=True),

    # ---- ranking ----------------------------------------------------------
    _pipeline("C9", "C8 + cross-encoder rerank",
              "whether an out-of-domain reranker helps or hurts",
              ENRICHED_HYBRID, use_hyde=True, use_expansion=True,
              use_cross_encoder=True),
    _pipeline("C10", "C9 + MMR diversity — THE SERVED SYSTEM",
              "the system users actually get",
              ENRICHED_HYBRID, use_hyde=True, use_expansion=True,
              use_cross_encoder=True, use_mmr=True),

    # ---- additional arms --------------------------------------------------
    _pipeline("C12", "C10 + explicit emotion-matching arm",
              "affective alignment as a retrieval signal",
              ENRICHED_HYBRID, use_hyde=True, use_expansion=True,
              use_cross_encoder=True, use_mmr=True, use_emotion_arm=True,
              requires="anthropic_api"),

    # ---- the served system, after acting on the measurements --------------
    # C10 was the served system until the grid showed the cross-encoder hurts
    # ranking and eval/calibrate.py showed its scores have AUC 0.4579. This is
    # the configuration that replaced it, and it is measured like any other so
    # the "evaluated system is the served system" property still holds.
    _pipeline("C13", "C8 + emotion arm, no cross-encoder — THE SERVED SYSTEM",
              "the configuration the evidence supports",
              ENRICHED_HYBRID, use_hyde=True, use_expansion=True,
              use_emotion_arm=True, requires="anthropic_api"),
]

#: Cross-lingual strategies. Run only over the Indic query subset, and only when
#: Sarvam is configured — otherwise translation silently returns the input and
#: L1 becomes an unlabelled duplicate of L2.
MULTILINGUAL_GRID: list[Condition] = [
    _pipeline("L1", "Indic query -> English pivot (Mayura), then C10",
              "translate-then-retrieve",
              ENRICHED_HYBRID, use_hyde=True, use_expansion=True,
              use_cross_encoder=True, use_mmr=True,
              multilingual_strategy="translate", requires="sarvam"),
    _pipeline("L2", "Indic query embedded directly (BGE-M3 multilingual)",
              "direct multilingual retrieval",
              ENRICHED_HYBRID, use_hyde=True, use_expansion=True,
              use_cross_encoder=True, use_mmr=True,
              multilingual_strategy="direct", requires="sarvam"),
    _pipeline("L3", "Both query forms fused",
              "whether fusing beats choosing",
              ENRICHED_HYBRID, use_hyde=True, use_expansion=True,
              use_cross_encoder=True, use_mmr=True,
              multilingual_strategy="both", requires="sarvam"),
]

#: Replication on a second corpus. A result on one corpus is a case study; the
#: same result on Meditations — different tradition, different source language,
#: public domain — is a method. M2 vs M5 is the identical single-factor contrast
#: as C2 vs C5, so the two are directly comparable.
MEDITATIONS_RAW = RetrievalConfig(
    index=MEDITATIONS_RAW_INDEX, dense=True, sparse=False,
    doc_types=("translation",), use_purport=False,
)
MEDITATIONS_ENRICHED = RetrievalConfig(
    index=MEDITATIONS_INDEX, dense=True, sparse=False,
    doc_types=("meaning",), use_purport=False,
)

GENERALISATION_GRID: list[Condition] = [
    Condition(
        key="M0",
        label="BM25 over raw Meditations passages",
        isolates="lexical floor on the second corpus",
        kind="bm25_meditations",
        requires="meditations_corpus",
    ),
    _pipeline("M2", "Dense over raw Meditations passages",
              "semantic floor without enrichment (second corpus)",
              MEDITATIONS_RAW, requires="meditations_index"),
    _pipeline("M5", "Dense over enriched Meditations passages",
              "THE ENRICHMENT, REPLICATED (vs M2)",
              MEDITATIONS_ENRICHED, requires="meditations_index"),
]

BY_KEY = {c.key: c for c in GRID + MULTILINGUAL_GRID + GENERALISATION_GRID}

#: The comparison each contrast is meant to license, stated before the numbers
#: exist so the analysis cannot be reverse-engineered from whatever came out.
PLANNED_CONTRASTS = [
    ("C2", "C5", "Does synthetic semantic enrichment improve retrieval?"),
    ("C0", "C5", "Does enrichment beat a real lexical baseline?"),
    ("P0", "C10", "Does retrieval beat the model's parametric memory?"),
    ("C6", "C7", "Does domain-calibrating the HyDE prompt matter?"),
    ("C5b", "C7", "Does HyDE help on top of enrichment?"),
    ("C8", "C9", "Does the out-of-domain cross-encoder help or hurt?"),
    ("C9", "C10", "What does MMR cost in relevance?"),
    ("C10", "C12", "Does an explicit emotion arm beat semantic similarity alone?"),
    ("C10", "C13", "Does dropping the out-of-domain reranker improve the served system?"),
    ("L2", "L1", "Is translate-then-retrieve better than direct multilingual?"),
    ("M2", "M5", "Does the enrichment effect replicate on a second corpus?"),
    ("M0", "M5", "Does enrichment beat lexical search on the second corpus?"),
]
