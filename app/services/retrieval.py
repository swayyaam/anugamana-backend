"""
Hybrid retrieval: dense (ChromaDB) + sparse (BGE-M3 lexical weights), fused with RRF.

The retriever is parameterised by an `IndexSpec` (which physical index) and a
`RetrievalConfig` (which arms are switched on). The API uses ENRICHED_FULL; the
evaluation harness instantiates the ablation conditions from the same code, so
the evaluated system is the served system by construction.

Chunk provenance (fixed 2026-08-31 — audit E-03)
------------------------------------------------
RRF fusion groups document-level hits into verses. The previous implementation
threw away *which* purport chunk won, then re-read metadata from the verse-level
`_meaning` document — which carries no chunk fields. rag.py therefore always fell
back to paragraphs 0-2 and the documented parent-child retrieval never ran.
`_fuse_and_group` now returns the winning purport chunk id per verse, and
`retrieve()` attaches that chunk's parent window to the verse dict.
"""

from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
import structlog
from FlagEmbedding import BGEM3FlagModel

from app.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RRF_K,
    SPARSE_FILE,
    TOP_K,
    TOP_VERSES,
)

logger = structlog.get_logger(__name__)

DATA_DIR = CHROMA_DIR.parent
RAW_CHROMA_DIR = DATA_DIR / "chroma_raw"
RAW_SPARSE_FILE = DATA_DIR / "sparse_index_raw.pkl"


# ---------------------------------------------------------------------------
# Index / retrieval specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexSpec:
    """A physical index on disk."""
    name: str
    chroma_dir: Path
    verses_collection: str
    purport_collection: str
    sparse_file: Path


#: The enriched index — verse vectors built from `text_for_embedding`.
ENRICHED_INDEX = IndexSpec(
    name="enriched",
    chroma_dir=CHROMA_DIR,
    verses_collection="gita_verses",
    purport_collection="gita_purport",
    sparse_file=SPARSE_FILE,
)

#: The control index — identical construction, but verse vectors are built from
#: the raw translation only. Built by scripts/build_raw_index.py. Without this
#: index the effect of enrichment cannot be isolated (audit F-02).
RAW_INDEX = IndexSpec(
    name="raw",
    chroma_dir=RAW_CHROMA_DIR,
    verses_collection="raw_verses",
    purport_collection="raw_purport",
    sparse_file=RAW_SPARSE_FILE,
)


#: Second corpus — Marcus Aurelius, Meditations. Public domain, a different
#: tradition and source language, and built with the identical enriched/raw pair
#: so the C2-vs-C5 contrast can be replicated rather than merely asserted.
MEDITATIONS_DIR = DATA_DIR / "corpora" / "meditations"
MEDITATIONS_INDEX = IndexSpec(
    name="meditations",
    chroma_dir=MEDITATIONS_DIR / "chroma",
    verses_collection="meditations_enriched",
    purport_collection="meditations_enriched",   # no separate commentary layer
    sparse_file=MEDITATIONS_DIR / "sparse_index.pkl",
)
MEDITATIONS_RAW_INDEX = IndexSpec(
    name="meditations_raw",
    chroma_dir=MEDITATIONS_DIR / "chroma_raw",
    verses_collection="meditations_raw",
    purport_collection="meditations_raw",
    sparse_file=MEDITATIONS_DIR / "sparse_index_raw.pkl",
)


@dataclass(frozen=True)
class RetrievalConfig:
    """Which retrieval arms are active. One ablation condition = one instance."""
    index: IndexSpec = ENRICHED_INDEX
    dense: bool = True
    sparse: bool = True
    #: verse-level vector types to search: "meaning" and/or "translation"
    doc_types: tuple[str, ...] = ("meaning", "translation")
    use_purport: bool = True
    top_k: int = TOP_K
    top_verses: int = TOP_VERSES
    #: Per-arm fusion weights. None means uniform, i.e. classic RRF.
    arm_weights: tuple[tuple[str, float], ...] | None = None

    @property
    def weights(self) -> dict[str, float] | None:
        return dict(self.arm_weights) if self.arm_weights else None

    def describe(self) -> str:
        arms = []
        if self.dense:
            arms.append("dense[" + "+".join(self.doc_types) + "]")
        if self.use_purport:
            arms.append("purport")
        if self.sparse:
            arms.append("sparse")
        return f"{self.index.name}:{'|'.join(arms) or 'none'}"


DEFAULT_CONFIG = RetrievalConfig()


# ---------------------------------------------------------------------------
# Lazily-loaded shared resources
# ---------------------------------------------------------------------------

# BGEM3FlagModel construction is not thread-safe, and lru_cache does not
# serialise concurrent misses: several worker threads entering _load_model() at
# once each begin building the model and end up sharing partially-initialised
# state, which surfaces as "Cannot copy out of meta tensor". The evaluation
# harness runs queries concurrently, so this is reachable in practice.
_MODEL_LOCK = threading.Lock()
_EMBED_LOCK = threading.Lock()
_model_instance: BGEM3FlagModel | None = None


def _load_model() -> BGEM3FlagModel:
    global _model_instance
    if _model_instance is None:
        with _MODEL_LOCK:
            if _model_instance is None:
                _model_instance = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)
    return _model_instance


_COLLECTIONS_LOCK = threading.Lock()
_collections_cache: dict[str, tuple] = {}


def _load_collections(spec: IndexSpec):
    """
    Serialised, like the model loaders. Constructing two PersistentClients
    concurrently races inside Chroma's tenant setup and surfaces as
    "Could not connect to tenant default_tenant" — intermittent, and it silently
    empties a condition's results for the affected queries.
    """
    cached = _collections_cache.get(spec.name)
    if cached is not None:
        return cached

    with _COLLECTIONS_LOCK:
        cached = _collections_cache.get(spec.name)
        if cached is not None:
            return cached

        client = chromadb.PersistentClient(path=str(spec.chroma_dir))
        verses_col = client.get_collection(spec.verses_collection)
        try:
            purport_col = client.get_collection(spec.purport_collection)
        except Exception:
            purport_col = None
            logger.warning("purport_collection_missing", index=spec.name)

        _collections_cache[spec.name] = (verses_col, purport_col)
        return _collections_cache[spec.name]


@lru_cache(maxsize=4)
def _load_sparse(sparse_file: Path) -> dict:
    with open(sparse_file, "rb") as f:
        return pickle.load(f)


# Backwards-compatible warm-up hooks used by app.main's lifespan.
def _load_chroma():
    return _load_collections(ENRICHED_INDEX)


def _load_sparse_default():
    return _load_sparse(ENRICHED_INDEX.sparse_file)


def _embed(texts: list[str]) -> tuple[np.ndarray, list[dict]]:
    model = _load_model()
    # A single FlagModel instance is not safe for concurrent encode() calls.
    # Torch already parallelises internally, so serialising here costs little and
    # still lets ChromaDB queries and API calls overlap across worker threads.
    with _EMBED_LOCK:
        out = model.encode(
            texts,
            batch_size=max(1, len(texts)),
            max_length=512,
            return_dense=True,
            return_sparse=True,
        )
    return out["dense_vecs"], out["lexical_weights"]


# ---------------------------------------------------------------------------
# Individual search arms
# ---------------------------------------------------------------------------

def _sparse_search(
    sparse_index: dict,
    query_weights: dict,
    top_k: int,
    id_filter: tuple[str, ...] | None = None,
) -> list[str]:
    """Score docs against query lexical weights; return top_k doc ids."""
    scores: dict[str, float] = {}
    for token_id, q_weight in query_weights.items():
        postings = sparse_index.get(str(token_id))
        if not postings:
            continue
        qw = float(q_weight)
        for doc_id, d_weight in postings.items():
            if id_filter and not any(suffix in doc_id for suffix in id_filter):
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + qw * float(d_weight)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:top_k]]


def _dense_search(
    collection,
    vector: np.ndarray,
    top_k: int,
    where: dict | None = None,
) -> list[str]:
    kwargs = {"query_embeddings": [vector.tolist()], "n_results": top_k}
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    return list(results["ids"][0])


def _probe_arms(probes: list[tuple[str, str]]) -> list[str]:
    return [arm for arm, _ in probes]


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def _verse_id_of(doc_id: str) -> str:
    """'2.47_meaning' -> '2.47'; '2.47_purport_3' -> '2.47'."""
    if "_purport_" in doc_id:
        return doc_id.split("_purport_")[0]
    return doc_id.rsplit("_", 1)[0]


#: Retrieval arms, for weighted fusion. Plain RRF treats every arm as equally
#: trustworthy, which is certainly false — the sparse arm and the dense meaning
#: arm are not equally good at this task. Weights are fitted on the benchmark by
#: eval/fit_fusion_weights.py; the defaults below are uniform, i.e. classic RRF.
ARMS = (
    "dense_meaning",
    "dense_translation",
    "dense_synonyms",
    "purport",
    "sparse",
    "emotion",
    "transliteration",
)

DEFAULT_ARM_WEIGHTS: dict[str, float] = {arm: 1.0 for arm in ARMS}


def _fuse_and_group(
    ranked_lists: list[tuple[str, list[str]]],
    arm_weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """
    Weighted-RRF fuse several labelled ranked lists, then group to verses.

    Each entry is (arm_name, doc_ids). A document's contribution from one list
    is `weight[arm] / (RRF_K + rank + 1)`.

    Returns:
        verse_scores      {verse_id: fused score}
        best_purport_doc  {verse_id: highest-scoring purport chunk id}, where one exists
    """
    weights = arm_weights or DEFAULT_ARM_WEIGHTS
    doc_scores: dict[str, float] = {}
    for arm, ranked in ranked_lists:
        weight = weights.get(arm, 1.0)
        if weight == 0.0:
            continue
        for rank, doc_id in enumerate(ranked):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + weight * _rrf_score(rank)

    verse_scores: dict[str, float] = {}
    best_purport_doc: dict[str, str] = {}
    best_purport_score: dict[str, float] = {}

    for doc_id, score in doc_scores.items():
        verse_id = _verse_id_of(doc_id)

        # Verse score = best evidence from any of its vectors.
        if score > verse_scores.get(verse_id, float("-inf")):
            verse_scores[verse_id] = score

        # Track the strongest purport chunk separately — this is the provenance
        # that generation needs and that used to be discarded.
        if "_purport_" in doc_id and score > best_purport_score.get(verse_id, float("-inf")):
            best_purport_score[verse_id] = score
            best_purport_doc[verse_id] = doc_id

    return verse_scores, best_purport_doc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    hyde_text: str,
    all_queries: list[str],
    config: RetrievalConfig = DEFAULT_CONFIG,
    extra_probes: list[str] | None = None,
) -> list[dict]:
    """
    Hybrid retrieval over one index.

    `extra_probes` are additional texts embedded and searched as further ranked
    lists before RRF fusion. The emotion arm (app/services/emotion.py) supplies
    one; a transliterated Devanagari form of a romanised query supplies another.
    Adding an arm here rather than post-hoc reweighting keeps every signal on the
    same rank-fusion footing, so a condition that switches one on differs from
    its baseline by exactly one ranked list.

    Returns up to `config.top_verses` verse dicts carrying verse metadata,
    `rrf_score`, and — when a purport chunk was the source of evidence —
    `chunk_index`, `parent_start`, `parent_end` for the generation stage.
    """
    verses_col, purport_col = _load_collections(config.index)
    ranked_lists = arm_lists(hyde_text, all_queries, config, extra_probes)

    if not ranked_lists:
        return []

    verse_scores, best_purport_doc = _fuse_and_group(ranked_lists, config.weights)
    top_verse_ids = sorted(
        verse_scores, key=lambda v: verse_scores[v], reverse=True
    )[: config.top_verses]

    if not top_verse_ids:
        return []

    return _hydrate(
        verses_col, purport_col, top_verse_ids, verse_scores, best_purport_doc, config
    )


def arm_lists(
    hyde_text: str,
    all_queries: list[str],
    config: RetrievalConfig = DEFAULT_CONFIG,
    extra_probes: list | None = None,
) -> list[tuple[str, list[str]]]:
    """
    Every retrieval arm's ranked doc-id list, labelled and unfused.

    Separated from `retrieve` so fusion weights can be fitted offline: the arms
    are computed once per query and then re-fused thousands of times during
    optimisation, instead of re-running retrieval on every iteration.
    """
    verses_col, purport_col = _load_collections(config.index)

    # extra_probes may be plain strings (legacy) or (arm, text) pairs.
    probes: list[tuple[str, str]] = []
    for probe in extra_probes or []:
        if isinstance(probe, tuple):
            arm, text = probe
        else:
            arm, text = "emotion", probe
        if text and text.strip():
            probes.append((arm, text))
    texts_to_embed = [hyde_text] + list(all_queries) + [text for _, text in probes]
    dense_vecs, sparse_weights = _embed(texts_to_embed)

    ranked_lists: list[tuple[str, list[str]]] = []

    # Which embedded text produced which vector, so probe arms can be labelled.
    # Order matches texts_to_embed: [hyde] + all_queries + probes.
    n_query_vecs = 1 + len(all_queries)
    probe_arms = _probe_arms(probes)

    if config.dense:
        for index, vec in enumerate(dense_vecs):
            probe_arm = None if index < n_query_vecs else probe_arms[index - n_query_vecs]
            for doc_type in config.doc_types:
                arm = probe_arm or f"dense_{doc_type}"
                ranked_lists.append((
                    arm,
                    _dense_search(verses_col, vec, config.top_k,
                                  where={"type": doc_type}),
                ))
            if config.use_purport and purport_col is not None:
                ranked_lists.append((
                    probe_arm or "purport",
                    _dense_search(purport_col, vec, config.top_k),
                ))

    if config.sparse:
        sparse_index = _load_sparse(config.index.sparse_file)
        # Restrict the lexical arm to the same document population the dense arm
        # sees, otherwise conditions are not comparable.
        suffixes = tuple(f"_{t}" for t in config.doc_types)
        if config.use_purport:
            suffixes = suffixes + ("_purport_",)
        for index, weights in enumerate(sparse_weights):
            probe_arm = (
                None if index < n_query_vecs else probe_arms[index - n_query_vecs]
            )
            ranked_lists.append((
                probe_arm or "sparse",
                _sparse_search(sparse_index, weights, config.top_k,
                               id_filter=suffixes),
            ))

    return ranked_lists


def _hydrate(
    verses_col,
    purport_col,
    verse_ids: list[str],
    verse_scores: dict[str, float],
    best_purport_doc: dict[str, str],
    config: RetrievalConfig,
) -> list[dict]:
    """Attach verse metadata + winning purport-chunk provenance."""
    # Verse metadata. Prefer the meaning doc; fall back to translation for
    # indexes that carry no meaning vectors (the raw control index).
    primary_type = config.doc_types[0] if config.doc_types else "translation"
    wanted = [f"{vid}_{primary_type}" for vid in verse_ids]
    fetched = verses_col.get(ids=wanted, include=["metadatas"])
    meta_by_id = dict(zip(fetched["ids"], fetched["metadatas"]))

    # Purport provenance, batched.
    chunk_meta_by_id: dict[str, dict] = {}
    chunk_ids = [best_purport_doc[v] for v in verse_ids if v in best_purport_doc]
    if chunk_ids and purport_col is not None:
        got = purport_col.get(ids=chunk_ids, include=["metadatas"])
        chunk_meta_by_id = dict(zip(got["ids"], got["metadatas"]))

    verses: list[dict] = []
    for vid in verse_ids:
        meta = meta_by_id.get(f"{vid}_{primary_type}")
        if not meta:
            continue
        verse = {**meta, "rrf_score": verse_scores[vid]}

        chunk_id = best_purport_doc.get(vid)
        chunk_meta = chunk_meta_by_id.get(chunk_id) if chunk_id else None
        if chunk_meta:
            verse["chunk_index"] = int(chunk_meta.get("chunk_index", 0))
            verse["parent_start"] = int(chunk_meta.get("parent_start", 0))
            verse["parent_end"] = int(chunk_meta.get("parent_end", 0))
            verse["evidence"] = "purport"
        else:
            # No purport chunk was retrieved for this verse — generation will
            # fall back to the head of the purport, and says so explicitly.
            verse["evidence"] = "verse"
        verses.append(verse)

    return verses


def retrieve_by_verse_id(
    verse_id: str, config: RetrievalConfig = DEFAULT_CONFIG
) -> list[dict]:
    """
    Direct verse lookup — bypasses embedding and search entirely.
    Returns a one-element list, or [] when the verse does not exist.
    """
    verses_col, _ = _load_collections(config.index)
    primary_type = config.doc_types[0] if config.doc_types else "translation"
    try:
        results = verses_col.get(
            ids=[f"{verse_id}_{primary_type}"], include=["metadatas"]
        )
    except Exception as e:
        logger.warning("direct_lookup_failed", verse_id=verse_id, error=str(e))
        return []
    if not results["ids"]:
        return []
    return [{**results["metadatas"][0], "rrf_score": 1.0, "evidence": "direct"}]
