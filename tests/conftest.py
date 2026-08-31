"""
Shared fixtures.

The previous suite mocked a Pinecone-era architecture that the pipeline no longer
imports: 16 of 52 tests failed outright and all but two of the passing ones
exercised dead modules. Everything here targets the code that actually ships.

Tests are fast by default — the BGE-M3 / ChromaDB / cross-encoder stack is never
loaded unless a test is marked `requires_index`, which is deselected unless
RUN_INDEX_TESTS=1 is set.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_index: needs the real ChromaDB index and embedding models",
    )


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_INDEX_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_INDEX_TESTS=1 to run against the real index")
    for item in items:
        if "requires_index" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Fake corpus
# ---------------------------------------------------------------------------

def make_verse(verse_id="2.47", chapter=2, verse=47, **extra):
    base = {
        "verse_id": verse_id,
        "chapter": chapter,
        "verse": verse,
        "devanagari": "कर्मण्येवाधिकारस्ते",
        "sanskrit": "karmaṇy evādhikāras te",
        "translation": "You have a right to perform your prescribed duties, "
                       "but you are not entitled to the fruits of action.",
        "rrf_score": 0.05,
        "evidence": "verse",
    }
    base.update(extra)
    return base


@pytest.fixture
def verses():
    return [
        make_verse("2.47", 2, 47, rrf_score=0.060),
        make_verse("3.19", 3, 19, rrf_score=0.055,
                   translation="Therefore, without being attached to the fruits of "
                               "activities, one should act as a matter of duty."),
        make_verse("18.48", 18, 48, rrf_score=0.050,
                   translation="Every endeavour is covered by some fault, as fire "
                               "is covered by smoke."),
    ]


# ---------------------------------------------------------------------------
# LLM stubs — no network in tests
# ---------------------------------------------------------------------------

def _text_response(text: str):
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    return response


@pytest.fixture
def stub_llms(monkeypatch):
    """
    Patch every Anthropic client in the service layer.

    Returns a dict of controls so a test can change one verdict without
    rebuilding the whole stack.
    """
    from app.services import emotion, guardrail, hyde, judge, rag, safety

    controls = {
        "safety_verdict": "safe",
        "guardrail_verdict": "relevant",
        "emotion_json": '{"primary": "despair", "secondary": ["exhaustion"], '
                        '"intensity": 4}',
        "language": ("en-IN", "default"),
        "translation": None,   # None => identity
        "hyde_text": "The conditioned soul must perform prescribed duty without "
                     "attachment to results, for such action purifies the heart.",
        "expansions": "how do I stop caring about outcomes\nwhy do results torment me\n"
                      "acting without attachment",
        "guidance": "This verse speaks to the anxiety of tying your worth to outcomes.",
        "judge_json": '{"grounded": 4, "relevant": 5, "restraint": 5, "unsupported": []}',
    }

    def client_for(kind):
        client = AsyncMock()

        async def create(**kwargs):
            if kind == "safety":
                return _text_response(controls["safety_verdict"])
            if kind == "guardrail":
                return _text_response(controls["guardrail_verdict"])
            if kind == "judge":
                return _text_response(controls["judge_json"])
            if kind == "rag":
                return _text_response(controls["guidance"])
            if kind == "emotion":
                return _text_response(controls["emotion_json"])
            system = kwargs.get("system", "")
            if "alternative phrasings" in system:
                return _text_response(controls["expansions"])
            return _text_response(controls["hyde_text"])

        client.messages.create = create
        return client

    monkeypatch.setattr(safety, "_client", client_for("safety"))
    monkeypatch.setattr(guardrail, "_client", client_for("guardrail"))
    monkeypatch.setattr(hyde, "_client", client_for("hyde"))
    monkeypatch.setattr(rag, "_client", client_for("rag"))
    monkeypatch.setattr(judge, "_client", client_for("judge"))
    monkeypatch.setattr(emotion, "_claude", client_for("emotion"))

    # Sarvam: no key in CI, and no test may reach the network.
    from app.services import sarvam

    async def fake_identify(text):
        return controls["language"]

    async def fake_translate(text, source, target, **kwargs):
        return controls["translation"] or text

    async def fake_to_devanagari(text):
        return text

    monkeypatch.setattr(sarvam, "identify_language", fake_identify)
    monkeypatch.setattr(sarvam, "translate", fake_translate)
    monkeypatch.setattr(sarvam, "to_devanagari", fake_to_devanagari)

    # Never touch the on-disk HyDE cache from a test run.
    monkeypatch.setattr(hyde, "_cache_read", lambda key: None)
    monkeypatch.setattr(hyde, "_cache_write", lambda key, payload: None)
    # Generation must not need the 5MB enriched corpus.
    monkeypatch.setattr(rag, "_commentary_for", lambda verse: ("Commentary text.", "test"))

    return controls


@pytest.fixture
def stub_retrieval(monkeypatch, verses):
    """Patch the retrieval + reranking stack so no index or model is loaded."""
    from app.services import pipeline, reranker, retrieval

    state = {"verses": verses, "direct": [make_verse()]}

    monkeypatch.setattr(
        retrieval, "retrieve", lambda *a, **kw: [dict(v) for v in state["verses"]]
    )
    monkeypatch.setattr(
        retrieval,
        "retrieve_by_verse_id",
        lambda vid, cfg=None: [dict(v) for v in state["direct"]],
    )
    monkeypatch.setattr(pipeline.retrieval, "retrieve", retrieval.retrieve)
    monkeypatch.setattr(
        pipeline.retrieval, "retrieve_by_verse_id", retrieval.retrieve_by_verse_id
    )

    def fake_rerank(query, candidates, *, use_cross_encoder=True, use_mmr=True, top_n=5):
        if not candidates:
            return [], [], "none"
        if not use_cross_encoder:
            ordered = sorted(candidates, key=lambda v: v["rrf_score"], reverse=True)
            for i, v in enumerate(ordered):
                v["relevance"] = round(1.0 / (i + 1), 4)
            return ordered[:top_n], [], "rrf"
        for i, verse in enumerate(candidates):
            verse["cross_score"] = 5.0 - i
            verse["relevance"] = reranker._sigmoid(5.0 - i)
        ordered = sorted(candidates, key=lambda v: v["relevance"], reverse=True)
        return ordered[:top_n], [], "cross_encoder"

    monkeypatch.setattr(reranker, "rerank", fake_rerank)
    monkeypatch.setattr(pipeline.reranker, "rerank", fake_rerank)
    return state


@pytest.fixture
def client(stub_llms, stub_retrieval, tmp_path, monkeypatch):
    """TestClient with a throwaway feedback database and no network."""
    from fastapi.testclient import TestClient

    from app.services import feedback_logger

    monkeypatch.setattr(feedback_logger, "FEEDBACK_DB", tmp_path / "feedback.db")

    import app.main as main_module

    @main_module.asynccontextmanager
    async def no_warmup(app):
        feedback_logger.init_db()
        yield

    monkeypatch.setattr(main_module.app.router, "lifespan_context", no_warmup)
    with TestClient(main_module.app) as test_client:
        yield test_client
