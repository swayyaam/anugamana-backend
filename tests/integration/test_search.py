import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from tests.conftest import FAKE_PC_RESULTS

VALID_PAYLOAD = {"query": "What is my duty?", "limit": 2}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_search_returns_200(client):
    with patch("app.routes.search.encode_query", return_value=[0.1] * 384), \
         patch("app.routes.search.rerank_pairs", return_value=[1.5, 0.8]), \
         patch("app.routes.search.cache.get_cached", return_value=None), \
         patch("app.routes.search.cache.set_cached"):
        response = client.post("/search", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_search_response_has_results_key(client):
    with patch("app.routes.search.encode_query", return_value=[0.1] * 384), \
         patch("app.routes.search.rerank_pairs", return_value=[1.5, 0.8]), \
         patch("app.routes.search.cache.get_cached", return_value=None), \
         patch("app.routes.search.cache.set_cached"):
        data = client.post("/search", json=VALID_PAYLOAD).json()
    assert "results" in data


def test_search_result_count_matches_limit(client):
    with patch("app.routes.search.encode_query", return_value=[0.1] * 384), \
         patch("app.routes.search.rerank_pairs", return_value=[1.5, 0.8]), \
         patch("app.routes.search.cache.get_cached", return_value=None), \
         patch("app.routes.search.cache.set_cached"):
        data = client.post("/search", json={**VALID_PAYLOAD, "limit": 2}).json()
    assert len(data["results"]) <= 2


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_search_returns_cached_result(client):
    cached = {"results": [{"text": "cached verse", "metadata": {}, "score": 0.9}]}
    with patch("app.routes.search.cache.get_cached", return_value=cached) as mock_get, \
         patch("app.routes.search.encode_query") as mock_embed:
        response = client.post("/search", json=VALID_PAYLOAD)
        mock_embed.assert_not_called()
    assert response.json() == cached


def test_search_cache_miss_calls_pinecone(client):
    mock_pc = MagicMock()
    mock_pc.query.return_value = FAKE_PC_RESULTS
    with patch("app.state.pc_index", mock_pc), \
         patch("app.routes.search.encode_query", return_value=[0.1] * 384), \
         patch("app.routes.search.rerank_pairs", return_value=[1.5, 0.8]), \
         patch("app.routes.search.cache.get_cached", return_value=None), \
         patch("app.routes.search.cache.set_cached"):
        client.post("/search", json=VALID_PAYLOAD)
    mock_pc.query.assert_called_once()


def test_search_caches_result_on_miss(client):
    with patch("app.routes.search.encode_query", return_value=[0.1] * 384), \
         patch("app.routes.search.rerank_pairs", return_value=[1.5, 0.8]), \
         patch("app.routes.search.cache.get_cached", return_value=None), \
         patch("app.routes.search.cache.set_cached") as mock_set:
        client.post("/search", json=VALID_PAYLOAD)
    mock_set.assert_called_once()


# ---------------------------------------------------------------------------
# Chapter filter
# ---------------------------------------------------------------------------

def test_search_passes_chapter_filter_to_pinecone(client):
    mock_pc = MagicMock()
    mock_pc.query.return_value = FAKE_PC_RESULTS
    with patch("app.state.pc_index", mock_pc), \
         patch("app.routes.search.encode_query", return_value=[0.1] * 384), \
         patch("app.routes.search.rerank_pairs", return_value=[1.5, 0.8]), \
         patch("app.routes.search.cache.get_cached", return_value=None), \
         patch("app.routes.search.cache.set_cached"):
        client.post("/search", json={**VALID_PAYLOAD, "chapter": 3})
    call_kwargs = mock_pc.query.call_args.kwargs
    assert call_kwargs["filter"] == {"chapter": {"$eq": 3}}


def test_search_no_chapter_filter_when_none(client):
    mock_pc = MagicMock()
    mock_pc.query.return_value = FAKE_PC_RESULTS
    with patch("app.state.pc_index", mock_pc), \
         patch("app.routes.search.encode_query", return_value=[0.1] * 384), \
         patch("app.routes.search.rerank_pairs", return_value=[1.5, 0.8]), \
         patch("app.routes.search.cache.get_cached", return_value=None), \
         patch("app.routes.search.cache.set_cached"):
        client.post("/search", json=VALID_PAYLOAD)
    call_kwargs = mock_pc.query.call_args.kwargs
    assert call_kwargs["filter"] is None


# ---------------------------------------------------------------------------
# RAG advice
# ---------------------------------------------------------------------------

def test_search_limit_1_calls_generate_advice(client):
    with patch("app.routes.search.encode_query", return_value=[0.1] * 384), \
         patch("app.routes.search.rerank_pairs", return_value=[1.5]), \
         patch("app.routes.search.cache.get_cached", return_value=None), \
         patch("app.routes.search.cache.set_cached"), \
         patch("app.routes.search.generate_advice", new_callable=AsyncMock, return_value="Seek peace.") as mock_advice:
        client.post("/search", json={"query": "What is duty?", "limit": 1})
    mock_advice.assert_called_once()


def test_search_limit_gt_1_skips_generate_advice(client):
    with patch("app.routes.search.encode_query", return_value=[0.1] * 384), \
         patch("app.routes.search.rerank_pairs", return_value=[1.5, 0.8]), \
         patch("app.routes.search.cache.get_cached", return_value=None), \
         patch("app.routes.search.cache.set_cached"), \
         patch("app.routes.search.generate_advice", new_callable=AsyncMock) as mock_advice:
        client.post("/search", json={"query": "What is duty?", "limit": 5})
    mock_advice.assert_not_called()


def test_search_advice_in_first_result_metadata(client):
    with patch("app.routes.search.encode_query", return_value=[0.1] * 384), \
         patch("app.routes.search.rerank_pairs", return_value=[1.5]), \
         patch("app.routes.search.cache.get_cached", return_value=None), \
         patch("app.routes.search.cache.set_cached"), \
         patch("app.routes.search.generate_advice", new_callable=AsyncMock, return_value="Act without attachment."):
        data = client.post("/search", json={"query": "What is duty?", "limit": 1}).json()
    assert data["results"][0]["metadata"]["ai_advice"] == "Act without attachment."


def test_search_llm_failure_returns_results_without_advice(client):
    with patch("app.routes.search.encode_query", return_value=[0.1] * 384), \
         patch("app.routes.search.rerank_pairs", return_value=[1.5]), \
         patch("app.routes.search.cache.get_cached", return_value=None), \
         patch("app.routes.search.cache.set_cached"), \
         patch("app.routes.search.generate_advice", new_callable=AsyncMock, side_effect=Exception("LLM down")):
        data = client.post("/search", json={"query": "What is duty?", "limit": 1}).json()
    assert "results" in data
    assert "ai_advice" not in data["results"][0]["metadata"]


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------

def test_search_empty_pinecone_results(client):
    mock_pc = MagicMock()
    mock_pc.query.return_value = {"matches": []}
    with patch("app.state.pc_index", mock_pc), \
         patch("app.routes.search.encode_query", return_value=[0.1] * 384), \
         patch("app.routes.search.cache.get_cached", return_value=None):
        data = client.post("/search", json=VALID_PAYLOAD).json()
    assert data == {"results": []}


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_search_503_when_services_not_ready(client):
    with patch("app.state.embedder", None), \
         patch("app.state.pc_index", None), \
         patch("app.state.reranker", None):
        response = client.post("/search", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_search_422_query_too_long(client):
    response = client.post("/search", json={"query": "x" * 501, "limit": 5})
    assert response.status_code == 422


def test_search_422_limit_zero(client):
    response = client.post("/search", json={"query": "test", "limit": 0})
    assert response.status_code == 422


def test_search_422_invalid_chapter(client):
    response = client.post("/search", json={"query": "test", "chapter": 19})
    assert response.status_code == 422
