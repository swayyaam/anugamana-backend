import numpy as np
import pytest
from unittest.mock import MagicMock
from app.services.embedder import encode_query, rerank_pairs


@pytest.fixture(autouse=True)
def mock_embedding_state(monkeypatch):
    fake_embedding = np.array([[0.6, 0.8]], dtype=np.float32)

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": np.array([[1, 2, 3]]),
        "attention_mask": np.array([[1, 1, 1]]),
    }

    mock_embedder = MagicMock()
    mock_embedder.return_value = [fake_embedding]

    mock_reranker = MagicMock()
    reranker_out = MagicMock()
    reranker_out.logits = np.array([[1.5], [0.8]], dtype=np.float32)
    mock_reranker.return_value = reranker_out

    monkeypatch.setattr("app.state.tokenizer_emb", mock_tokenizer)
    monkeypatch.setattr("app.state.embedder", mock_embedder)
    monkeypatch.setattr("app.state.tokenizer_rerank", mock_tokenizer)
    monkeypatch.setattr("app.state.reranker", mock_reranker)


# ---------------------------------------------------------------------------
# encode_query
# ---------------------------------------------------------------------------

def test_encode_query_returns_list_of_floats():
    result = encode_query("What is duty?")
    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


def test_encode_query_is_normalized():
    result = encode_query("What is duty?")
    norm = sum(x ** 2 for x in result) ** 0.5
    assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# rerank_pairs
# ---------------------------------------------------------------------------

def test_rerank_pairs_returns_list():
    scores = rerank_pairs("duty", ["verse one", "verse two"])
    assert isinstance(scores, list)


def test_rerank_pairs_length_matches_texts():
    texts = ["a", "b"]
    scores = rerank_pairs("query", texts)
    assert len(scores) == len(texts)


def test_rerank_pairs_single_text(monkeypatch):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": np.array([[1, 2, 3]]),
        "attention_mask": np.array([[1, 1, 1]]),
    }
    mock_reranker = MagicMock()
    single_out = MagicMock()
    single_out.logits = np.array([[1.5]], dtype=np.float32)
    mock_reranker.return_value = single_out
    monkeypatch.setattr("app.state.tokenizer_rerank", mock_tokenizer)
    monkeypatch.setattr("app.state.reranker", mock_reranker)

    scores = rerank_pairs("query", ["single verse"])
    assert len(scores) == 1
