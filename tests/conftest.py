import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared fake data
# ---------------------------------------------------------------------------

FAKE_VERSE = {
    "id": "c2v47",
    "chapter": 2,
    "verse": 47,
    "text": "karmany evadhikaras te",
    "translation": "You have a right to perform your duty",
    "meaning": "Act without attachment to the fruits of action",
    "score": 0.95,
}

FAKE_PC_RESULTS = {
    "matches": [
        {
            "id": "c2v47",
            "score": 0.95,
            "metadata": {
                "chapter": 2,
                "verse": 47,
                "text": "karmany evadhikaras te",
                "translation": "You have a right to perform your duty",
                "meaning": "Act without attachment to the fruits of action",
            },
        },
        {
            "id": "c3v19",
            "score": 0.80,
            "metadata": {
                "chapter": 3,
                "verse": 19,
                "text": "tasmad asaktah satatam",
                "translation": "Therefore without attachment perform your duty",
                "meaning": "Perform action as a sacrifice",
            },
        },
    ]
}

FAKE_EMBEDDING = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
FAKE_RERANK_LOGITS = np.array([[1.5], [0.8]], dtype=np.float32)


# ---------------------------------------------------------------------------
# State mock fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_state(monkeypatch):
    """Patch all global state with controllable mocks."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": np.array([[1, 2, 3]]),
        "attention_mask": np.array([[1, 1, 1]]),
    }

    mock_embedder = MagicMock()
    mock_embedder.return_value = [FAKE_EMBEDDING]

    mock_reranker = MagicMock()
    reranker_output = MagicMock()
    reranker_output.logits = FAKE_RERANK_LOGITS
    mock_reranker.return_value = reranker_output

    mock_pc_index = MagicMock()
    mock_pc_index.query.return_value = FAKE_PC_RESULTS

    mock_claude = AsyncMock()
    claude_response = MagicMock()
    claude_response.content = [MagicMock(text="Seek peace within yourself.")]
    mock_claude.messages.create = AsyncMock(return_value=claude_response)

    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True

    monkeypatch.setattr("app.state.embedder", mock_embedder)
    monkeypatch.setattr("app.state.reranker", mock_reranker)
    monkeypatch.setattr("app.state.pc_index", mock_pc_index)
    monkeypatch.setattr("app.state.tokenizer_emb", mock_tokenizer)
    monkeypatch.setattr("app.state.tokenizer_rerank", mock_tokenizer)
    monkeypatch.setattr("app.state.claude", mock_claude)
    monkeypatch.setattr("app.state.redis", mock_redis)

    return {
        "embedder": mock_embedder,
        "reranker": mock_reranker,
        "pc_index": mock_pc_index,
        "tokenizer_emb": mock_tokenizer,
        "tokenizer_rerank": mock_tokenizer,
        "claude": mock_claude,
        "redis": mock_redis,
    }


# ---------------------------------------------------------------------------
# TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client(mock_state):
    """FastAPI TestClient with all external services mocked."""
    from app.main import app
    with TestClient(app) as c:
        yield c
