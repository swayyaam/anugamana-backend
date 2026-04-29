import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.llm import generate_advice
from app.config import LLM_MODEL


@pytest.fixture(autouse=True)
def mock_claude(monkeypatch):
    mock = AsyncMock()
    response = MagicMock()
    response.content = [MagicMock(text="Act without attachment.")]
    mock.messages.create = AsyncMock(return_value=response)
    monkeypatch.setattr("app.state.claude", mock)
    return mock


@pytest.mark.asyncio
async def test_generate_advice_returns_string(mock_claude):
    result = await generate_advice("I feel lost", "Perform your duty.")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_generate_advice_none_when_no_client(monkeypatch):
    monkeypatch.setattr("app.state.claude", None)
    result = await generate_advice("I feel lost", "Perform your duty.")
    assert result is None


@pytest.mark.asyncio
async def test_generate_advice_uses_correct_model(mock_claude):
    await generate_advice("query", "verse")
    call_kwargs = mock_claude.messages.create.call_args.kwargs
    assert call_kwargs["model"] == LLM_MODEL


@pytest.mark.asyncio
async def test_generate_advice_includes_query_in_prompt(mock_claude):
    await generate_advice("my specific question", "verse text")
    call_kwargs = mock_claude.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    assert "my specific question" in user_content


@pytest.mark.asyncio
async def test_generate_advice_raises_after_retries(monkeypatch):
    import tenacity

    mock = AsyncMock()
    mock.messages.create = AsyncMock(side_effect=Exception("API error"))
    monkeypatch.setattr("app.state.claude", mock)

    # Tenacity wraps the original error in RetryError after exhausting attempts
    with pytest.raises(tenacity.RetryError):
        await generate_advice("query", "verse")
