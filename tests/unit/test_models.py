import pytest
from pydantic import ValidationError
from app.models import SearchRequest


def test_valid_request_all_fields():
    req = SearchRequest(query="What is duty?", limit=3, chapter=2)
    assert req.query == "What is duty?"
    assert req.limit == 3
    assert req.chapter == 2


def test_valid_request_defaults():
    req = SearchRequest(query="What is duty?")
    assert req.limit == 5
    assert req.chapter is None


def test_valid_request_no_chapter():
    req = SearchRequest(query="What is peace?", limit=10)
    assert req.chapter is None


def test_query_too_long():
    with pytest.raises(ValidationError):
        SearchRequest(query="x" * 501)


def test_query_max_length_ok():
    req = SearchRequest(query="x" * 500)
    assert len(req.query) == 500


def test_limit_too_low():
    with pytest.raises(ValidationError):
        SearchRequest(query="test", limit=0)


def test_limit_too_high():
    with pytest.raises(ValidationError):
        SearchRequest(query="test", limit=21)


def test_limit_boundary_valid():
    assert SearchRequest(query="test", limit=1).limit == 1
    assert SearchRequest(query="test", limit=20).limit == 20


def test_chapter_too_low():
    with pytest.raises(ValidationError):
        SearchRequest(query="test", chapter=0)


def test_chapter_too_high():
    with pytest.raises(ValidationError):
        SearchRequest(query="test", chapter=19)


def test_chapter_boundary_valid():
    assert SearchRequest(query="test", chapter=1).chapter == 1
    assert SearchRequest(query="test", chapter=18).chapter == 18
