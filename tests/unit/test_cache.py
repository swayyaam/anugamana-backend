import json
import pytest
from unittest.mock import MagicMock
from app.services import cache


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    m = MagicMock()
    m.get.return_value = None
    m.set.return_value = True
    monkeypatch.setattr("app.state.redis", m)
    return m


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

def test_cache_key_same_query_same_key():
    assert cache.make_cache_key("what is duty") == cache.make_cache_key("what is duty")


def test_cache_key_case_insensitive():
    assert cache.make_cache_key("What Is Duty") == cache.make_cache_key("what is duty")


def test_cache_key_strips_whitespace():
    assert cache.make_cache_key("  what is duty  ") == cache.make_cache_key("what is duty")


def test_cache_key_different_queries_different_keys():
    assert cache.make_cache_key("duty") != cache.make_cache_key("peace")


def test_cache_key_format():
    key = cache.make_cache_key("test")
    assert key.startswith("search_cache:")


# ---------------------------------------------------------------------------
# get_cached
# ---------------------------------------------------------------------------

def test_get_cached_miss_returns_none(mock_redis):
    mock_redis.get.return_value = None
    assert cache.get_cached("search_cache:abc") is None


def test_get_cached_hit_string_returns_dict(mock_redis):
    data = {"results": [{"text": "verse"}]}
    mock_redis.get.return_value = json.dumps(data)
    result = cache.get_cached("search_cache:abc")
    assert result == data


def test_get_cached_hit_dict_returns_dict(mock_redis):
    data = {"results": []}
    mock_redis.get.return_value = data
    result = cache.get_cached("search_cache:abc")
    assert result == data


# ---------------------------------------------------------------------------
# set_cached
# ---------------------------------------------------------------------------

def test_set_cached_calls_redis_set(mock_redis):
    data = {"results": []}
    cache.set_cached("search_cache:abc", data)
    mock_redis.set.assert_called_once_with(
        "search_cache:abc", json.dumps(data), ex=86400
    )


def test_set_cached_uses_24h_ttl(mock_redis):
    cache.set_cached("key", {})
    _, kwargs = mock_redis.set.call_args
    assert kwargs.get("ex") == 86400 or mock_redis.set.call_args[0][2] == 86400
