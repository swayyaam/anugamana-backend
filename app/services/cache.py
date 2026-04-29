import hashlib
import json
from app import state

CACHE_TTL = 86400  # 24 hours


def make_cache_key(query: str) -> str:
    normalized = query.lower().strip()
    query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"search_cache:{query_hash}"


def get_cached(key: str) -> dict | None:
    result = state.redis.get(key)
    if result:
        return result if isinstance(result, dict) else json.loads(result)
    return None


def set_cached(key: str, data: dict) -> None:
    state.redis.set(key, json.dumps(data), ex=CACHE_TTL)
