"""
Bulbul text-to-speech.

Two distinct use cases with different economics:

* **Verse audio** — 700 Sanskrit verses, fixed text, never changes. Generated
  once by scripts/generate_audio.py and served from disk. Zero per-request cost.
* **Guidance audio** — generated per request, so it is cached by content hash and
  only produced on explicit demand.

Audio is returned as base64 so the caller can stream it, embed it, or persist it
without this module deciding.
"""

from __future__ import annotations

import base64
import hashlib

import structlog

from app.config import (
    DATA_DIR,
    PIVOT_LANGUAGE,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_SPEAKER,
)
from app.services.sarvam.client import SarvamError, SarvamUnavailable, client, pick

logger = structlog.get_logger(__name__)

AUDIO_CACHE = DATA_DIR / "audio_cache"

#: Bulbul rejects very long inputs; purports and guidance are chunked to this.
MAX_TTS_CHARS = 480


def _cache_path(text: str, language: str, speaker: str) -> "object":
    key = hashlib.sha256(
        f"{SARVAM_TTS_MODEL}|{speaker}|{language}|{text}".encode("utf-8")
    ).hexdigest()
    return AUDIO_CACHE / f"{key}.wav"


def _chunk(text: str, limit: int = MAX_TTS_CHARS) -> list[str]:
    """Split on sentence boundaries so prosody is not cut mid-clause."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for sentence in text.replace("\n", " ").split(". "):
        candidate = f"{current}. {sentence}" if current else sentence
        if len(candidate) > limit and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


async def synthesize(
    text: str,
    language: str = PIVOT_LANGUAGE,
    speaker: str = SARVAM_TTS_SPEAKER,
    *,
    use_cache: bool = True,
) -> str | None:
    """
    Returns base64-encoded WAV, or None when Sarvam is unavailable or fails.
    Audio is an enhancement; its absence must never fail a search response.
    """
    if not text.strip():
        return None

    path = _cache_path(text, language, speaker)
    if use_cache and path.exists():
        return base64.b64encode(path.read_bytes()).decode("ascii")

    try:
        segments: list[bytes] = []
        for piece in _chunk(text):
            response = await client.post(
                "/text-to-speech",
                {
                    "inputs": [piece],
                    "target_language_code": language,
                    "speaker": speaker,
                    "model": SARVAM_TTS_MODEL,
                },
            )
            audios = pick(response, "audios", "audio", "output")
            if isinstance(audios, str):
                audios = [audios]
            segments.extend(base64.b64decode(a) for a in audios)

        if not segments:
            return None

        audio = b"".join(segments)
        if use_cache:
            AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
        return base64.b64encode(audio).decode("ascii")

    except SarvamUnavailable:
        return None
    except SarvamError as e:
        logger.warning("sarvam_tts_failed", language=language, error=str(e))
        return None
