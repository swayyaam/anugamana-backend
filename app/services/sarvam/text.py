"""
Sarvam text services: language identification, translation (Mayura), and
transliteration.

Every function degrades to a defined, honest fallback when Sarvam is
unconfigured or failing — the search pipeline must keep working for English
users regardless of Indic availability.

Why transliteration matters here
--------------------------------
A large share of real queries for this corpus are Sanskrit typed in Roman script
("karmanye vadhikaraste ma phaleshu"). The index holds Devanagari and IAST, so
the lexical arm sees almost no overlap with a Roman-typed query. Mapping
Roman → Devanagari before the sparse search is a cheap, measurable intervention
rather than a convenience feature — condition C11 in the ablation grid.
"""

from __future__ import annotations

import asyncio
import re

import structlog

from app.config import (
    PIVOT_LANGUAGE,
    SARVAM_TRANSLATE_MODEL,
    SUPPORTED_LANGUAGES,
)
from app.services.sarvam.client import SarvamError, SarvamUnavailable, client, pick

logger = structlog.get_logger(__name__)

# Unicode blocks for the scripts we can identify without a network call.
_SCRIPT_RANGES = {
    "hi-IN": r"ऀ-ॿ",   # Devanagari — also Marathi, Sanskrit
    "bn-IN": r"ঀ-৿",
    "pa-IN": r"਀-੿",
    "gu-IN": r"઀-૿",
    "od-IN": r"଀-୿",
    "ta-IN": r"஀-௿",
    "te-IN": r"ఀ-౿",
    "kn-IN": r"ಀ-೿",
    "ml-IN": r"ഀ-ൿ",
}
_SCRIPT_RE = {
    lang: re.compile(f"[{ranges}]") for lang, ranges in _SCRIPT_RANGES.items()
}

# Sanskrit/Hindi words that commonly appear romanised in real queries. Used only
# as a hint that a Roman-script query may benefit from transliteration.
_ROMANISED_HINTS = {
    "karma", "dharma", "yoga", "atma", "atman", "moksha", "bhakti", "guna",
    "maya", "krishna", "arjuna", "gita", "geeta", "shloka", "sloka", "vedanta",
    "samsara", "nishkama", "phala", "brahman", "jiva", "prakriti", "purusha",
    "vairagya", "sannyasa", "tapasya", "ahimsa", "shanti", "kshetra",
}


def detect_script(text: str) -> str | None:
    """Offline script detection. Returns a BCP-47 code or None for Latin/unknown."""
    for lang, pattern in _SCRIPT_RE.items():
        if pattern.search(text):
            return lang
    return None


def looks_romanised_indic(text: str) -> bool:
    """True when a Latin-script query contains recognisable Sanskrit vocabulary."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & _ROMANISED_HINTS)


async def identify_language(text: str) -> tuple[str, str]:
    """
    Returns (language_code, method).

    method is "script" (offline, free, certain for non-Latin scripts),
    "sarvam" (the /text-lid endpoint), or "default" (assumed English).
    """
    script_lang = detect_script(text)
    if script_lang:
        return script_lang, "script"

    try:
        response = await client.post("/text-lid", {"input": text})
        code = pick(response, "language_code", "lang_code", "language", required=False)
        if code and code in SUPPORTED_LANGUAGES:
            return code, "sarvam"
        if code:
            logger.info("sarvam_lid_unsupported_language", code=code)
    except SarvamUnavailable:
        pass
    except SarvamError as e:
        logger.warning("sarvam_lid_failed", error=str(e))

    return PIVOT_LANGUAGE, "default"


#: Mayura rejects input beyond this. Generated guidance routinely exceeds it,
#: and the failure is silent from the user's side — they simply receive English.
MAX_TRANSLATE_CHARS = 900


def _split_for_translation(text: str, limit: int = MAX_TRANSLATE_CHARS) -> list[str]:
    """Split on sentence boundaries so no clause is translated out of context."""
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > limit and current:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


async def translate(
    text: str,
    source: str,
    target: str,
    *,
    mode: str = "formal",
) -> str:
    """
    Translate via Mayura. Returns the input unchanged on any failure — a partial
    pipeline is better than a 500, and the caller records the degradation.
    """
    if not text.strip() or source == target:
        return text

    if len(text) > MAX_TRANSLATE_CHARS:
        pieces = await asyncio.gather(*(
            translate(piece, source, target, mode=mode)
            for piece in _split_for_translation(text)
        ))
        return " ".join(pieces)

    try:
        response = await client.post(
            "/translate",
            {
                "input": text,
                "source_language_code": source,
                "target_language_code": target,
                "model": SARVAM_TRANSLATE_MODEL,
                "mode": mode,
            },
        )
        return pick(response, "translated_text", "output", "text")
    except SarvamUnavailable:
        return text
    except SarvamError as e:
        logger.warning("sarvam_translate_failed", source=source, target=target, error=str(e))
        return text


async def transliterate(text: str, source: str, target: str) -> str:
    """Script conversion within a language (Roman <-> Devanagari)."""
    if not text.strip():
        return text
    try:
        response = await client.post(
            "/transliterate",
            {
                "input": text,
                "source_language_code": source,
                "target_language_code": target,
            },
        )
        return pick(response, "transliterated_text", "output", "text")
    except SarvamUnavailable:
        return text
    except SarvamError as e:
        logger.warning("sarvam_transliterate_failed", error=str(e))
        return text


async def to_devanagari(text: str) -> str:
    """Romanised Sanskrit -> Devanagari, for the lexical retrieval arm."""
    return await transliterate(text, PIVOT_LANGUAGE, "hi-IN")
