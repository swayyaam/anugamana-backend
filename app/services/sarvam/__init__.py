"""
Sarvam AI integration — Indic language support for the retrieval pipeline.

Layout:
    client.py  transport, retries, schema-tolerant response reading
    text.py    language identification, translation (Mayura), transliteration
    tts.py     speech synthesis (Bulbul)

Everything here is optional. With no SARVAM_API_KEY configured every entry point
degrades to a defined fallback and the English pipeline is unaffected.
"""

from app.services.sarvam.client import (
    SarvamClient,
    SarvamError,
    SarvamUnavailable,
    client,
)
from app.services.sarvam.text import (
    detect_script,
    identify_language,
    looks_romanised_indic,
    to_devanagari,
    translate,
    transliterate,
)
from app.services.sarvam.tts import synthesize

__all__ = [
    "SarvamClient",
    "SarvamError",
    "SarvamUnavailable",
    "client",
    "detect_script",
    "identify_language",
    "looks_romanised_indic",
    "synthesize",
    "to_devanagari",
    "translate",
    "transliterate",
]
