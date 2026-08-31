"""
Sarvam AI integration — Indic language support for the retrieval pipeline.

Layout:
    client.py  transport, retries, schema-tolerant response reading
    text.py    language identification, translation (Mayura), transliteration
    tts.py     speech synthesis (Bulbul)

Everything here is optional. With no SARVAM_API_KEY configured every entry point
degrades to a defined fallback and the English pipeline is unaffected.
"""

# NB: the singleton in client.py is deliberately NOT re-exported here. Naming it
# `client` at package level shadows the `client` submodule, so
# `import app.services.sarvam.client` would bind the instance rather than the
# module — which breaks any attempt to patch or reload it. Import it explicitly:
#     from app.services.sarvam.client import client
from app.services.sarvam.client import (
    SarvamClient,
    SarvamError,
    SarvamUnavailable,
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
    "detect_script",
    "identify_language",
    "looks_romanised_indic",
    "synthesize",
    "to_devanagari",
    "translate",
    "transliterate",
]
