"""
Sarvam integration.

Two properties matter most and are tested hardest:

1. **Absence is a supported state.** With no API key every entry point degrades
   to a defined fallback. Indic support must never be a hard dependency of
   English search.
2. **Schema drift fails loudly.** Sarvam's response field names are not pinned by
   this codebase, so `pick()` must raise a legible error naming what it actually
   received rather than yielding None into a research result.
"""

import httpx
import pytest

from app.services.sarvam import text as sarvam_text
from app.services.sarvam.client import (
    SarvamClient,
    SarvamError,
    SarvamUnavailable,
    pick,
)


class TestSchemaTolerance:
    def test_accepts_any_documented_key(self):
        assert pick({"translated_text": "a"}, "translated_text", "output") == "a"
        assert pick({"output": "b"}, "translated_text", "output") == "b"

    def test_prefers_the_first_candidate(self):
        payload = {"output": "second", "translated_text": "first"}
        assert pick(payload, "translated_text", "output") == "first"

    def test_missing_key_names_what_was_received(self):
        with pytest.raises(SarvamError) as excinfo:
            pick({"unexpected": 1, "other": 2}, "translated_text", "output")
        message = str(excinfo.value)
        assert "translated_text" in message
        assert "unexpected" in message and "other" in message

    def test_optional_key_returns_none(self):
        assert pick({}, "language_code", required=False) is None

    def test_null_value_is_treated_as_absent(self):
        assert pick({"output": None, "text": "x"}, "output", "text") == "x"


class TestUnconfiguredClient:
    @pytest.mark.asyncio
    async def test_reports_unavailable(self):
        client = SarvamClient(api_key=None)
        assert client.available is False
        with pytest.raises(SarvamUnavailable):
            await client.post("/translate", {})

    @pytest.mark.asyncio
    async def test_translate_returns_input_unchanged(self, monkeypatch):
        monkeypatch.setattr(sarvam_text, "client", SarvamClient(api_key=None))
        assert await sarvam_text.translate("मैं थक गया हूँ", "hi-IN", "en-IN") == "मैं थक गया हूँ"

    @pytest.mark.asyncio
    async def test_language_id_falls_back_to_script_then_default(self, monkeypatch):
        monkeypatch.setattr(sarvam_text, "client", SarvamClient(api_key=None))
        # Script detection is offline and still works.
        assert await sarvam_text.identify_language("मैं थक गया") == ("hi-IN", "script")
        # Latin script with no API access defaults to the pivot.
        assert await sarvam_text.identify_language("I am tired") == ("en-IN", "default")


class TestScriptDetection:
    @pytest.mark.parametrize("text,expected", [
        ("कर्मण्येवाधिकारस्ते", "hi-IN"),
        ("আমি ক্লান্ত", "bn-IN"),
        ("நான் சோர்வாக", "ta-IN"),
        ("నేను అలసిపోయాను", "te-IN"),
        ("ನಾನು ಸುಸ್ತಾಗಿದ್ದೇನೆ", "kn-IN"),
        ("ഞാൻ ക്ഷീണിതനാണ്", "ml-IN"),
        ("હું થાકી ગયો છું", "gu-IN"),
        ("I am exhausted", None),
    ])
    def test_detects_script(self, text, expected):
        assert sarvam_text.detect_script(text) == expected


class TestRomanisedDetection:
    @pytest.mark.parametrize("query", [
        "what is karma yoga",
        "explain the atman",
        "nishkama karma meaning",
        "how does prakriti bind the jiva",
    ])
    def test_recognises_romanised_sanskrit(self, query):
        assert sarvam_text.looks_romanised_indic(query) is True

    @pytest.mark.parametrize("query", [
        "I keep failing at work and feel like giving up",
        "how do I stop overthinking",
    ])
    def test_ignores_plain_english(self, query):
        assert sarvam_text.looks_romanised_indic(query) is False


class TestTransport:
    @pytest.mark.asyncio
    async def test_successful_translation(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["api-subscription-key"] == "test-key"
            return httpx.Response(200, json={"translated_text": "I am tired"})

        client = SarvamClient(api_key="test-key")
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.sarvam.ai",
            headers={"api-subscription-key": "test-key"},
        )
        monkeypatch.setattr(sarvam_text, "client", client)
        assert await sarvam_text.translate("मैं थक गया", "hi-IN", "en-IN") == "I am tired"

    @pytest.mark.asyncio
    async def test_client_error_is_not_retried_and_degrades(self, monkeypatch):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(400, text="bad language code")

        client = SarvamClient(api_key="k")
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://api.sarvam.ai"
        )
        monkeypatch.setattr(sarvam_text, "client", client)
        # translate() swallows the error and returns the input unchanged.
        assert await sarvam_text.translate("hello", "en-IN", "hi-IN") == "hello"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_identical_languages_short_circuit(self, monkeypatch):
        def handler(request):  # pragma: no cover - must never be called
            raise AssertionError("no request should be made")

        client = SarvamClient(api_key="k")
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://api.sarvam.ai"
        )
        monkeypatch.setattr(sarvam_text, "client", client)
        assert await sarvam_text.translate("hello", "en-IN", "en-IN") == "hello"
