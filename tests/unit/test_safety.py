"""
Crisis routing — tests for audit S-01.

The topical guardrail is deliberately generous about personal distress ("personal
struggles, existential questions, and moral dilemmas all count"), so a suicidal
query classified as *relevant* and the pipeline generated warm, personalised
guidance from verses about the body being temporary and the soul eternal.

These tests pin the two properties that matter: crisis queries never reach
retrieval, and ordinary philosophical questions about death still do.
"""

import pytest

from app.services import safety


class TestLexicalPrefilter:
    @pytest.mark.parametrize("query", [
        "I want to end my life",
        "I want to kill myself",
        "thinking about killing myself tonight",
        "I want to die",
        "I wish I was dead",
        "everyone would be better off without me",
        "I don't want to live anymore",
        "there is no reason to go on",
        "I have been hurting myself again",
        "I've been cutting myself",
        "I am going to hang myself",
        "I keep thinking about an overdose",
    ])
    def test_flags_crisis(self, query):
        assert safety.lexical_crisis_check(query) is True

    @pytest.mark.parametrize("query", [
        "what does the Gita say about death",
        "why does Krishna say the soul never dies",
        "my grandfather died and I feel numb",
        "Arjuna did not want to kill his relatives",
        "how do I deal with the death of a pet",
        "I feel like giving up on this project",
        "I am dead tired of my job",
    ])
    def test_does_not_flag_ordinary_queries(self, query):
        assert safety.lexical_crisis_check(query) is False


class TestClassifier:
    @pytest.mark.asyncio
    async def test_lexical_hit_short_circuits(self, stub_llms):
        # Even if the model says "safe", a lexical hit must win.
        stub_llms["safety_verdict"] = "safe"
        assert await safety.classify("I want to kill myself") == "crisis"

    @pytest.mark.asyncio
    async def test_model_can_flag_what_lexicon_misses(self, stub_llms):
        stub_llms["safety_verdict"] = "crisis"
        query = "I have made my peace and arranged everything for after I am gone"
        assert not safety.lexical_crisis_check(query)
        assert await safety.classify(query) == "crisis"

    @pytest.mark.asyncio
    async def test_api_failure_still_honours_lexicon(self, monkeypatch):
        broken = type("Broken", (), {})()
        broken.messages = type("M", (), {})()

        async def boom(**kwargs):
            raise RuntimeError("API down")

        broken.messages.create = boom
        monkeypatch.setattr(safety, "_client", broken)

        # Outage must not disable the offline prefilter...
        assert await safety.classify("I want to kill myself") == "crisis"
        # ...and must not block ordinary users either.
        assert await safety.classify("how do I find peace") == "safe"


class TestCrisisResponse:
    def test_is_a_fixed_string_not_generated(self):
        assert isinstance(safety.CRISIS_RESPONSE, str)
        assert len(safety.CRISIS_RESPONSE) > 100

    def test_carries_real_helplines(self):
        text = safety.CRISIS_RESPONSE
        assert "14416" in text          # Tele-MANAS, India
        assert "988" in text            # US
        assert "116 123" in text        # Samaritans
        assert "findahelpline.com" in text

    def test_does_not_offer_a_verse(self):
        lowered = safety.CRISIS_RESPONSE.lower()
        assert "verse" not in lowered or "with a verse" in lowered
