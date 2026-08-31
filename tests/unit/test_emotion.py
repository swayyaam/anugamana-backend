"""
Affective classification used as a retrieval signal.

The taxonomy is a research artifact, not a UI enum: every label is grounded in
the corpus's own affective vocabulary and anchored to a Sanskrit term from the
text. These tests pin that contract so a casual edit cannot quietly change what
condition C12 is measuring.
"""

import pytest

from app.services import emotion


class TestTaxonomy:
    def test_every_label_is_anchored_to_the_text(self):
        for key, label in emotion.TAXONOMY.items():
            assert label.key == key
            assert label.gita_term, f"{key} has no Sanskrit anchor"
            assert label.gloss
            assert len(label.probe.split()) >= 5, f"{key} probe is too thin to embed"

    def test_covers_the_corpus_dominant_states(self):
        """
        The most frequent affective terms across the 700 enriched `emotions`
        fields: anxiety 278, frustration 239, exhaustion 163, fear 160,
        shame 133, despair 131, doubt 115, confusion 108.
        """
        for label in (
            "anxiety", "frustration", "exhaustion", "fear",
            "shame", "despair", "doubt", "confusion",
        ):
            assert label in emotion.TAXONOMY

    def test_includes_a_non_distress_label(self):
        """Philosophical questions must not be forced into a distress bucket."""
        assert "seeking" in emotion.TAXONOMY

    def test_includes_the_state_the_text_points_toward(self):
        assert emotion.TAXONOMY["equanimity"].gita_term == "samatva"


class TestParsing:
    def test_valid_payload(self):
        result = emotion._build(
            {"primary": "despair", "secondary": ["exhaustion"], "intensity": 4}, "test"
        )
        assert result.primary == "despair"
        assert result.secondary == ["exhaustion"]
        assert result.intensity == 4
        assert result.detected

    def test_unknown_primary_label_is_rejected(self):
        with pytest.raises(ValueError, match="taxonomy"):
            emotion._build({"primary": "ennui"}, "test")

    def test_unknown_secondary_labels_are_dropped(self):
        result = emotion._build(
            {"primary": "fear", "secondary": ["ennui", "doubt"]}, "test"
        )
        assert result.secondary == ["doubt"]

    def test_secondary_is_capped(self):
        result = emotion._build(
            {"primary": "fear", "secondary": ["doubt", "shame", "anger", "grief"]},
            "test",
        )
        assert len(result.secondary) == 2

    def test_intensity_is_clamped(self):
        assert emotion._build({"primary": "fear", "intensity": 99}, "t").intensity == 5
        assert emotion._build({"primary": "fear", "intensity": -3}, "t").intensity == 1

    def test_tolerates_prose_around_the_json(self):
        payload = emotion._parse(
            'Here is my answer:\n```json\n{"primary": "grief"}\n```\nHope that helps.'
        )
        assert payload["primary"] == "grief"

    def test_missing_json_raises(self):
        with pytest.raises(ValueError, match="no JSON"):
            emotion._parse("I could not classify this.")


class TestProbeText:
    def test_undetected_yields_no_probe(self):
        assert emotion.EmotionResult().probe_text() == ""

    def test_combines_primary_and_secondary(self):
        result = emotion._build(
            {"primary": "despair", "secondary": ["exhaustion"]}, "test"
        )
        probe = result.probe_text()
        assert emotion.TAXONOMY["despair"].probe in probe
        assert emotion.TAXONOMY["exhaustion"].probe in probe

    def test_probe_is_in_the_register_of_the_enrichment(self):
        """
        Probes are embedded against the enrichment's `emotions` field, so they
        must read like it — plain modern language, no Sanskrit.
        """
        for label in emotion.TAXONOMY.values():
            assert label.gita_term not in label.probe


class TestBackendSelection:
    @pytest.mark.asyncio
    async def test_records_which_backend_produced_the_label(self, stub_llms):
        result = await emotion.classify("I have nothing left to give", prefer="claude")
        assert result.backend == "claude"
        assert result.primary == "despair"

    @pytest.mark.asyncio
    async def test_falls_back_when_sarvam_is_unconfigured(self, stub_llms):
        """auto with no Sarvam key must still classify, via Claude."""
        result = await emotion.classify("I have nothing left to give")
        assert result.detected
        assert result.backend == "claude"

    @pytest.mark.asyncio
    async def test_total_failure_yields_an_undetected_result(
        self, stub_llms, monkeypatch
    ):
        # stub_llms forces Sarvam offline; here we break the remaining backend
        # too, so every route to a label is unavailable.
        class Broken:
            messages = type("M", (), {})()

        async def boom(**kwargs):
            raise RuntimeError("down")

        broken = Broken()
        broken.messages.create = boom
        monkeypatch.setattr(emotion, "_claude", broken)

        result = await emotion.classify("anything")
        assert not result.detected
        assert result.backend == "none"
        assert result.probe_text() == ""

    @pytest.mark.asyncio
    async def test_malformed_output_does_not_crash_the_pipeline(
        self, stub_llms, monkeypatch
    ):
        stub_llms["emotion_json"] = "I think they are sad."
        result = await emotion.classify("I am sad")
        assert not result.detected
