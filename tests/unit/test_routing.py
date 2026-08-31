"""
Query routing — regression tests for audit E-01.

The bug: the prefix in the verse-reference pattern was optional and the separator
class contained whitespace, so any "N M" digit pair anywhere in a sentence routed
to direct_lookup and skipped the entire semantic pipeline. Users silently got one
arbitrary verse.

The first test class below is that bug. It must never regress.
"""

import pytest

from app.services.routing import (
    CHAPTER_LENGTHS,
    classify_query,
    has_devanagari,
    is_valid_reference,
)


class TestWhitespaceDigitPairsAreNotReferences:
    """E-01: these all used to hijack the pipeline."""

    @pytest.mark.parametrize("query", [
        "I only sleep 4 5 hours a night and I'm exhausted",
        "I have been meditating for 2 3 months with no result",
        "I work 9 5 and hate it",
        "we have 2 3 kids and no time for anything",
        "I've been in this job 6 7 years now",
        "I drink 8 10 cups of coffee to get through the day",
    ])
    def test_stays_semantic(self, query):
        route, meta = classify_query(query)
        assert route == "semantic", f"{query!r} hijacked to {meta}"


class TestDecimalsAreNotReferences:
    @pytest.mark.parametrize("query", [
        "I lost 2.5 kg this month and still feel awful",
        "my rating dropped to 3.2 and I am ashamed",
        "I earn 4.5 lakhs and still feel worthless",
    ])
    def test_bare_decimal_without_cue_is_semantic(self, query):
        assert classify_query(query)[0] == "semantic"


class TestGenuineReferences:
    @pytest.mark.parametrize("query,expected", [
        ("verse 2.47", "2.47"),
        ("BG 2.47", "2.47"),
        ("bg 18.66", "18.66"),
        ("gita 4.7", "4.7"),
        ("chapter 2 verse 47", "2.47"),
        ("chapter 2, verse 47", "2.47"),
        ("what does 18.66 mean", "18.66"),
        ("2.47", "2.47"),
        ("  2:47  ", "2.47"),
        ("explain 9.22 to me", "9.22"),
    ])
    def test_routes_to_direct_lookup(self, query, expected):
        route, meta = classify_query(query)
        assert route == "direct_lookup"
        assert meta["verse_id"] == expected


class TestReferenceRangeValidation:
    @pytest.mark.parametrize("query", [
        "bg 19.5",       # no chapter 19
        "verse 0.5",     # no chapter 0
        "verse 2.99",    # chapter 2 has 72 verses
        "chapter 12 verse 40",  # chapter 12 has 20 verses
    ])
    def test_impossible_reference_falls_back_to_semantic(self, query):
        assert classify_query(query)[0] == "semantic"

    def test_chapter_lengths_cover_all_18(self):
        assert sorted(CHAPTER_LENGTHS) == list(range(1, 19))

    def test_boundaries(self):
        assert is_valid_reference(18, 78)
        assert not is_valid_reference(18, 79)
        assert is_valid_reference(1, 1)
        assert not is_valid_reference(1, 0)


class TestSanskritRoute:
    def test_devanagari_routes_to_sanskrit(self):
        assert classify_query("कर्मण्येवाधिकारस्ते")[0] == "sanskrit"

    def test_detection(self):
        assert has_devanagari("धर्म")
        assert not has_devanagari("dharma")

    def test_reference_beats_script(self):
        # An explicit citation is cheaper and more precise than embedding.
        assert classify_query("verse 2.47 कर्मण्येवाधिकारस्ते")[0] == "direct_lookup"


class TestOrdinaryQueries:
    @pytest.mark.parametrize("query", [
        "I feel lost and anxious about my career",
        "how do I stop overthinking every decision",
        "what is the nature of the soul",
        "my father died and I cannot grieve properly",
    ])
    def test_semantic(self, query):
        assert classify_query(query)[0] == "semantic"
