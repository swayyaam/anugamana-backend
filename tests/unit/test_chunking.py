"""
Purport chunking — the contract the indexer and the request path must share.

parent_start/parent_end offsets stored at index time only mean anything if the
request path reconstructs an identical chunk list. Both now import this module
(audit E-03, E-08).
"""

import pytest

from app.services.chunking import (
    MAX_CHUNK_WORDS,
    MIN_CHUNK_WORDS,
    chunk_purport,
    chunk_purport_cached,
    get_parent_window,
    split_sentences,
)


def paragraph(words: int, token: str = "word") -> str:
    return " ".join([token] * words)


class TestChunking:
    def test_empty_input(self):
        assert chunk_purport("") == []
        assert chunk_purport("   \n\n  ") == []

    def test_single_normal_paragraph(self):
        text = paragraph(100)
        assert len(chunk_purport(text)) == 1

    def test_short_paragraph_merges_forward(self):
        short = paragraph(MIN_CHUNK_WORDS - 10, "a")
        normal = paragraph(100, "b")
        chunks = chunk_purport(f"{short}\n\n{normal}")
        assert len(chunks) == 1
        assert "a" in chunks[0] and "b" in chunks[0]

    def test_oversized_paragraph_splits_at_sentences(self):
        sentence = "This is a sentence with several words in it. "
        chunks = chunk_purport(sentence * 120)
        assert len(chunks) > 1
        assert all(len(c.split()) <= MAX_CHUNK_WORDS for c in chunks)

    def test_trailing_short_paragraph_is_kept(self):
        """A short final paragraph has nothing to merge into and must survive."""
        chunks = chunk_purport(f"{paragraph(100, 'a')}\n\n{paragraph(5, 'b')}")
        assert any("b" in c for c in chunks)

    def test_deterministic(self):
        text = f"{paragraph(80, 'x')}\n\n{paragraph(90, 'y')}"
        assert chunk_purport(text) == chunk_purport(text)


class TestParentWindow:
    @pytest.mark.parametrize("n,idx,expected", [
        (5, 0, (0, 1)),
        (5, 2, (1, 3)),
        (5, 4, (3, 4)),
        (1, 0, (0, 0)),
    ])
    def test_window_is_clamped(self, n, idx, expected):
        assert get_parent_window([""] * n, idx) == expected

    def test_window_always_contains_the_child(self):
        chunks = [""] * 7
        for i in range(7):
            start, end = get_parent_window(chunks, i)
            assert start <= i <= end


class TestSentenceSplitting:
    def test_splits_on_terminators(self):
        assert len(split_sentences("One. Two! Three?")) == 3

    def test_drops_empties(self):
        assert all(s.strip() for s in split_sentences("A.  \n B."))


class TestCache:
    def test_returns_tuple_and_matches_uncached(self):
        text = f"{paragraph(80, 'x')}\n\n{paragraph(90, 'y')}"
        cached = chunk_purport_cached(text)
        assert isinstance(cached, tuple)
        assert list(cached) == chunk_purport(text)
