"""
Purport chunking — the single source of truth for how a purport is split.

Both the offline indexer (scripts/indexer.py) and the online RAG stage
(app/services/rag.py) import from here. They must agree exactly: the indexer
stores parent_start/parent_end offsets that only mean anything if the request
path reconstructs the same chunk list.

Chunking is paragraph-based rather than fixed-width because Prabhupada's
paragraphs are already semantic units:
    < MIN_CHUNK_WORDS   → merged forward into the next paragraph
    within bounds       → one chunk
    > MAX_CHUNK_WORDS   → split at sentence boundaries into ~TARGET_SPLIT_WORDS
"""

import re
from functools import lru_cache

MIN_CHUNK_WORDS = 40
MAX_CHUNK_WORDS = 350
TARGET_SPLIT_WORDS = 200


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def chunk_purport(purport: str) -> list[str]:
    if not purport or not purport.strip():
        return []

    raw_paragraphs = [p.strip() for p in purport.split("\n\n") if p.strip()]

    # Merge short paragraphs forward
    merged: list[str] = []
    i = 0
    while i < len(raw_paragraphs):
        para = raw_paragraphs[i]
        if len(para.split()) < MIN_CHUNK_WORDS and i + 1 < len(raw_paragraphs):
            raw_paragraphs[i + 1] = para + "\n\n" + raw_paragraphs[i + 1]
            i += 1
            continue
        merged.append(para)
        i += 1

    # Split oversized paragraphs at sentence boundaries
    chunks: list[str] = []
    for para in merged:
        if len(para.split()) <= MAX_CHUNK_WORDS:
            chunks.append(para)
            continue

        current: list[str] = []
        current_words = 0
        for sent in split_sentences(para):
            sent_words = len(sent.split())
            if current_words + sent_words > TARGET_SPLIT_WORDS and current:
                chunks.append(" ".join(current))
                current = [sent]
                current_words = sent_words
            else:
                current.append(sent)
                current_words += sent_words
        if current:
            chunks.append(" ".join(current))

    return chunks


def get_parent_window(chunks: list[str], child_idx: int) -> tuple[int, int]:
    """The +/-1 paragraph window around a retrieved child chunk."""
    return max(0, child_idx - 1), min(len(chunks) - 1, child_idx + 1)


@lru_cache(maxsize=1024)
def chunk_purport_cached(purport: str) -> tuple[str, ...]:
    """
    Memoised chunking for the request path — the same verse is re-chunked on
    every generation otherwise. Returns a tuple so it stays hashable/immutable.
    """
    return tuple(chunk_purport(purport))
