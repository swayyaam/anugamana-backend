#!/usr/bin/env python3
"""
Build a second corpus: Marcus Aurelius, *Meditations*.

Why a second corpus
-------------------
A result on one corpus is a case study. The same result on a second, from a
different tradition and a different source language, is a method — and it is the
difference between "we built a Gita search engine" and "here is how to retrieve
across a register gap, demonstrated twice".

It also resolves the licensing bind. The Gita translations and purports are
Bhaktivedanta Book Trust copyright, so the enriched corpus cannot be released.
*Meditations* (Casaubon translation, Project Gutenberg) is public domain, so this
half of the artifact is fully releasable: corpus, enrichment, index and
benchmark, end to end reproducible by anyone.

Why this corpus specifically
----------------------------
It has the property under study. A reader arrives with "my colleague took credit
for my work and I can't stop replaying it"; the text says "Begin the morning by
saying to thyself, I shall meet with the busy-body, the ungrateful, arrogant,
deceitful". Same register gap, same absence of shared vocabulary, entirely
different tradition — so any effect that replicates here is not an artifact of
Prabhupada's prose style.

Stages (each resumable):
    --fetch     download and segment into passages
    --enrich    generate the same four meaning_fields per passage
    --index     build enriched and raw indexes for the ablation grid

Usage:
    python scripts/build_second_corpus.py --fetch
    python scripts/build_second_corpus.py --enrich
    python scripts/build_second_corpus.py --index
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests  # noqa: E402
from anthropic import AsyncAnthropic  # noqa: E402

from app.config import ANTHROPIC_API_KEY, DATA_DIR, LLM_MODEL  # noqa: E402

CORPUS_DIR = DATA_DIR / "corpora" / "meditations"
RAW_TEXT = CORPUS_DIR / "raw.txt"
PASSAGES_FILE = CORPUS_DIR / "passages.json"
ENRICHED_FILE = CORPUS_DIR / "enriched.json"
CHROMA_DIR = CORPUS_DIR / "chroma"
RAW_CHROMA_DIR = CORPUS_DIR / "chroma_raw"
SPARSE_FILE = CORPUS_DIR / "sparse_index.pkl"
RAW_SPARSE_FILE = CORPUS_DIR / "sparse_index_raw.pkl"

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/2680/pg2680.txt"
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"

BOOK_RE = re.compile(r"^THE\s+(\w+)\s+BOOK\s*$", re.MULTILINE)
PASSAGE_RE = re.compile(r"^([IVXLC]+)\.\s+", re.MULTILINE)

ORDINALS = {
    "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5, "SIXTH": 6,
    "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9, "TENTH": 10, "ELEVENTH": 11,
    "TWELFTH": 12,
}

BOOK_THEMES = {
    1: "debts and lessons — what he learned from each person who shaped him",
    2: "mortality and the discipline of the present moment",
    3: "the value of time and the integrity of the ruling faculty",
    4: "retreat into oneself; opinion as the source of disturbance",
    5: "duty, rising to work, and acting according to nature",
    6: "acceptance of the whole; kindness toward those who wrong you",
    7: "endurance, perspective on pain, and the shortness of fame",
    8: "correcting mistakes; the self-sufficiency of a good character",
    9: "justice, truthfulness, and dying without complaint",
    10: "simplicity, wholeness, and seeing things as they actually are",
    11: "the nature of the rational soul; dealing with offence and anger",
    12: "the final accounting — living the remainder well, letting go",
}

client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# The same four fields as the Gita enrichment. Keeping the schema identical is
# what makes the cross-corpus comparison a comparison rather than two studies.
ENRICHMENT_TOOL = {
    "name": "generate_meaning_fields",
    "description": "Generate four semantic enrichment fields for a passage.",
    "input_schema": {
        "type": "object",
        "properties": {
            "situations": {"type": "string", "description":
                "3-5 specific real-world situations a modern person faces when "
                "this passage becomes relevant. Concrete life scenarios in "
                "everyday language."},
            "teaching": {"type": "string", "description":
                "The core insight in plain modern English, 2-4 sentences. State "
                "it directly; never begin with 'This passage teaches'."},
            "emotions": {"type": "string", "description":
                "The emotional states this passage directly addresses, in "
                "everyday language, no philosophical jargon."},
            "concepts": {"type": "string", "description":
                "The philosophical concepts involved. Greek or Stoic terms are "
                "allowed but must be explained in plain English first."},
        },
        "required": ["situations", "teaching", "emotions", "concepts"],
    },
}

ENRICHMENT_SYSTEM = """\
You are an expert at bridging ancient philosophy and the language of modern life.

For each passage from Marcus Aurelius' Meditations, generate four semantic fields
that will power vector search. A modern person will type something like "my
colleague took credit for my work and I can't stop replaying it" — these fields
must let the system match that query to the right passage.

Rules for every field:
- Use the everyday language of someone describing a real problem to a friend
- Be specific and concrete, never generic self-help
- situations and emotions must contain no philosophical jargon
- Ground everything strictly in what this passage actually says
- Never invent meaning that is not in the text\
"""


# ---------------------------------------------------------------------------
# Fetch and segment
# ---------------------------------------------------------------------------

def fetch() -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_TEXT.exists():
        text = RAW_TEXT.read_text(encoding="utf-8")
        print(f"using cached {RAW_TEXT.name} ({len(text):,} chars)")
    else:
        print(f"downloading {GUTENBERG_URL}")
        response = requests.get(GUTENBERG_URL, timeout=60)
        response.raise_for_status()
        text = response.text
        RAW_TEXT.write_text(text, encoding="utf-8")
        print(f"saved {len(text):,} chars")

    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1:
        print("could not locate Gutenberg content markers")
        return 1
    body = text[text.find("\n", start) : end]

    # Split into books, then into numbered passages within each book.
    book_positions = [(m.start(), m.group(1).upper()) for m in BOOK_RE.finditer(body)]
    if not book_positions:
        print("no book headers found")
        return 1

    # Gutenberg appends editorial apparatus — the Fronto correspondence, notes
    # and a glossary — after the twelfth book. Without this cut it is absorbed
    # into the final passage, which then arrives as a 7,149-word "meditation"
    # and poisons both the enrichment and the index.
    #
    # The search must start after the last book header: "APPENDIX" also appears
    # in the table of contents near the top of the file, and trimming there
    # deletes the entire work.
    last_book_start = book_positions[-1][0]
    apparatus = re.search(r"\bAPPENDIX\b", body[last_book_start:])
    if apparatus:
        cut = last_book_start + apparatus.start()
        print(f"  trimmed editorial apparatus at char {cut:,} "
              f"({len(body) - cut:,} chars removed)")
        body = body[:cut]

    passages = []
    for index, (position, ordinal) in enumerate(book_positions):
        book_number = ORDINALS.get(ordinal)
        if book_number is None:
            continue
        stop = (
            book_positions[index + 1][0]
            if index + 1 < len(book_positions)
            else len(body)
        )
        chunk = body[position:stop]

        marks = list(PASSAGE_RE.finditer(chunk))
        for j, mark in enumerate(marks):
            tail = marks[j + 1].start() if j + 1 < len(marks) else len(chunk)
            passage = chunk[mark.end() : tail].strip()
            passage = re.sub(r"\s*\n\s*", " ", passage).strip()
            if len(passage.split()) < 12:
                continue  # fragments and headers, not passages
            passages.append({
                "verse_id": f"{book_number}.{j + 1}",
                "chapter": book_number,
                "verse": j + 1,
                "devanagari": "",
                "sanskrit": "",
                "translation": passage,
                "purport": "",
                "book_theme": BOOK_THEMES.get(book_number, ""),
            })

    PASSAGES_FILE.write_text(
        json.dumps(passages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lengths = sorted(len(p["translation"].split()) for p in passages)
    print(f"\nsegmented {len(passages)} passages across "
          f"{len({p['chapter'] for p in passages})} books -> {PASSAGES_FILE.name}")
    print(f"  words per passage: min={lengths[0]} "
          f"median={lengths[len(lengths) // 2]} max={lengths[-1]}")
    print(f"  sample [{passages[0]['verse_id']}]: {passages[0]['translation'][:100]}...")
    return 0


# ---------------------------------------------------------------------------
# Enrich
# ---------------------------------------------------------------------------

async def enrich_one(passage: dict, semaphore: asyncio.Semaphore) -> dict | None:
    async with semaphore:
        message = (
            f"Book {passage['chapter']} — {passage['book_theme']}\n\n"
            f"Passage {passage['verse_id']}:\n{passage['translation']}"
        )
        try:
            response = await client.messages.create(
                model=LLM_MODEL,
                max_tokens=1500,
                system=ENRICHMENT_SYSTEM,
                tools=[ENRICHMENT_TOOL],
                tool_choice={"type": "tool", "name": "generate_meaning_fields"},
                messages=[{"role": "user", "content": message}],
            )
            block = next(b for b in response.content if b.type == "tool_use")
            fields = block.input
        except Exception as e:
            print(f"\n  {passage['verse_id']} failed: {e}")
            return None

    return {
        **passage,
        "meaning_fields": fields,
        "text_for_embedding": "\n".join([
            fields["situations"], fields["teaching"],
            fields["emotions"], fields["concepts"],
        ]),
    }


async def enrich(concurrency: int) -> int:
    if not PASSAGES_FILE.exists():
        print("run --fetch first")
        return 1

    passages = json.loads(PASSAGES_FILE.read_text(encoding="utf-8"))
    done = {}
    if ENRICHED_FILE.exists():
        done = {p["verse_id"]: p for p in json.loads(ENRICHED_FILE.read_text())}
        print(f"resuming — {len(done)} already enriched")

    todo = [p for p in passages if p["verse_id"] not in done]
    print(f"enriching {len(todo)} passages (concurrency {concurrency})")

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    results = list(done.values())

    for start in range(0, len(todo), 40):
        batch = todo[start : start + 40]
        enriched = await asyncio.gather(
            *(enrich_one(p, semaphore) for p in batch)
        )
        results.extend(p for p in enriched if p)
        completed += len(batch)
        results.sort(key=lambda p: (p["chapter"], p["verse"]))
        ENRICHED_FILE.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\r  {completed}/{len(todo)}  (saved {len(results)})", end="", flush=True)

    print(f"\nenriched {len(results)} passages -> {ENRICHED_FILE.name}")
    return 0


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def index() -> int:
    import chromadb
    import pickle
    from FlagEmbedding import BGEM3FlagModel
    from tqdm import tqdm

    from app.config import EMBEDDING_MODEL

    if not ENRICHED_FILE.exists():
        print("run --enrich first")
        return 1

    passages = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
    print(f"indexing {len(passages)} passages")
    model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)

    # Enriched and raw indexes, built identically apart from the text embedded —
    # the same single-factor contrast the Gita conditions C2 vs C5 rest on.
    for label, chroma_dir, sparse_file, field, doc_type in (
        ("enriched", CHROMA_DIR, SPARSE_FILE, "text_for_embedding", "meaning"),
        ("raw", RAW_CHROMA_DIR, RAW_SPARSE_FILE, "translation", "translation"),
    ):
        client_db = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client_db.get_or_create_collection(
            f"meditations_{label}", metadata={"hnsw:space": "cosine"}
        )
        sparse_index: dict[str, dict[str, float]] = {}

        for start in tqdm(range(0, len(passages), 32), desc=label):
            batch = passages[start : start + 32]
            texts = [p[field] for p in batch]
            out = model.encode(
                texts, batch_size=32, max_length=512,
                return_dense=True, return_sparse=True,
            )
            ids = [f"{p['verse_id']}_{doc_type}" for p in batch]
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=[{
                    "verse_id": p["verse_id"], "chapter": int(p["chapter"]),
                    "verse": int(p["verse"]), "translation": p["translation"],
                    "devanagari": "", "sanskrit": "", "type": doc_type,
                } for p in batch],
                embeddings=[v.tolist() for v in out["dense_vecs"]],
            )
            for doc_id, weights in zip(ids, out["lexical_weights"]):
                for token_id, weight in weights.items():
                    sparse_index.setdefault(str(token_id), {})[doc_id] = float(weight)

        sparse_file.write_bytes(pickle.dumps(sparse_index))
        print(f"  {label}: {collection.count()} vectors, "
              f"{len(sparse_index)} sparse tokens")

    print(f"\nDone. Register these indexes in eval/conditions.py to run the grid "
          f"on Meditations.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--enrich", action="store_true")
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    if args.fetch:
        return fetch()
    if args.enrich:
        return asyncio.run(enrich(args.concurrency))
    if args.index:
        return index()
    parser.error("choose --fetch, --enrich or --index")


if __name__ == "__main__":
    raise SystemExit(main())
