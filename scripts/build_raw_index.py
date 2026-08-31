#!/usr/bin/env python3
"""
Build the UNENRICHED control index.

This is the artifact the retracted evaluation did not have, and without which the
project's central claim cannot be measured. Every condition in the old harness
retrieved over `_meaning` vectors, and `_meaning` vectors *are* the enrichment —
so the condition called "baseline" was "sparse retrieval over enriched text".
The effect of enrichment had never been isolated.

This index is constructed identically to the enriched one — same embedding model,
same chunking, same fusion-ready id scheme — differing in exactly one factor:
verse vectors are built from the raw translation instead of `text_for_embedding`.
That single-factor difference is what makes C2 vs C5 an attributable comparison.

Produces:
    data/chroma_raw/          raw_verses  ({vid}_translation)
                              raw_purport ({vid}_purport_N)
    data/sparse_index_raw.pkl BGE-M3 lexical weights over the same documents

Usage:
    python scripts/build_raw_index.py
    python scripts/build_raw_index.py --reset
"""

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import chromadb
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import EMBEDDING_MODEL, ENRICHED_FILE  # noqa: E402
from app.services.chunking import chunk_purport, get_parent_window  # noqa: E402
from app.services.retrieval import RAW_CHROMA_DIR, RAW_SPARSE_FILE  # noqa: E402

EMBED_BATCH = 32
MAX_LENGTH = 512


def embed(model, texts):
    out = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
    )
    return out["dense_vecs"], out["lexical_weights"]


def update_sparse(index, doc_id, weights):
    for token_id, weight in weights.items():
        index.setdefault(str(token_id), {})[doc_id] = float(weight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if not ENRICHED_FILE.exists():
        print(f"ERROR: {ENRICHED_FILE} not found.")
        return 1

    if args.reset:
        shutil.rmtree(RAW_CHROMA_DIR, ignore_errors=True)
        RAW_SPARSE_FILE.unlink(missing_ok=True)
        print("wiped existing raw index")

    verses = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(verses)} verses (raw fields only — enrichment ignored).")

    print("Loading BGE-M3...")
    model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)

    client = chromadb.PersistentClient(path=str(RAW_CHROMA_DIR))
    verses_col = client.get_or_create_collection(
        "raw_verses", metadata={"hnsw:space": "cosine"}
    )
    purport_col = client.get_or_create_collection(
        "raw_purport", metadata={"hnsw:space": "cosine"}
    )

    sparse_index: dict[str, dict[str, float]] = {}
    if RAW_SPARSE_FILE.exists() and not args.reset:
        sparse_index = pickle.loads(RAW_SPARSE_FILE.read_bytes())
        print(f"resuming with {len(sparse_index)} sparse tokens")

    done_verses = set(verses_col.get(include=[])["ids"])
    done_purport = set(purport_col.get(include=[])["ids"])

    # --- verse-level vectors: TRANSLATION ONLY -----------------------------
    todo = [v for v in verses if f"{v['verse_id']}_translation" not in done_verses]
    print(f"\nVerse vectors: {len(todo)} to build, {len(verses) - len(todo)} done.")

    for start in tqdm(range(0, len(todo), EMBED_BATCH), desc="raw verses"):
        batch = todo[start : start + EMBED_BATCH]
        texts = [v["translation"] for v in batch]
        dense, sparse = embed(model, texts)

        ids, docs, metas, embeddings = [], [], [], []
        for offset, verse in enumerate(batch):
            vid = verse["verse_id"]
            doc_id = f"{vid}_translation"
            ids.append(doc_id)
            docs.append(texts[offset])
            metas.append({
                "verse_id": vid,
                "chapter": int(verse["chapter"]),
                "verse": int(verse["verse"]),
                "devanagari": verse.get("devanagari", ""),
                "sanskrit": verse.get("sanskrit", ""),
                "translation": verse["translation"],
                "type": "translation",
            })
            embeddings.append(dense[offset].tolist())
            update_sparse(sparse_index, doc_id, sparse[offset])

        if ids:
            verses_col.add(ids=ids, documents=docs, metadatas=metas,
                           embeddings=embeddings)

    # --- purport chunks: identical chunking to the enriched index ----------
    print("\nPurport chunks...")
    records = []
    for verse in verses:
        vid = verse["verse_id"]
        header = f"Verse {vid}: {verse['translation']}"
        chunks = chunk_purport(verse.get("purport", ""))
        for index, chunk in enumerate(chunks):
            doc_id = f"{vid}_purport_{index}"
            if doc_id in done_purport:
                continue
            parent_start, parent_end = get_parent_window(chunks, index)
            records.append((doc_id, f"{header}\n\n{chunk}", {
                "verse_id": vid,
                "chapter": int(verse["chapter"]),
                "verse": int(verse["verse"]),
                "type": "purport_chunk",
                "chunk_index": index,
                "parent_start": parent_start,
                "parent_end": parent_end,
                "translation": verse["translation"],
            }))

    print(f"{len(records)} purport chunks to build.")
    for start in tqdm(range(0, len(records), EMBED_BATCH), desc="raw purport"):
        batch = records[start : start + EMBED_BATCH]
        texts = [r[1] for r in batch]
        dense, sparse = embed(model, texts)
        for offset, (doc_id, _, _) in enumerate(batch):
            update_sparse(sparse_index, doc_id, sparse[offset])
        purport_col.add(
            ids=[r[0] for r in batch],
            documents=texts,
            metadatas=[r[2] for r in batch],
            embeddings=[d.tolist() for d in dense],
        )

    RAW_SPARSE_FILE.write_bytes(pickle.dumps(sparse_index))

    print(f"\nDone.")
    print(f"  raw_verses   {verses_col.count()}")
    print(f"  raw_purport  {purport_col.count()}")
    print(f"  sparse index {len(sparse_index)} tokens -> {RAW_SPARSE_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
