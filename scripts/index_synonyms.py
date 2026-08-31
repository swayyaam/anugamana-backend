#!/usr/bin/env python3
"""
Add the word-by-word Sanskrit glosses to the index.

Every verse in the corpus carries a `synonyms` field — Prabhupada's word-by-word
breakdown:

    karmaṇi — in prescribed duties; eva — certainly; adhikāraḥ — right;
    te — of you; mā — never; phaleṣu — in the fruits; kadācana — at any time

It is present on all 700 verses and indexed nowhere. That is free signal, and it
is aimed squarely at the query types the pipeline currently handles worst:
Sanskrit typed in Devanagari or Roman script, and half-remembered phrases like
"karmanye vadhikaraste". Those queries have almost no surface overlap with an
English translation, but very high overlap with this field.

This adds a `{verse_id}_synonyms` vector to the existing enriched collection
rather than rebuilding it, and extends the sparse index in place. Whether it
actually helps is condition C17 — it is not switched on for the served pipeline
until it has earned that.

Usage:
    python scripts/index_synonyms.py
    python scripts/index_synonyms.py --dry-run
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb  # noqa: E402
from FlagEmbedding import BGEM3FlagModel  # noqa: E402
from tqdm import tqdm  # noqa: E402

from app.config import EMBEDDING_MODEL, ENRICHED_FILE  # noqa: E402
from app.services.retrieval import ENRICHED_INDEX  # noqa: E402

EMBED_BATCH = 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    verses = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
    with_synonyms = [v for v in verses if (v.get("synonyms") or "").strip()]

    lengths = sorted(len(v["synonyms"].split()) for v in with_synonyms)
    print(f"{len(with_synonyms)}/{len(verses)} verses have a synonyms field")
    if lengths:
        print(f"  words: min={lengths[0]} median={lengths[len(lengths) // 2]} "
              f"max={lengths[-1]}")
        print(f"  sample: {with_synonyms[0]['synonyms'][:110]}...")

    if args.dry_run:
        return 0
    if not with_synonyms:
        print("nothing to index")
        return 1

    client = chromadb.PersistentClient(path=str(ENRICHED_INDEX.chroma_dir))
    collection = client.get_collection(ENRICHED_INDEX.verses_collection)

    existing = set(collection.get(include=[])["ids"])
    todo = [
        v for v in with_synonyms
        if f"{v['verse_id']}_synonyms" not in existing
    ]
    print(f"\n{len(todo)} to embed, {len(with_synonyms) - len(todo)} already present")
    if not todo:
        print("already indexed")
        return 0

    print("loading BGE-M3...")
    model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)

    sparse_index = pickle.loads(ENRICHED_INDEX.sparse_file.read_bytes())
    print(f"sparse index has {len(sparse_index)} tokens before")

    for start in tqdm(range(0, len(todo), EMBED_BATCH), desc="synonyms"):
        batch = todo[start : start + EMBED_BATCH]
        # Prefix with the verse reference so the vector carries a little
        # context; the glosses alone are a bag of Sanskrit fragments.
        texts = [
            f"Verse {v['verse_id']} word meanings: {v['synonyms']}" for v in batch
        ]
        out = model.encode(
            texts, batch_size=EMBED_BATCH, max_length=512,
            return_dense=True, return_sparse=True,
        )

        ids = [f"{v['verse_id']}_synonyms" for v in batch]
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=[{
                "verse_id": v["verse_id"],
                "chapter": int(v["chapter"]),
                "verse": int(v["verse"]),
                "devanagari": v.get("devanagari", ""),
                "sanskrit": v.get("sanskrit", ""),
                "translation": v["translation"],
                "type": "synonyms",
            } for v in batch],
            embeddings=[vec.tolist() for vec in out["dense_vecs"]],
        )
        for doc_id, weights in zip(ids, out["lexical_weights"]):
            for token_id, weight in weights.items():
                sparse_index.setdefault(str(token_id), {})[doc_id] = float(weight)

    ENRICHED_INDEX.sparse_file.write_bytes(pickle.dumps(sparse_index))

    print(f"\nDone.")
    print(f"  collection now holds {collection.count()} vectors")
    print(f"  sparse index has {len(sparse_index)} tokens")
    print(f"\nMeasure it before serving it:")
    print(f"  python -m eval.run --conditions C17")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
