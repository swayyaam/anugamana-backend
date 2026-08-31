#!/usr/bin/env python3
"""
Phase 3: Index all 700 enriched verses into ChromaDB + sparse index.

Reads data/gita_enriched.json and builds:
  data/chroma_db/          — ChromaDB with two collections:
    gita_verses            — meaning vectors + translation vectors (1 per verse × 2)
    gita_purport           — paragraph-level purport chunk vectors
  data/sparse_index.pkl    — BGE-M3 lexical weights for keyword retrieval

Per verse, 3 vector types are produced:
  {verse_id}_meaning      → dense(text_for_embedding)       catches emotional/semantic queries
  {verse_id}_translation  → dense(translation)              catches paraphrase queries
  {verse_id}_purport_N    → dense(header + paragraph_N)     catches commentary queries

Purport chunking is paragraph-based (semantic) not fixed-size:
  < 40 words  → merge with next paragraph
  40–350 words → one child chunk
  > 350 words → split at sentence boundaries into ~200-word sub-chunks

Parent-child retrieval: child chunk retrieved, parent window (±1 paragraph) sent to Claude.

Usage:
    source venv/bin/activate
    python scripts/indexer.py
    python scripts/indexer.py --reset   # wipe and rebuild from scratch
"""

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import chromadb
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.services.chunking import chunk_purport, get_parent_window  # noqa: E402

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
INPUT_FILE = DATA_DIR / "gita_enriched.json"
CHROMA_DIR = DATA_DIR / "chroma_db"
SPARSE_FILE = DATA_DIR / "sparse_index.pkl"

EMBED_BATCH = 32
MAX_LENGTH = 512



# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(
    model: BGEM3FlagModel,
    texts: list[str],
) -> tuple[np.ndarray, list[dict]]:
    output = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
    )
    return output["dense_vecs"], output["lexical_weights"]


# ---------------------------------------------------------------------------
# Sparse index
# ---------------------------------------------------------------------------

def update_sparse_index(
    sparse_index: dict[str, dict[str, float]],
    doc_id: str,
    lexical_weights: dict,
) -> None:
    for token_id, weight in lexical_weights.items():
        token_key = str(token_id)
        if token_key not in sparse_index:
            sparse_index[token_key] = {}
        sparse_index[token_key][doc_id] = float(weight)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Build ChromaDB + sparse index")
    parser.add_argument("--reset", action="store_true", help="Wipe existing index and rebuild")
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found — run scripts/enrich.py first")
        sys.exit(1)

    if args.reset:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
            print("Wiped chroma_db/")
        if SPARSE_FILE.exists():
            SPARSE_FILE.unlink()
            print("Wiped sparse_index.pkl")

    print("Loading gita_enriched.json...")
    verses = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(verses)} verses.")

    print("\nLoading BGE-M3 (first run downloads ~570MB)...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("BGE-M3 ready.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    verses_col = client.get_or_create_collection(
        name="gita_verses",
        metadata={"hnsw:space": "cosine"},
    )
    purport_col = client.get_or_create_collection(
        name="gita_purport",
        metadata={"hnsw:space": "cosine"},
    )

    sparse_index: dict[str, dict[str, float]] = {}
    if SPARSE_FILE.exists() and not args.reset:
        with open(SPARSE_FILE, "rb") as f:
            sparse_index = pickle.load(f)
        print(f"Loaded existing sparse index ({len(sparse_index)} tokens).")

    existing_verse_ids = set(verses_col.get(include=[])["ids"])
    existing_purport_ids = set(purport_col.get(include=[])["ids"])

    # ------------------------------------------------------------------
    # Verse-level vectors (meaning + translation)
    # ------------------------------------------------------------------
    print("\n--- Verse-level vectors (meaning + translation) ---")
    to_index = [v for v in verses if f"{v['verse_id']}_meaning" not in existing_verse_ids]
    print(f"{len(to_index)} verses to index, {len(verses) - len(to_index)} already done.")

    for i in tqdm(range(0, len(to_index), EMBED_BATCH), desc="Verse vectors"):
        batch = to_index[i: i + EMBED_BATCH]
        meaning_texts = [v["text_for_embedding"] for v in batch]
        trans_texts = [v["translation"] for v in batch]

        m_dense, m_sparse = embed_batch(model, meaning_texts)
        t_dense, t_sparse = embed_batch(model, trans_texts)

        ids, docs, metas, embeds = [], [], [], []
        for j, v in enumerate(batch):
            vid = v["verse_id"]
            base_meta = {
                "verse_id":    vid,
                "chapter":     int(v["chapter"]),
                "verse":       int(v["verse"]),
                "devanagari":  v.get("devanagari", ""),
                "sanskrit":    v.get("sanskrit", ""),
                "translation": v["translation"],
            }
            ids += [f"{vid}_meaning", f"{vid}_translation"]
            docs += [meaning_texts[j], trans_texts[j]]
            metas += [
                {**base_meta, "type": "meaning"},
                {**base_meta, "type": "translation"},
            ]
            embeds += [m_dense[j].tolist(), t_dense[j].tolist()]
            update_sparse_index(sparse_index, f"{vid}_meaning", m_sparse[j])
            update_sparse_index(sparse_index, f"{vid}_translation", t_sparse[j])

        verses_col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

    # ------------------------------------------------------------------
    # Purport chunk vectors
    # ------------------------------------------------------------------
    print("\n--- Purport chunk vectors ---")
    purport_records: list[tuple] = []

    for v in verses:
        vid = v["verse_id"]
        header = f"Verse {vid}: {v['translation']}"
        chunks = chunk_purport(v.get("purport", ""))

        for chunk_idx, chunk_text in enumerate(chunks):
            doc_id = f"{vid}_purport_{chunk_idx}"
            if doc_id in existing_purport_ids:
                continue
            parent_start, parent_end = get_parent_window(chunks, chunk_idx)
            purport_records.append((
                doc_id,
                f"{header}\n\n{chunk_text}",
                {
                    "verse_id":     vid,
                    "chapter":      int(v["chapter"]),
                    "verse":        int(v["verse"]),
                    "type":         "purport_chunk",
                    "chunk_index":  chunk_idx,
                    "parent_start": parent_start,
                    "parent_end":   parent_end,
                    "translation":  v["translation"],
                },
            ))

    print(f"{len(purport_records)} chunks to index.")

    for i in tqdm(range(0, len(purport_records), EMBED_BATCH), desc="Purport chunks"):
        batch = purport_records[i: i + EMBED_BATCH]
        texts = [r[1] for r in batch]
        dense, sparse_weights = embed_batch(model, texts)

        ids, docs, metas, embeds = [], [], [], []
        for j, (doc_id, document, meta) in enumerate(batch):
            ids.append(doc_id)
            docs.append(document)
            metas.append(meta)
            embeds.append(dense[j].tolist())
            update_sparse_index(sparse_index, doc_id, sparse_weights[j])

        purport_col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

    # ------------------------------------------------------------------
    # Save sparse index
    # ------------------------------------------------------------------
    print("\nSaving sparse index...")
    with open(SPARSE_FILE, "wb") as f:
        pickle.dump(sparse_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    v_count = verses_col.count()
    p_count = purport_col.count()
    print(f"\nDone.")
    print(f"  gita_verses:  {v_count} vectors  (expected ~{len(verses) * 2})")
    print(f"  gita_purport: {p_count} vectors")
    print(f"  sparse_index: {len(sparse_index)} unique tokens")
    print(f"  Total:        {v_count + p_count} vectors")


if __name__ == "__main__":
    main()
