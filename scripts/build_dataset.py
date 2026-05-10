#!/usr/bin/env python3
"""
Build the golden evaluation dataset for Phase 5.

Sources (in order of priority):
  1. data/golden_manual.json   — hand-curated pairs you write yourself (highest quality)
  2. data/gita_enriched.json   — mined from situations field (auto-generated seeds)

Output: data/golden_dataset.json
Format: [{"query": "...", "verse_id": "2.47", "source": "manual|mined"}, ...]

How to add manual pairs:
  Create data/golden_manual.json as a list:
  [
    {"query": "I keep failing at work and feel like giving up", "verse_id": "2.47"},
    {"query": "what does the Gita say about the soul after death", "verse_id": "2.20"},
    ...
  ]
  Then re-run this script to merge.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --max-mined 60   # limit auto-mined pairs
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
ENRICHED_FILE = DATA_DIR / "gita_enriched.json"
MANUAL_FILE = DATA_DIR / "golden_manual.json"
OUTPUT_FILE = DATA_DIR / "golden_dataset.json"


def mine_from_situations(verses: list[dict], max_pairs: int) -> list[dict]:
    """
    Extract individual situation sentences from meaning_fields.situations
    and pair each with its verse_id.
    """
    pairs = []
    for v in verses:
        situations_text = v.get("meaning_fields", {}).get("situations", "")
        if not situations_text:
            continue

        # Split on semicolons or sentence boundaries
        raw = [s.strip().rstrip(";.,") for s in situations_text.split(";")]
        situations = [s for s in raw if len(s.split()) >= 5]

        for sit in situations:
            # Capitalise first letter
            query = sit[0].upper() + sit[1:] if sit else sit
            pairs.append({
                "query": query,
                "verse_id": v["verse_id"],
                "source": "mined",
            })

    # Shuffle and cap
    random.shuffle(pairs)
    return pairs[:max_pairs]


def main():
    parser = argparse.ArgumentParser(description="Build golden evaluation dataset")
    parser.add_argument("--max-mined", type=int, default=80,
                        help="Max auto-mined pairs from situations field (default 80)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    verses = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(verses)} verses.")

    # Load manual pairs (highest quality, always included)
    manual_pairs = []
    if MANUAL_FILE.exists():
        manual_pairs = json.loads(MANUAL_FILE.read_text(encoding="utf-8"))
        # Add source tag if missing
        for p in manual_pairs:
            p.setdefault("source", "manual")
        print(f"Loaded {len(manual_pairs)} manual pairs from {MANUAL_FILE.name}")
    else:
        print(f"No manual pairs found ({MANUAL_FILE.name} does not exist).")
        print("Create it to add high-quality hand-curated pairs.")

    # Mine from situations
    mined_pairs = mine_from_situations(verses, args.max_mined)
    print(f"Mined {len(mined_pairs)} pairs from situations field.")

    # Merge: manual first, then mined (deduplicate by verse_id to ensure coverage)
    manual_verse_ids = {p["verse_id"] for p in manual_pairs}
    # Keep mined pairs for verses not already covered by manual
    mined_new = [p for p in mined_pairs if p["verse_id"] not in manual_verse_ids]

    dataset = manual_pairs + mined_new
    random.shuffle(dataset)

    OUTPUT_FILE.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Stats
    sources = {}
    verse_coverage = set()
    for p in dataset:
        sources[p["source"]] = sources.get(p["source"], 0) + 1
        verse_coverage.add(p["verse_id"])

    print(f"\nDataset: {len(dataset)} pairs → {OUTPUT_FILE.name}")
    print(f"  manual: {sources.get('manual', 0)}")
    print(f"  mined:  {sources.get('mined', 0)}")
    print(f"  verse coverage: {len(verse_coverage)} unique verses")


if __name__ == "__main__":
    main()
