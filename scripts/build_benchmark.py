#!/usr/bin/env python3
"""
Build a decontaminated evaluation benchmark.

The methodological point
------------------------
The retracted benchmark was generated *from* documents: each query was a
sentence lifted out of a verse's enrichment, which is the text the retriever
searches. That is document-conditioned generation, and it guarantees leakage —
80/80 queries were verbatim substrings of the index.

This script generates queries **verse-blind**. No verse, translation, purport or
enrichment field is ever placed in the generation prompt. Queries are produced
from the information-need side — a grid of (affective state x life domain x
register) — exactly as a real IR benchmark elicits topics from users rather than
from the collection. Contamination becomes structurally impossible instead of
merely filtered, and the gate in scripts/check_contamination.py is a check on
that property rather than the mechanism enforcing it.

Which verses answer a query is decided afterwards, by pooled judgment over what
every condition retrieves (scripts/pool_and_judge.py). Nobody nominates a gold
verse in advance, so the benchmark cannot encode one system's idea of the answer.

Provenance is recorded per query. These are model-authored queries: a silver
resource that needs a human-verified subset before publication. See
data/benchmark/DATASET_CARD.md.

Usage:
    python scripts/build_benchmark.py                 # ~350 queries
    python scripts/build_benchmark.py --target 500
    python scripts/build_benchmark.py --dry-run       # print the grid, no API
"""

import argparse
import asyncio
import json
import re
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import AsyncAnthropic  # noqa: E402

from app.config import ANTHROPIC_API_KEY, DATA_DIR, LLM_MODEL  # noqa: E402
from app.services.emotion import TAXONOMY  # noqa: E402

BENCHMARK_DIR = DATA_DIR / "benchmark"
OUTPUT_FILE = BENCHMARK_DIR / "queries.json"

client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

#: Where in a life the situation arises. Chosen to span the concerns the corpus
#: actually addresses without naming any of its vocabulary.
DOMAINS = [
    "work and career",
    "family duty and obligation",
    "grief and bereavement",
    "romantic relationships",
    "money and material security",
    "health, ageing and the body",
    "friendship and betrayal",
    "identity and self-worth",
    "moral dilemmas with no clean answer",
    "faith, doubt and meaning",
    "study, practice and self-discipline",
    "conflict and confrontation",
]

#: How a real person phrases it. Register variation matters: a benchmark of
#: uniformly well-formed sentences overstates performance on real traffic.
REGISTERS = [
    ("first_person_distress",
     "first person, describing what they are going through right now, "
     "emotionally direct, 15-30 words"),
    ("terse",
     "very short and blunt, 4-9 words, the way someone types into a search box"),
    ("rambling",
     "rambling and unstructured, 35-60 words, with context and digression, "
     "the way someone types when upset"),
    ("third_person",
     "asking about someone else's situation — a friend, a parent, a child — "
     "15-30 words"),
    ("abstract",
     "an abstract or philosophical question with no personal detail, 8-20 words"),
]

SYSTEM = """\
You write realistic search queries for a system that helps people find guidance
for life situations.

Absolute constraints:
- NEVER mention the Bhagavad Gita, Krishna, Arjuna, dharma, karma, yoga, the
  soul, scripture, Hinduism, or any religious or Sanskrit vocabulary.
- NEVER quote or paraphrase scripture.
- Write what an ordinary person would type. Modern, everyday, secular language.
- Be concrete and specific. "worried about my performance review on Friday"
  beats "worried about work".
- Vary the details: different jobs, ages, family structures, cities, situations.
- Do not number the lines or add commentary.

You will be given an emotional state, a life domain, and a register. Produce
queries matching all three.

Output one query per line. Nothing else."""


def grid(target: int) -> list[tuple[str, str, str, str, int]]:
    """(emotion, domain, register_name, register_desc, n) cells."""
    emotions = [k for k in TAXONOMY if k != "equanimity"]
    cells = list(product(emotions, DOMAINS, REGISTERS))

    # Deterministic spread across the grid rather than a random sample, so the
    # benchmark's composition is reproducible and reportable.
    step = max(1, len(cells) // max(1, target // 3))
    selected = cells[::step]
    per_cell = max(1, round(target / max(1, len(selected))))
    return [(e, d, r[0], r[1], per_cell) for e, d, r in selected]


async def generate_cell(
    emotion: str, domain: str, register: str, register_desc: str, count: int
) -> list[dict]:
    prompt = (
        f"Emotional state: {TAXONOMY[emotion].gloss}\n"
        f"Life domain: {domain}\n"
        f"Register: {register_desc}\n\n"
        f"Write {count} different queries."
    )
    try:
        response = await client.messages.create(
            model=LLM_MODEL,
            max_tokens=900,
            temperature=1.0,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        lines = [
            re.sub(r'^[\s\-\*\d.)"]+|"$', "", line).strip()
            for line in response.content[0].text.strip().splitlines()
        ]
        return [
            {
                "query": line,
                "emotion_cell": emotion,
                "domain_cell": domain,
                "register": register,
                "source": "verse_blind_generation",
                "verified_by_human": False,
            }
            for line in lines
            if 3 <= len(line.split()) <= 70
        ]
    except Exception as e:
        print(f"  cell failed ({emotion}/{domain}/{register}): {e}")
        return []


def deduplicate(queries: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for item in queries:
        key = " ".join(re.findall(r"[a-z]+", item["query"].lower()))
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


LEAK_TERMS = re.compile(
    r"\b(gita|geeta|krishna|krsna|arjuna|dharma|karma|yoga|yogi|vedic|veda|"
    r"scripture|prabhupada|sanskrit|hindu|atman|moksha|bhakti|guna|brahman|"
    r"verse|sloka|shloka)\b",
    re.IGNORECASE,
)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=350)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cells = grid(args.target)
    print(f"Grid: {len(cells)} cells x ~{cells[0][4]} queries "
          f"-> ~{len(cells) * cells[0][4]} queries")
    print(f"  {len(TAXONOMY) - 1} affective states x {len(DOMAINS)} domains "
          f"x {len(REGISTERS)} registers")

    if args.dry_run:
        for emotion, domain, register, _, count in cells[:15]:
            print(f"  {emotion:<12} {domain:<36} {register:<22} n={count}")
        print(f"  ... {len(cells) - 15} more")
        return 0

    print("\nGenerating (verse-blind — no corpus text enters any prompt)...")
    semaphore = asyncio.Semaphore(6)

    async def run_cell(cell):
        async with semaphore:
            return await generate_cell(*cell)

    batches = await asyncio.gather(*(run_cell(c) for c in cells))
    queries = [q for batch in batches for q in batch]
    print(f"  generated {len(queries)}")

    queries = deduplicate(queries)
    print(f"  after dedup {len(queries)}")

    # The generator was told to avoid corpus vocabulary. Verify rather than trust.
    leaked = [q for q in queries if LEAK_TERMS.search(q["query"])]
    queries = [q for q in queries if not LEAK_TERMS.search(q["query"])]
    print(f"  removed {len(leaked)} with leaked religious/Sanskrit vocabulary")

    for index, item in enumerate(queries):
        item["query_id"] = f"q{index:04d}"

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from collections import Counter
    print(f"\nWrote {len(queries)} queries -> {OUTPUT_FILE}")
    print("  registers:", dict(Counter(q["register"] for q in queries)))
    print("  emotions: ", len(Counter(q["emotion_cell"] for q in queries)), "distinct")
    print("\nNext:")
    print("  python scripts/check_contamination.py data/benchmark/queries.json")
    print("  python scripts/pool_and_judge.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
