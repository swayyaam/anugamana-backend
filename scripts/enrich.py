#!/usr/bin/env python3
"""
Phase 2: Enrich every verse with meaning_fields for semantic retrieval.

For each of the 700 verses, Claude generates 4 fields:
  situations  — real-world modern situations where this verse is relevant
  teaching    — core message in plain modern English, no Sanskrit
  emotions    — emotional states this verse directly addresses
  concepts    — philosophical/spiritual concepts introduced

These bridge the vocabulary gap between casual user queries ("I'm paralyzed by
fear of making the wrong decision") and scholarly verse text ("You have a right
to perform your duties, not to the fruits").

Design:
  - Chapter analysis from gita_chapter_analyses.json injected per verse
  - Prompt caching: system prompt + chapter context cached per chapter batch
    → cache creation paid once per chapter, reads free for the rest
  - Forced tool use → guaranteed structured JSON output
  - Resume: skips already-enriched verses (safe to interrupt and re-run)
  - tenacity retry on transient API errors
  - Error log for manual review

Usage:
    source venv/bin/activate
    python scripts/enrich.py
    python scripts/enrich.py --reset   # clear output, start from scratch
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
INPUT_FILE = DATA_DIR / "gita_full.json"
CHAPTER_CACHE_FILE = DATA_DIR / "gita_chapter_analyses.json"
OUTPUT_FILE = DATA_DIR / "gita_enriched.json"
ERROR_LOG_FILE = DATA_DIR / "enrich_errors.json"

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1500
SAVE_EVERY = 20  # checkpoint every N newly enriched verses

CHAPTER_THEMES = {
    1:  "Arjuna's grief and moral crisis on the battlefield",
    2:  "Sankhya philosophy — the eternal soul, duty, and the foundation of yoga",
    3:  "Karma yoga — selfless action and why action cannot be avoided",
    4:  "Knowledge, divine incarnation, and the yoga of wisdom",
    5:  "Renunciation and action — both paths lead to liberation",
    6:  "Dhyana yoga — meditation, mind control, and the steady yogi",
    7:  "Knowledge of the Absolute — understanding God's nature and energies",
    8:  "The imperishable Brahman, death, cosmic cycles, and the path beyond",
    9:  "Royal knowledge — devotion, surrender, and God's immanence in all things",
    10: "Divine opulences — how God manifests as the best of everything",
    11: "The universal form — Arjuna's vision of the cosmic manifestation",
    12: "Bhakti yoga — devotion as the highest path and qualities of a devotee",
    13: "The field and the knower — matter, consciousness, and the self",
    14: "The three modes of nature and how they bind and condition the soul",
    15: "The supreme person — the tree of material existence and transcendence",
    16: "Divine and demoniac natures — qualities that lead to liberation or bondage",
    17: "Three divisions of faith — how the modes shape worship, food, and conduct",
    18: "Final conclusion — renunciation, the highest truth, and total surrender",
}

ENRICHMENT_SYSTEM = """\
You are an expert at bridging ancient Vedic wisdom and the language of modern life.

Your task: for each Bhagavad Gita verse, generate 4 semantic fields that will power \
vector search retrieval. A modern person will type something like "I'm paralyzed by \
fear of making the wrong decision" — these 4 fields must enable the system to match \
their query to the right verse.

Rules for every field:
- Use the everyday language of someone describing a real problem to a friend
- Be specific and concrete — "fear of consequences from a hard decision" beats "fear"
- situations and emotions must use zero Sanskrit terms
- teaching must state the insight directly — never start with "This verse teaches..."
- concepts may use Sanskrit terms but always explain them in plain English first
- Ground everything strictly in the verse translation and Prabhupada's purport
- Never invent meaning that isn't in the text
- Never produce generic spiritual platitudes — every word must be specific to THIS verse\
"""

MEANING_FIELDS_TOOL = {
    "name": "generate_meaning_fields",
    "description": (
        "Generate 4 semantic enrichment fields for a Bhagavad Gita verse. "
        "These fields are embedded for retrieval — they bridge modern everyday language "
        "to the verse's content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "situations": {
                "type": "string",
                "description": (
                    "3-5 specific real-world situations a modern person faces when this verse "
                    "becomes relevant. Written as concrete life scenarios in everyday language. "
                    "Example: 'stuck in a job you hate but too scared to quit; "
                    "agonizing over a decision you cannot undo; "
                    "taking on a project with no guarantee it will succeed'"
                ),
            },
            "teaching": {
                "type": "string",
                "description": (
                    "The core insight of this verse in plain modern English. "
                    "2-4 sentences. No Sanskrit. No 'Prabhupada says'. "
                    "State the teaching as a direct insight the reader can grasp immediately."
                ),
            },
            "emotions": {
                "type": "string",
                "description": (
                    "The emotional states, feelings, and mental conditions this verse directly "
                    "addresses or is a response to. Write as a flowing description, not a list. "
                    "Example: 'the anxiety that comes from obsessing over whether your actions "
                    "will succeed, the paralysis of attachment to outcomes, the restlessness "
                    "of a mind that cannot act freely because it fears failure'"
                ),
            },
            "concepts": {
                "type": "string",
                "description": (
                    "The philosophical or spiritual concepts this verse introduces or develops. "
                    "Explain each in plain English first, Sanskrit term in parentheses if relevant. "
                    "Example: 'acting without attachment to results (nishkama karma) — the idea "
                    "that you have full right to work but no claim on what that work produces; "
                    "duty-consciousness (dharma) as the basis of action rather than personal gain'"
                ),
            },
        },
        "required": ["situations", "teaching", "emotions", "concepts"],
    },
}


def format_verse(verse: dict) -> str:
    lines = [f"Verse {verse['verse_id']}"]
    if verse.get("devanagari"):
        lines.append(f"Devanagari: {verse['devanagari']}")
    if verse.get("sanskrit"):
        lines.append(f"Sanskrit: {verse['sanskrit']}")
    lines.append(f"Translation: {verse['translation']}")
    if verse.get("purport"):
        purport = verse["purport"]
        if len(purport) > 2500:
            purport = purport[:2500] + " [...]"
        lines.append(f"Purport: {purport}")
    return "\n".join(lines)


def build_chapter_context(ch_num: int, ch_analysis: str) -> str:
    theme = CHAPTER_THEMES[ch_num]
    # Include the full chapter analysis — it has framing, edge cases, HyDE vocab.
    # Cached across all verses in this chapter so the overhead is paid only once.
    return (
        f"CHAPTER {ch_num} — {theme}\n\n"
        f"CHAPTER ANALYSIS (use this to ground your enrichment for every verse in this chapter):\n"
        f"{ch_analysis}"
    )


@retry(
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIStatusError)),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(4),
)
def enrich_verse(
    client: anthropic.Anthropic,
    verse: dict,
    chapter_context: str,
) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": ENRICHMENT_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[MEANING_FIELDS_TOOL],
        tool_choice={"type": "tool", "name": "generate_meaning_fields"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        # Chapter context is constant within a chapter — cache it.
                        # Cache creation cost paid once; reads are ~10% of normal cost.
                        "text": chapter_context,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": f"Generate meaning_fields for this verse:\n\n{format_verse(verse)}",
                    },
                ],
            }
        ],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    fields = tool_block.input

    for key in ("situations", "teaching", "emotions", "concepts"):
        if not fields.get(key, "").strip():
            raise ValueError(f"Empty field '{key}' in response for {verse['verse_id']}")

    text_for_embedding = "\n".join([
        fields["situations"],
        fields["teaching"],
        fields["emotions"],
        fields["concepts"],
    ])

    return {
        **verse,
        "meaning_fields": {
            "situations": fields["situations"],
            "teaching":   fields["teaching"],
            "emotions":   fields["emotions"],
            "concepts":   fields["concepts"],
        },
        "text_for_embedding": text_for_embedding,
    }


def save(enriched: dict[str, dict], original_order: list[str]) -> None:
    output = [enriched[vid] for vid in original_order if vid in enriched]
    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Enrich verses with meaning_fields")
    parser.add_argument("--reset", action="store_true", help="Clear output and start from scratch")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    for f in (INPUT_FILE, CHAPTER_CACHE_FILE):
        if not f.exists():
            print(f"ERROR: {f} not found — run previous phase first")
            sys.exit(1)

    if args.reset and OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print("Output cleared.")

    client = anthropic.Anthropic(api_key=api_key)

    print("Loading data...")
    verses = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    raw_analyses = json.loads(CHAPTER_CACHE_FILE.read_text(encoding="utf-8"))
    chapter_analyses = {int(k): v for k, v in raw_analyses.items()}

    original_order = [v["verse_id"] for v in verses]

    # Resume: load already-enriched verses
    enriched: dict[str, dict] = {}
    if OUTPUT_FILE.exists():
        for v in json.loads(OUTPUT_FILE.read_text(encoding="utf-8")):
            enriched[v["verse_id"]] = v
        print(f"Resuming: {len(enriched)}/{len(verses)} already enriched.")

    # Error log
    errors: dict[str, str] = {}
    if ERROR_LOG_FILE.exists():
        errors = json.loads(ERROR_LOG_FILE.read_text(encoding="utf-8"))

    # Group by chapter (preserving order) for cache efficiency
    chapters: dict[int, list[dict]] = {}
    for v in verses:
        chapters.setdefault(v["chapter"], []).append(v)

    total = len(verses)
    done = len(enriched)
    new_since_save = 0

    for ch_num in range(1, 19):
        ch_verses = chapters.get(ch_num, [])
        remaining = [v for v in ch_verses if v["verse_id"] not in enriched]

        if not remaining:
            print(f"  Ch {ch_num:2d}: all {len(ch_verses)} done.")
            continue

        ch_analysis = chapter_analyses.get(ch_num, "")
        chapter_context = build_chapter_context(ch_num, ch_analysis)

        print(f"  Ch {ch_num:2d}: {len(remaining)}/{len(ch_verses)} to enrich  [{done}/{total} total]")

        for verse in remaining:
            vid = verse["verse_id"]
            try:
                enriched[vid] = enrich_verse(client, verse, chapter_context)
                done += 1
                new_since_save += 1

                if new_since_save >= SAVE_EVERY:
                    save(enriched, original_order)
                    new_since_save = 0
                    print(f"    checkpoint  [{done}/{total}]")

            except Exception as e:
                errors[vid] = str(e)
                ERROR_LOG_FILE.write_text(
                    json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(f"    FAIL {vid}: {e}")

    save(enriched, original_order)
    print(f"\nDone. {done}/{total} enriched → {OUTPUT_FILE}")
    if errors:
        print(f"Errors ({len(errors)} verses) → {ERROR_LOG_FILE}")
        print("Re-run the script to retry failed verses.")


if __name__ == "__main__":
    main()
