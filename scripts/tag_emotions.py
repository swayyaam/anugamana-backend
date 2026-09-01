#!/usr/bin/env python3
"""
Tag every verse with a weighted emotion profile.

The hypothesis
--------------
The corpus already has an `emotions` field, but it is free prose buried inside
`text_for_embedding` alongside situations, teaching and concepts — so an
affective query competes against three other kinds of vocabulary for the same
similarity budget. That is the likeliest reason the query-side-only emotion arm
measured a real but small gain (+0.0203 nDCG@10, p = 0.0024).

Pulling emotion out into a structured, weighted profile lets query emotion be
matched against verse emotion *directly*, instead of hoping a blended embedding
preserves it. Whether that helps is condition C14 — it is measured before it is
served.

Design
------
Five labels per verse from the 24-label taxonomy. Each label gets an
independent 0-100 intensity — deliberately NOT a distribution summing to 1.0,
because asking for shares of a fixed budget produced the same template on 93% of
verses. Raw intensities are kept alongside a normalised form so matching can use
either, and so the evaluation can decide whether the weights add anything over
the label order alone. Sonnet is used rather than Haiku: this is a judgement
task over a whole purport, and the enrichment it must be consistent with was
itself generated once and reused forever.

The model sees the verse, its purport, and the existing free-text emotions field
— that field is the corpus's own account of the verse's affective content, and
ignoring it would invite a second, inconsistent opinion.

Usage:
    python scripts/tag_emotions.py                 # resumes
    python scripts/tag_emotions.py --limit 20      # sample first
    python scripts/tag_emotions.py --model claude-sonnet-5
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import AsyncAnthropic  # noqa: E402

from app.config import ANTHROPIC_API_KEY, DATA_DIR, ENRICHED_FILE  # noqa: E402

TAXONOMY_FILE = DATA_DIR / "emotion_taxonomy.json"
OUT_FILE = DATA_DIR / "verse_emotions.json"

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
TAGS = 5
SAVE_EVERY = 40

client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

BOLD, DIM, GREEN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"
)


def load_taxonomy():
    data = json.loads(TAXONOMY_FILE.read_text(encoding="utf-8"))
    return data["labels"]


def build_tool(labels):
    catalogue = "\n".join(
        f"  {l['key']:<15} ({l['gita_term']}) — {l['gloss']}" for l in labels
    )
    return {
        "name": "tag_emotions",
        "description": (
            "Assign exactly five weighted emotion labels to a verse, describing "
            "the affective states the verse speaks to."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "emotions": {
                    "type": "array",
                    "minItems": TAGS,
                    "maxItems": TAGS,
                    "description": (
                        f"Exactly {TAGS} labels, strongest first.\n\n"
                        f"Available labels:\n{catalogue}"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "enum": [l["key"] for l in labels],
                            },
                            "intensity": {
                                "type": "integer",
                                "description": (
                                    "How strongly this verse speaks to this "
                                    "state, 0-100, judged INDEPENDENTLY of the "
                                    "other four. These do NOT sum to any total. "
                                    "A verse squarely about one state should "
                                    "score it 90+ and a marginal fifth label 15."
                                ),
                            },
                        },
                        "required": ["label", "intensity"],
                    },
                }
            },
            "required": ["emotions"],
        },
    }


SYSTEM = """\
You are building a retrieval index over the Bhagavad-gita As It Is.

For each verse, assign five weighted emotion labels describing the affective
states the verse SPEAKS TO — the states a reader would be in when this verse
becomes the right thing to read.

Rules:
- Tag what the verse addresses, not what it depicts. A verse describing Krishna's
  opulence may speak to insecurity or longing even though it names neither.
- Score each label's intensity 0-100 INDEPENDENTLY. They are five separate
  judgements, not shares of a fixed budget, and they must not sum to anything.
- Use the full range and let the verse decide the shape. Some verses are
  overwhelmingly about one state (95, 40, 25, 15, 10). Some genuinely touch
  several at once (70, 65, 60, 45, 30). Some are mild throughout (45, 35, 30,
  25, 20). All three shapes are correct when the verse is like that.
- Do NOT produce the same shape every time. A near-identical spread on every
  verse means you are filling in a template instead of reading the verse, and
  it destroys the signal this field exists to carry.
- The fifth label is still a real judgement, not filler.
- Use the verse's existing emotions description as evidence of what the
  commentary itself emphasises, but you assign the labels.\
"""


async def tag_one(verse, tool, model, semaphore):
    async with semaphore:
        purport = (verse.get("purport") or "")[:2500]
        existing = verse.get("meaning_fields", {}).get("emotions", "")
        message = (
            f"Verse {verse['verse_id']} (chapter {verse['chapter']})\n\n"
            f"Translation:\n{verse['translation']}\n\n"
            f"Commentary:\n{purport}\n\n"
            f"The corpus describes this verse's emotional content as:\n{existing}"
        )
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=800,
                system=SYSTEM,
                tools=[tool],
                tool_choice={"type": "tool", "name": "tag_emotions"},
                messages=[{"role": "user", "content": message}],
            )
            block = next(b for b in response.content if b.type == "tool_use")
            raw = block.input["emotions"]
        except Exception as e:
            print(f"\n  {verse['verse_id']} failed: {str(e)[:90]}")
            return None

    intensity = {}
    for item in raw:
        label = item.get("label")
        value = float(item.get("intensity", 0) or 0)
        if label and value > 0:
            intensity[label] = max(intensity.get(label, 0.0), value)
    total = sum(intensity.values())
    if total <= 0:
        return None

    ordered = sorted(intensity.items(), key=lambda x: -x[1])
    return {
        "verse_id": verse["verse_id"],
        # Raw 0-100 intensities are kept: they carry the absolute strength a
        # normalised distribution throws away, and matching can use either.
        "intensity": {k: int(v) for k, v in ordered},
        "emotions": {k: round(v / total, 4) for k, v in ordered},
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    labels = load_taxonomy()
    tool = build_tool(labels)
    verses = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
    if args.limit:
        verses = verses[: args.limit]

    done = {}
    if OUT_FILE.exists():
        done = json.loads(OUT_FILE.read_text(encoding="utf-8"))
        print(f"{DIM}resuming — {len(done)} verses already tagged{RESET}")

    todo = [v for v in verses if v["verse_id"] not in done]
    print(f"{BOLD}Tagging {len(todo)} verses{RESET} with {TAGS} weighted labels "
          f"from {len(labels)} using {args.model}")

    semaphore = asyncio.Semaphore(args.concurrency)
    for start in range(0, len(todo), SAVE_EVERY):
        batch = todo[start : start + SAVE_EVERY]
        results = await asyncio.gather(
            *(tag_one(v, tool, args.model, semaphore) for v in batch)
        )
        for item in results:
            if item:
                done[item["verse_id"]] = {
                    "emotions": item["emotions"],
                    "intensity": item["intensity"],
                }
        OUT_FILE.write_text(json.dumps(done, indent=2), encoding="utf-8")
        print(f"\r  {min(start + SAVE_EVERY, len(todo))}/{len(todo)} "
              f"(saved {len(done)})", end="", flush=True)
    print()

    # --- distribution report ----------------------------------------------
    profiles = {k: v["emotions"] for k, v in done.items()}
    primary = Counter(max(p, key=p.get) for p in profiles.values())
    mass = Counter()
    for profile in profiles.values():
        for label, weight in profile.items():
            mass[label] += weight

    print(f"\n{BOLD}Tagged {len(done)} verses{RESET}")
    print(f"\n  {'label':<16}{'primary':>9}{'total mass':>12}")
    for label in [l["key"] for l in labels]:
        print(f"  {label:<16}{primary.get(label, 0):>9}{mass.get(label, 0):>12.1f}")

    unused = [l["key"] for l in labels if mass.get(l["key"], 0) == 0]
    if unused:
        print(f"\n  {YELLOW}never used: {', '.join(unused)}{RESET} — candidates "
              f"for removal from the taxonomy")

    top_weights = sorted(max(p.values()) for p in profiles.values())
    distinct = len({round(w, 2) for w in top_weights})
    print(f"\n  dominant-label share: p10={top_weights[len(top_weights)//10]:.2f} "
          f"median={top_weights[len(top_weights)//2]:.2f} "
          f"p90={top_weights[9*len(top_weights)//10]:.2f}")
    print(f"  distinct dominant-share values: {distinct}")

    tops = sorted(max(v["intensity"].values()) for v in done.values())
    spread = sorted(
        max(v["intensity"].values()) - min(v["intensity"].values())
        for v in done.values()
    )
    print(f"  top intensity:  p10={tops[len(tops)//10]} "
          f"median={tops[len(tops)//2]} p90={tops[9*len(tops)//10]}")
    print(f"  within-verse spread (top - bottom): "
          f"p10={spread[len(spread)//10]} median={spread[len(spread)//2]} "
          f"p90={spread[9*len(spread)//10]}")
    distinct_tops = len(set(tops))
    distinct_spreads = len(set(spread))
    print(f"  distinct top intensities: {distinct_tops} · "
          f"distinct spreads: {distinct_spreads}")
    if distinct_tops < 5 or distinct_spreads < 5:
        print(f"  {YELLOW}WARNING: intensities barely vary across verses — the "
              f"model is filling in a template, and the weights carry no "
              f"signal beyond the label order{RESET}")
    else:
        print(f"  {DIM}intensities vary per verse; whether that variation helps "
              f"retrieval is condition C14's job to decide{RESET}")
    print(f"\n{DIM}written -> {OUT_FILE}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
