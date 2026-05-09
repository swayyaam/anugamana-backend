#!/usr/bin/env python3
"""
Phase 1: Deep systematic analysis of the Bhagavad Gita As It Is.

Reads data/gita_full.json, analyzes each chapter with Claude,
then synthesizes a unified reference document to data/gita_analysis.md.

This document drives all downstream prompt engineering:
- Enrichment prompts (Phase 2)
- HyDE prompt style and vocabulary (Phase 4)
- Input guardrail vocabulary (Phase 6)
- Golden evaluation dataset seed queries (Phase 5)

Resume-safe: chapter analyses are cached to data/gita_chapter_analyses.json.
Interrupt and re-run; completed chapters are skipped.

Usage:
    source venv/bin/activate
    python scripts/analyze_gita.py
    python scripts/analyze_gita.py --force   # overwrite existing output
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
INPUT_FILE = DATA_DIR / "gita_full.json"
OUTPUT_FILE = DATA_DIR / "gita_analysis.md"
CHAPTER_CACHE_FILE = DATA_DIR / "gita_chapter_analyses.json"

MODEL = "claude-sonnet-4-6"
MAX_PURPORT_CHARS = 3000  # truncate very long purports to keep chapter prompts manageable

CHAPTER_NAMES = {
    1:  "Observing the Armies on the Battlefield of Kurukṣetra",
    2:  "Contents of the Gītā Summarized",
    3:  "Karma-yoga",
    4:  "Transcendental Knowledge",
    5:  "Karma-yoga — Action in Kṛṣṇa Consciousness",
    6:  "Dhyana-yoga — Meditation",
    7:  "Knowledge of the Absolute",
    8:  "Attaining the Supreme",
    9:  "The Most Confidential Knowledge",
    10: "The Opulence of the Absolute",
    11: "The Universal Form",
    12: "Devotional Service",
    13: "Nature, the Enjoyer, and Consciousness",
    14: "The Three Modes of Material Nature",
    15: "The Yoga of the Supreme Person",
    16: "The Divine and Demoniac Natures",
    17: "The Divisions of Faith",
    18: "Conclusion — The Perfection of Renunciation",
}

CHAPTER_ANALYSIS_SYSTEM = """\
You are a deep scholar of the Bhagavad Gita As It Is by Srila Prabhupada.
You will analyze one chapter of the Bhagavad Gita and produce a structured reference analysis.

Be specific and grounded in the actual text. Do not generalize or drift into generic spirituality.
Every claim must reference specific verse IDs. Use the verse IDs exactly as given (e.g., "2.47").

Produce the following sections, in order:

## THEMATIC TAXONOMY
Every distinct theme in this chapter. For each:
- Theme name (concise label, e.g. "attachment to outcomes", "the eternal soul")
- Verse IDs where this theme appears
- One sentence: what the Gita teaches about this theme in these verses

## EMOTIONAL LANDSCAPE
Every human emotional state the chapter directly addresses. For each:
- Emotion name (e.g., grief, fear, confusion, envy, devotion, equanimity)
- Verse IDs where it appears
- How the Gita responds: what it teaches about or prescribes for this emotion

## PHILOSOPHICAL CONCEPTS
Every philosophical or spiritual concept introduced or developed here. For each:
- Concept name: Sanskrit term (if any) + plain English name
- Plain-English definition, 1-2 sentences, zero Sanskrit jargon
- Verse IDs where it is defined or used

## QUERY ARCHETYPES
6–10 questions a modern person might ask that a verse in this chapter best answers.
For each:
- The user query (natural modern language — no Gita jargon, as if texting a friend)
- Best matching verse ID(s)
- Why this verse answers the query (1 sentence)

## CHAPTER FRAMING
What is this chapter's specific contribution to the Gita? What context or lens must any prompt for \
this chapter's verses carry? Write 3–5 sentences capturing the chapter's unique perspective, \
tone, and what makes its verses distinct from other chapters.

## HYDE VOCABULARY
The specific vocabulary, phrases, and rhetorical patterns that appear in Prabhupada's purports \
for this chapter. Used to write HyDE (Hypothetical Document Embedding) prompts that land in the \
same semantic space as the indexed text.
List:
- Key Sanskrit terms with their meanings as used by Prabhupada
- Characteristic Prabhupada phrases (exact phrases, in quotes)
- Key analogies or metaphors used in this chapter's purports
- Tone markers (formal, urgent, devotional, instructive, etc.)

## EDGE CASES
Verses in this chapter requiring special prompt handling. For each:
- Verse ID(s)
- Why it is an edge case (purely cosmological, no modern application, specialized vocabulary, etc.)
- What kind of query could still surface it\
"""

SYNTHESIS_SYSTEM = """\
You are a deep scholar of the Bhagavad Gita As It Is by Srila Prabhupada.

You will receive detailed analyses of all 18 chapters and synthesize them into a single \
unified reference document that will guide an AI RAG system.

This document will be used to:
1. Write the enrichment prompt for every verse (meaning_fields generation)
2. Design HyDE prompts that embed in the same space as indexed Gita commentary
3. Write the input guardrail classifier vocabulary
4. Seed the golden retrieval evaluation dataset with real query archetypes
5. Guide chapter-specific prompting in the enrichment phase

Be exhaustive and specific. This is a working technical reference, not an essay.\
"""


def format_chapter_for_prompt(verses: list[dict]) -> str:
    parts = []
    for v in verses:
        lines = [f"### Verse {v['verse_id']}"]
        if v.get("devanagari"):
            lines.append(f"**Devanagari:** {v['devanagari']}")
        if v.get("sanskrit"):
            lines.append(f"**Sanskrit:** {v['sanskrit']}")
        if v.get("synonyms"):
            lines.append(f"**Synonyms:** {v['synonyms']}")
        lines.append(f"**Translation:** {v['translation']}")
        if v.get("purport"):
            purport = v["purport"]
            if len(purport) > MAX_PURPORT_CHARS:
                purport = purport[:MAX_PURPORT_CHARS] + " [...]"
            lines.append(f"**Purport:** {purport}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def analyze_chapter(client: anthropic.Anthropic, chapter_num: int, verses: list[dict]) -> str:
    chapter_name = CHAPTER_NAMES[chapter_num]
    verse_text = format_chapter_for_prompt(verses)

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=[
            {
                "type": "text",
                "text": CHAPTER_ANALYSIS_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Analyze Chapter {chapter_num} of the Bhagavad Gita As It Is:\n"
                    f"**{chapter_name}**\n\n"
                    f"{verse_text}"
                ),
            }
        ],
    )

    usage = response.usage
    cache_note = ""
    if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
        cache_note = f" (cache hit: {usage.cache_read_input_tokens} tokens)"
    print(
        f"    tokens in={usage.input_tokens} out={usage.output_tokens}{cache_note}"
    )

    return response.content[0].text


# One call per section — each gets the full 16k token budget with no bundling.
SYNTHESIS_SECTIONS = [
    (
        "Section 1 — Thematic Taxonomy",
        """\
Produce SECTION 1: THEMATIC TAXONOMY.

A structured taxonomy of every major theme across the entire 18-chapter Gita.
Group related themes into named clusters. For each cluster:
- Cluster name
- Sub-themes (with verse IDs for each sub-theme)
- Primary chapters
- One paragraph: the Gita's unified teaching on this theme cluster

Be exhaustive — cover every distinct theme present in the text.\
""",
    ),
    (
        "Section 2 — Emotional Landscape",
        """\
Produce SECTION 2: EMOTIONAL LANDSCAPE.

A complete map of every human emotional state the Gita directly addresses across all 18 chapters.
For each emotion:
- Emotion name and 1-sentence description
- Primary chapters that address it
- Key verse IDs
- The Gita's response: what it teaches about or prescribes for this emotion (2-4 sentences)

Cover all emotions present: grief, fear, confusion, despondency, attachment, anger, pride, \
envy, equanimity, devotion, wonder/awe, guilt, duty-conflict, and any others found in the text.\
""",
    ),
    (
        "Section 3 — Philosophical Concept Inventory",
        """\
Produce SECTION 3: PHILOSOPHICAL CONCEPT INVENTORY.

Every major philosophical and spiritual concept in the Gita.
For each concept:
- Sanskrit term + plain English name
- Plain-English definition (2-3 sentences, zero Sanskrit jargon)
- Chapters where it is introduced and developed
- How a modern person encounters this concept — the real-life situation that surfaces it

Cover all key concepts: ātmā, deha, karma, dharma, yoga (all types), guṇas, māyā, mokṣa, \
bhakti, jñāna, varṇāśrama, avatāra, paramātmā, brahman, ahiṁsā, and all others.\
""",
    ),
    (
        "Section 4 — Query Archetypes",
        """\
Produce SECTION 4: QUERY ARCHETYPES.

A comprehensive taxonomy of query types that users will bring to this Bhagavad Gita search system.
Organize into these categories:
- Emotional queries ("I feel X", "I'm struggling with Y")
- Philosophical queries ("What is the nature of X", "What does the Gita say about Y")
- Practical / ethical queries ("How should I handle X", "Is it wrong to Y")
- Devotional queries ("How do I develop love for God", "What is the highest form of worship")
- Metaphysical / cosmological queries ("What happens after death", "What is the nature of time")
- Direct lookup queries (verse reference, Sanskrit term, "what does 2.47 say")

For each archetype:
- Query pattern (e.g., "I feel [emotion] about [situation]")
- 3 specific example queries written in natural modern language (no Gita jargon)
- Best matching chapters / verse types
- Retrieval notes: what makes this query type easy or hard for the pipeline (vocabulary gap, \
  specificity, Sanskrit terms, etc.)

Aim for 5-8 archetypes per category.\
""",
    ),
    (
        "Section 5 — Chapter-Specific Framing",
        """\
Produce SECTION 5: CHAPTER-SPECIFIC PROMPTING FRAMES.

For ALL 18 chapters, a prompt engineering reference note. Cover every chapter from 1 to 18.
For each chapter:
- Chapter number and name
- What makes this chapter's verses unique (1-2 sentences)
- The lens or frame that must be present in any enrichment prompt for verses from this chapter
- The chapter's emotional register (the dominant mood/tone)
- 2-3 specific prompting pitfalls to avoid for this chapter

Be precise and specific to each chapter. Do not give generic spiritual advice.\
""",
    ),
    (
        "Section 6 — HyDE Vocabulary and Style Guide",
        """\
Produce SECTION 6: HYDE VOCABULARY AND STYLE GUIDE.

This section is a practical reference for writing HyDE (Hypothetical Document Embeddings) \
prompts. The hypothetical documents must embed in the same semantic space as Prabhupada's \
commentary as indexed in the vector store. This section must be detailed and actionable.

Include:

**6.1 Prabhupada's Characteristic Phrases**
Exact phrases from the purports that characterize Prabhupada's commentary style. \
These are phrases a HyDE document should use to match the indexed text.
List at least 20 specific phrases.

**6.2 Key Sanskrit Terms in Commentary**
Every Sanskrit term Prabhupada uses regularly in his purports (not just verse translations). \
For each: the term, its English gloss, and how Prabhupada typically deploys it in explanatory prose.

**6.3 Tone, Register, and Sentence Structure**
Describe the specific rhetorical character of Prabhupada's commentary:
- Sentence length and complexity patterns
- How he moves between Sanskrit authority and modern application
- His characteristic argumentative moves (e.g., authority citation, analogy, direct address)
- The vocabulary register (formal? devotional? instructive?)

**6.4 What to Avoid**
Specific language patterns that will cause a HyDE document to embed in the wrong space:
- New Age / self-help vocabulary (list specific words and phrases to avoid)
- Academic Sanskrit scholarship register (different from Prabhupada's devotional register)
- Generic motivational language
- Anachronistic psychological framing

**6.5 Example HyDE Documents**
3 complete example HyDE hypothetical documents, each 4-6 sentences, showing the correct style.
Write one for each query type:
1. An emotional query: "I am paralyzed by fear of making the wrong decision"
2. A philosophical query: "What is the nature of the soul according to the Gita"
3. A practical query: "How do I stop being controlled by my desires"\
""",
    ),
    (
        "Section 7 — Edge Cases and Special Handling",
        """\
Produce SECTION 7: EDGE CASES AND SPECIAL HANDLING.

Verses requiring special treatment in enrichment prompting or retrieval.
Be specific and exhaustive. Cover all of the following categories:

**7.1 Cosmological / Purely Metaphysical Verses**
Verses with no direct human emotional application (cosmic time cycles, creation descriptions, \
descriptions of the spiritual sky, etc.). For each: verse IDs, the challenge, \
how to make them retrievable despite the gap.

**7.2 Multi-Verse Passages (Mandatory Co-Retrieval)**
Verses that form a single argument and must be enriched and retrieved together. \
For each block: verse IDs, why they must be treated as a unit, \
how to tag them for co-retrieval in the indexer.

**7.3 Purport-Dominant Verses**
Verses where Prabhupada's purport dramatically reframes or expands beyond the literal translation. \
For each: verse IDs, the gap between literal and Prabhupada's interpretation, \
prompting instruction to bridge the gap.

**7.4 Sanskrit-Vocabulary-Dense Verses**
Verses where specialized Sanskrit vocabulary in the translation/synonyms affects retrieval \
(a modern user would never use these words). For each: verse IDs, the problematic terms, \
recommended plain-English bridge language for the meaning_fields.

**7.5 Narrative / Historical Verses (Chapter 1 and elsewhere)**
Verses that are purely narrative (describing who stood where, what weapons were used) \
with no philosophical content. How to handle enrichment without over-reading into them.\
""",
    ),
]


def synthesize_section(
    client: anthropic.Anthropic,
    section_label: str,
    section_request: str,
    all_chapters_text: str,
) -> str:
    """Run one synthesis pass for a subset of sections, with chapter analyses cached."""
    user_content = (
        "Below are detailed analyses of all 18 chapters of the Bhagavad Gita As It Is.\n\n"
        f"{all_chapters_text}\n\n"
        "---\n\n"
        f"{section_request}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=[
            {
                "type": "text",
                "text": SYNTHESIS_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                # Cache the large shared chapter context; only the section request varies
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Below are detailed analyses of all 18 chapters of the "
                            "Bhagavad Gita As It Is.\n\n"
                            f"{all_chapters_text}\n\n---\n\n"
                        ),
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": section_request,
                    },
                ],
            }
        ],
    )

    usage = response.usage
    cache_note = ""
    if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
        cache_note = f" (cache read: {usage.cache_read_input_tokens})"
    if hasattr(usage, "cache_creation_input_tokens") and usage.cache_creation_input_tokens:
        cache_note += f" (cache write: {usage.cache_creation_input_tokens})"
    print(f"    {section_label}: in={usage.input_tokens} out={usage.output_tokens}{cache_note}")

    return response.content[0].text


def synthesize(client: anthropic.Anthropic, chapter_analyses: dict[int, str]) -> str:
    """Run all synthesis sections and concatenate into the final document."""
    all_chapters_text = "\n\n---\n\n".join(
        f"# Chapter {ch}: {CHAPTER_NAMES[ch]}\n\n{analysis}"
        for ch, analysis in sorted(chapter_analyses.items())
    )

    parts = []
    for label, request in SYNTHESIS_SECTIONS:
        print(f"  Synthesizing: {label}...")
        section_text = synthesize_section(client, label, request, all_chapters_text)
        parts.append(section_text)
        time.sleep(0.5)

    return "\n\n---\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Analyze the Bhagavad Gita with Claude")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run synthesis even if gita_analysis.md already exists",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete chapter cache and start from scratch",
    )
    parser.add_argument(
        "--resynthesize",
        action="store_true",
        help="Skip chapter analysis (use cached results) and only rerun synthesis",
    )
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment (.env or shell)")
        sys.exit(1)

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found — run scripts/scraper.py first")
        sys.exit(1)

    if OUTPUT_FILE.exists() and not args.force and not args.resynthesize:
        print(f"{OUTPUT_FILE} already exists. Use --force to overwrite.")
        sys.exit(0)

    if args.reset and CHAPTER_CACHE_FILE.exists():
        CHAPTER_CACHE_FILE.unlink()
        print("Chapter cache cleared.")

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Loading {INPUT_FILE.name}...")
    verses = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(verses)} verses.")

    # Group by chapter
    chapters: dict[int, list[dict]] = {}
    for v in verses:
        chapters.setdefault(v["chapter"], []).append(v)

    # Load existing chapter analyses (resume support)
    chapter_analyses: dict[int, str] = {}
    if CHAPTER_CACHE_FILE.exists():
        raw = json.loads(CHAPTER_CACHE_FILE.read_text(encoding="utf-8"))
        chapter_analyses = {int(k): v for k, v in raw.items()}
        print(f"Resuming: {len(chapter_analyses)}/18 chapters already analyzed.")

    # Analyze each chapter (skip if --resynthesize)
    if args.resynthesize:
        if len(chapter_analyses) < 18:
            print(f"ERROR: --resynthesize requires all 18 chapters cached, only {len(chapter_analyses)} found.")
            sys.exit(1)
        print("--resynthesize: skipping chapter analysis, using cached results.")
    else:
        for ch_num in range(1, 19):
            if ch_num in chapter_analyses:
                print(f"  Chapter {ch_num:2d}: already done, skipping.")
                continue

            ch_name = CHAPTER_NAMES[ch_num]
            ch_verses = chapters.get(ch_num, [])
            print(f"  Chapter {ch_num:2d}: {ch_name} ({len(ch_verses)} verses)...")

            chapter_analyses[ch_num] = analyze_chapter(client, ch_num, ch_verses)

            # Save after each chapter so interruption is safe
            CHAPTER_CACHE_FILE.write_text(
                json.dumps(chapter_analyses, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            if ch_num < 18:
                time.sleep(0.5)

    # Final synthesis pass
    print("\nRunning synthesis pass...")
    final_doc = synthesize(client, chapter_analyses)

    header = (
        "# Gita Analysis Reference Document\n\n"
        f"*Source: Bhagavad Gita As It Is, Srila Prabhupada — {len(verses)} verses, 18 chapters*\n"
        "*Generated by scripts/analyze_gita.py (Phase 1)*\n\n"
        "This document drives all downstream prompt engineering:\n"
        "enrichment prompts, HyDE vocabulary, guardrail classifier, and eval query seeds.\n\n"
        "---\n\n"
    )

    OUTPUT_FILE.write_text(header + final_doc, encoding="utf-8")
    print(f"\nDone. Output: {OUTPUT_FILE}")
    print(f"Chapter cache: {CHAPTER_CACHE_FILE} (safe to delete after synthesis)")


if __name__ == "__main__":
    main()
