# Anugamana — RAG Pipeline Plan

> Semantic search over the Bhagavad Gita As It Is (Srila Prabhupada).
> User describes a life situation or question in natural language → gets the most relevant verse(s) + Claude-generated guidance.

---

## Current State

- FastAPI backend exists but is being rebuilt from scratch
- Scraper exists and has been improved (range pages, Devanagari, retry, checkpoint)
- 627 verses already in `data/gita_full.json`
- Old pipeline used: MiniLM-L6 embeddings + Pinecone + basic reranking

**We are rebuilding the full data and RAG pipeline with a proper foundation.**

---

## Phase 0 — Scraping (Done / In Progress)

### Goal
Collect all 700 verses of the Bhagavad Gita from vedabase.io with complete fields.

### Data Schema per verse
```json
{
  "verse_id":    "2.47",
  "chapter":     2,
  "verse":       47,
  "devanagari":  "कर्मण्येवाधिकारस्ते...",
  "sanskrit":    "karmaṇy evādhikāras te...",
  "synonyms":    "karmaṇi — in prescribed duties...",
  "translation": "You have a right to perform your prescribed duties...",
  "purport":     "..."
}
```

### Key improvements over old scraper
- Handles multi-verse range pages (`/en/library/bg/2/41-43/`) — old scraper missed 3 verses
- `div.av-devanagari` scraped for Devanagari script
- Retry with exponential backoff (tenacity, 4 attempts)
- Checkpoint/resume — saves after every page, safe to interrupt
- Sorted output by chapter/verse

### Run
```bash
source venv/bin/activate
python scripts/scraper.py
```

Output: `data/gita_full.json`

---

## Phase 1 — Gita Analysis (Before Enrichment Prompt)

### Goal
Before writing the enrichment prompt, do a **deep, systematic analysis of the entire Bhagavad Gita** using Claude.
This analysis drives every prompt engineering decision downstream — enrichment fields, HyDE prompt style, guardrail vocabulary, evaluation query archetypes.

### Why this matters
The Gita covers wildly different territory across 18 chapters:

| Chapters | Domain |
|---|---|
| 1 | Grief, moral crisis, emotional paralysis |
| 2 | Soul vs body, Sankhya philosophy, duty |
| 3, 5, 18 | Karma yoga — ethics of action |
| 4 | Knowledge, divine incarnation, sacrifice |
| 6 | Meditation, mind control, yogic practice |
| 7, 9, 10, 11 | Nature of God, devotion, cosmic form |
| 8, 13, 15 | Death, consciousness, metaphysics |
| 12 | Bhakti yoga — qualities of a devotee |
| 14, 17 | Three modes of nature (gunas), faith |
| 16 | Divine vs demoniac qualities |

A single generic prompt collapses all of this diversity into average output.

### What the analysis produces
1. **Thematic taxonomy** — every major theme, named and grouped
2. **Emotional landscape** — every human emotional state the Gita addresses
3. **Philosophical concept inventory** — key concepts with plain-English definitions
4. **Query archetypes** — real user question types mapped to verse types (seeds the golden eval dataset)
5. **Chapter-specific framing** — what makes each chapter's verses unique to prompt
6. **HyDE vocabulary** — the style, tone, and terminology the hypothetical documents must use
7. **Edge cases** — purely metaphysical/cosmological verses that need different prompting

### Output
`data/gita_analysis.md` — reference document used across all phases.

---

## Phase 2 — Enrichment (`scripts/enrich.py`)

### Goal
Add a `meaning_fields` object to every verse — this is the **primary text we embed**.
Raw fields (translation alone, chunked purport alone) are not good enough for semantic retrieval.

### Why raw fields fail
A user query: *"I'm paralyzed by fear of making the wrong decision"*
needs to match verse 2.47: *"You have a right to perform your duties, not to the fruits"*

These share almost zero vocabulary. Only a semantic bridge written in modern emotional language bridges this gap.

### The 4-field structured output

Per verse, Claude generates:

```json
{
  "situations": "Real-world situations a modern person would be in when they need this verse. Specific, uses everyday language.",
  "teaching":   "The core message of this verse in plain modern English. No Sanskrit terms.",
  "emotions":   "Emotional states, feelings, or mental conditions this verse directly addresses.",
  "concepts":   "Philosophical or spiritual concepts this verse introduces or explains."
}
```

Different fields catch different user query angles:

| User query type | Matched by |
|---|---|
| *"I'm anxious about failing"* | `emotions` |
| *"what is the nature of the soul"* | `concepts` |
| *"how do I stop overthinking"* | `situations` |
| *"karma yoga teaching"* | `teaching` |

### Chapter-aware prompting

Each verse prompt includes the chapter theme as context — Claude is not asked to infer the frame from scratch:

```python
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
```

### `text_for_embedding` field

After enrichment, each verse gets a concatenated field used as the meaning embedding input:

```python
text_for_embedding = (
    f"{fields['situations']}\n"
    f"{fields['teaching']}\n"
    f"{fields['emotions']}\n"
    f"{fields['concepts']}"
)
```

### Enrichment implementation details
- Batch with prompt caching (Anthropic `cache_control`) to minimize cost
- Structured JSON output with schema validation per response
- Resume support — skip verses already enriched
- Error logging with manual retry on failures
- One-time cost: ~627 Claude API calls ≈ $0.50–$1.00 total

### Output
`data/gita_enriched.json` — adds `meaning_fields` and `text_for_embedding` to every verse.

---

## Phase 3 — Indexing (`scripts/indexer.py`)

### Embedding Model: BGE-M3

**Model:** `BAAI/bge-m3`

| Property | Value |
|---|---|
| Dimensions | 1024 |
| Languages | 100+ (handles Sanskrit, Devanagari, English) |
| Retrieval modes | Dense + Sparse + Multi-vector (ColBERT) |
| Size | ~570MB |
| Cost | Free, runs locally |

Why BGE-M3:
- Multilingual — understands Sanskrit transliteration and Devanagari directly
- Dense catches semantic meaning; sparse catches exact terms (`"2.47"`, `"karmanye vadhikaraste"`, `"dharma"`)
- Best-in-class free model on MTEB retrieval benchmarks
- One model handles both retrieval modes — no separate BM25 process needed

### Vector Store: ChromaDB

- Pure Python, zero infrastructure, persists to disk at `data/chroma_db/`
- Easy to inspect locally, swappable for a hosted store later

### Hybrid Search Architecture

BGE-M3 produces two outputs per text in a single forward pass:

```python
output = model.encode(text)
output["dense_vecs"]       # 1024-dim vector  → stored in ChromaDB
output["lexical_weights"]  # sparse token weights → stored in sparse index (disk)
```

At query time, both are searched independently then fused:

```
dense_results  (top_k=20)  ─┐
sparse_results (top_k=20)  ─┤── Reciprocal Rank Fusion (RRF)
                              │   score = 1/(60 + dense_rank) + 1/(60 + sparse_rank)
                              ▼
                    unified ranked list
```

**Sparse index storage:** serialized dict `{token: {doc_id: weight}}` saved to
`data/sparse_index.pkl` alongside ChromaDB. Loaded into memory at app startup.

### Multi-vector Strategy

Every verse produces multiple vectors across two ChromaDB collections:

```
Verse 2.47
├── gita_verses collection
│   ├── "2.47_meaning"      → dense(text_for_embedding)    [type: meaning]
│   └── "2.47_translation"  → dense(translation)           [type: translation]
│
└── gita_purport collection
    ├── "2.47_purport_0"    → dense(header + paragraph_1)  [type: purport_chunk]
    ├── "2.47_purport_1"    → dense(header + paragraph_2)  [type: purport_chunk]
    └── "2.47_purport_2"    → dense(header + paragraph_3)  [type: purport_chunk]
```

Why 3 vector types:

| Query | Matched by |
|---|---|
| *"fear of failure, paralyzed by duty"* | `meaning` vector |
| *"right to perform duties not fruits"* | `translation` vector |
| *"Arjuna disciplic succession"* | `purport_chunk` vector |
| *"karmanye vadhikaraste"* | sparse index |
| *"what does 2.47 say"* | sparse + `translation` vector |

### Semantic Chunking (Paragraph-Based + Parent-Child)

Prabhupada's purports are written in structured paragraphs — each paragraph is one complete thought.
We respect those boundaries instead of cutting at arbitrary word counts.

**Chunking rules:**
```
Raw purport
    │
    ▼
Split on paragraph breaks (\n\n)
    │
    ├── paragraph < 40 words?   → merge with next paragraph (too short, incomplete thought)
    ├── paragraph 40–350 words? → keep as-is, one child chunk  ← ideal
    └── paragraph > 350 words?  → split at sentence boundaries into ~200-word sub-chunks
```

**Parent-child structure:**
- **Child chunk** = single paragraph → used for *retrieval* (precise, high-scoring match)
- **Parent chunk** = that paragraph + one paragraph before + one after → returned to Claude for *generation context*

```
Purport paragraphs: [P1] [P2] [P3] [P4] [P5]

Child retrieved: P3
Parent sent to Claude: P2 + P3 + P4   ← full argument context
```

Child chunks are prefixed with the verse header for embedding context:
```
"Verse 2.47: You have a right to perform your prescribed duties...\n\n{paragraph_text}"
```

### Metadata Schema

**`gita_verses` collection:**
```python
{
  "id": "2.47_meaning",
  "document": "<text_for_embedding or translation>",
  "metadata": {
    "verse_id":    "2.47",
    "chapter":     2,
    "verse":       47,
    "type":        "meaning",        # or "translation"
    "devanagari":  "...",
    "sanskrit":    "...",
    "translation": "...",
  }
}
```

**`gita_purport` collection:**
```python
{
  "id": "2.47_purport_0",
  "document": "Verse 2.47: ...\n\n<child_chunk_text>",
  "metadata": {
    "verse_id":     "2.47",
    "chapter":      2,
    "verse":        47,
    "type":         "purport_chunk",
    "chunk_index":  0,
    "parent_start": 0,              # paragraph indices for parent window
    "parent_end":   2,
    "translation":  "...",
  }
}
```

Full purport is NOT stored in metadata (size). Fetched from `gita_enriched.json` by `verse_id` at serve time using the parent window indices.

### Run
```bash
python scripts/indexer.py
```

Total vectors: ~1254 (verse-level) + ~3000–5000 (purport chunks) ≈ **4000–6000 vectors**

---

## Phase 4 — Search & RAG (`app/services/search.py`)

### Input Guardrail

Before hitting the pipeline, classify whether the query is on-topic.

```
Is this query related to: spiritual guidance, life situations, dharma,
emotions, philosophy, the Bhagavad Gita, or related themes?
```

Implementation: single fast Claude call, `max_tokens=5`, returns `"relevant"` or `"off_topic"`.
Adds ~150–200ms. Off-topic queries return a short redirect message without touching retrieval.

The classifier vocabulary is derived from Phase 1 (Gita analysis) so it knows the full thematic range.

### Query Transformation

Two transformations applied in parallel before retrieval:

#### HyDE (Hypothetical Document Embeddings)

Instead of embedding the raw query, generate a *hypothetical Gita commentary* that would answer it, then embed that.

**Why this works:** The hypothetical lives in the same semantic space as your indexed text.
Embedding a spiritual commentary matches indexed spiritual commentary far better than embedding a casual user message.

```
query:      "I'm terrified of making the wrong decision and ruining everything"

naive embed:  embed("I'm terrified of making the wrong decision...")
              → lands in self-help / casual language space
              → weak match against Gita commentary

HyDE embed:   Claude generates:
              "The Bhagavad Gita teaches that fear of consequences arises
               from attachment to outcomes. The soul is eternal — no action
               taken in accordance with one's dharma is ever wasted. One must
               surrender anxiety about results and act from a place of duty
               rather than fear of failure."
              embed(hypothetical) → lands in Gita commentary space → strong match
```

**HyDE prompt requirements (refined during Phase 1):**
- Output must sound like Srila Prabhupada's commentary style
- Must use the vocabulary and concepts present in the indexed corpus
- Must be 3–5 sentences, not a generic self-help answer
- Must not hallucinate verse references (no `"as stated in verse X"`)

#### Query Expansion

Generate 3 semantic rephrasings of the original query, retrieve for all 4 (original + 3), deduplicate and merge results.

```python
original:    "fear of failure"
expansion_1: "anxiety about outcomes and consequences"
expansion_2: "paralyzed by attachment to results"
expansion_3: "unable to act without certainty of success"
```

This improves recall significantly for short or ambiguous queries.

### Retrieval

Both HyDE vector and expansion vectors are used:

```
HyDE vector + expansion vectors
        │
        ▼
Dense search  → ChromaDB gita_verses  (top_k=15) ─┐
Dense search  → ChromaDB gita_purport (top_k=15) ─┤
Sparse search → sparse_index          (top_k=15) ─┤── all parallel
                                                   │
        ▼
RRF fusion across all results
        │
        ▼
Group by verse_id
verse score = max(meaning, translation, best_purport_chunk, sparse) scores
        │
        ▼
Top 10 verses → reranker
```

### Reranking

Cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores each `(query, verse_translation)` pair directly.
Cross-encoders read both texts together — far more accurate than vector similarity alone.

After reranking: apply **MMR (Maximal Marginal Relevance)** to ensure result diversity.
Prevents returning 5 verses all from the same chapter saying the same thing.

```python
# MMR: balance relevance vs diversity
score = λ * relevance_score - (1 - λ) * max_similarity_to_already_selected
# λ = 0.7 (favour relevance, penalise redundancy)
```

### RAG Generation

Claude receives a structured context block, not a raw dump:

```
SYSTEM:
You are a guide helping someone understand the Bhagavad Gita As It Is.
Answer using ONLY the provided verse and commentary. Do not add outside knowledge.
If the verse does not directly address the question, say so honestly.

USER:
Question: {user_query}

Verse {verse_id} — {translation}

Commentary:
{parent_chunk_text}   ← 3-paragraph parent window, not just the matched child
```

**Output faithfulness constraint** is baked into the system prompt — no separate validation call needed.

### Full Query Pipeline

```
user query
    │
    ▼
[input guardrail] — on-topic? if not → redirect message
    │
    ▼
[query transformation] — HyDE + expansion (parallel Claude calls)
    │
    ▼
[hybrid retrieval] — dense (ChromaDB) + sparse (sparse_index), parallel
    │
    ▼
[RRF fusion] — merge dense + sparse rankings
    │
    ▼
[group by verse_id] — collapse multi-vector hits per verse
    │
    ▼
[cross-encoder rerank] — top 10 → scored
    │
    ▼
[MMR] — top 5 diverse results
    │
    ▼
[RAG generation] — Claude with parent chunk context + faithfulness constraint
    │
    ▼
response + [evaluation logging]
```

### Response Schema
```json
{
  "results": [
    {
      "verse_id":    "2.47",
      "chapter":     2,
      "verse":       47,
      "devanagari":  "...",
      "sanskrit":    "...",
      "translation": "...",
      "score":       0.94,
      "ai_guidance": "..."
    }
  ],
  "query_meta": {
    "guardrail":       "relevant",
    "retrieval_ms":    120,
    "rerank_ms":       45,
    "generation_ms":   800
  }
}
```

---

## Phase 5 — Evaluation Framework (`scripts/evaluate.py`)

### Why This Exists
Without evaluation, every pipeline change is a guess. This is the feedback mechanism that tells you if things are actually improving.

### Layer 1 — Retrieval Quality (Offline)

**Golden dataset:** 80–100 hand-curated `(query, expected_verse_id)` pairs.

Sources:
- Hand-craft 40 pairs: write real questions, note which verse you'd expect
- Mine from enriched `situations` field: each verse's generated situations become queries with that verse as the expected answer
- Phase 1 query archetypes: use the analysis document's archetypes as seed queries

**Metrics:**
```
MRR@5   — Mean Reciprocal Rank: is the right verse in top 5? how high?
Recall@5 — what % of queries return the right verse in top 5?
NDCG@5  — normalised discounted cumulative gain (ranked quality score)
```

Run as: `python scripts/evaluate.py --dataset data/golden_dataset.json`

Re-run after every pipeline change. Results logged to `data/eval_results.json`.

### Layer 2 — Generation Quality (Online, Per Response)

After Claude generates guidance, a separate lightweight judge call checks:

```
Given the retrieved verse and the generated guidance:
1. Is the guidance faithful? (does it stay within what the text says?)
2. Does it answer the user's question?
Output: {"faithful": true/false, "relevant": true/false, "score": 1-5}
```

This runs async, does not block the response. Score is logged with the query.

### Layer 3 — System Metrics (Always Running)

Logged per request:
```
retrieval_latency_ms    (embed + search + rerank, per stage)
generation_latency_ms
faithfulness_score      (from Layer 2)
guardrail_result        (relevant / off_topic)
cache_hit               (true/false)
```

Rolling averages visible via `GET /metrics` endpoint.

---

## Phase 6 — Guardrails

### Input Guardrail (query classifier)

**Position:** first step in the pipeline, before any embedding or retrieval.

**Implementation:** fast Claude call with a tight prompt derived from Phase 1 vocabulary.
Returns `"relevant"` or `"off_topic"` in `max_tokens=5`.

```python
# approximate prompt shape (refined after Phase 1 analysis)
GUARDRAIL_SYSTEM = """
You are a classifier for a Bhagavad Gita search engine.
Decide if the user's query is related to: spiritual guidance, life situations,
dharma, karma, the soul, emotions, philosophy, meditation, devotion, duty,
attachment, liberation, or the Bhagavad Gita itself.

Reply with exactly one word: "relevant" or "off_topic".
"""
```

**Off-topic response:**
```json
{
  "error": "off_topic",
  "message": "Anugamana is designed for spiritual and philosophical guidance. Try asking about a life situation, an emotion, or a concept from the Bhagavad Gita."
}
```

### Output Guardrail (faithfulness constraint)

Baked into the RAG generation system prompt — no extra API call:

```
Only use information present in the provided verse and commentary.
Do not draw from outside knowledge.
If the verse does not directly address the query, say so honestly.
```

---

## Phase 7 — Feedback Loop

### v1 — Passive Logging (Build Now)

Every search response is logged to `data/feedback.db` (SQLite):

```sql
CREATE TABLE responses (
  id           INTEGER PRIMARY KEY,
  query        TEXT,
  hyde_query   TEXT,
  verse_ids    TEXT,     -- JSON array of returned verse_ids
  top_verse_id TEXT,
  faith_score  REAL,
  latency_ms   INTEGER,
  created_at   DATETIME
);
```

No user interaction required — just observability.

### v2 — Active Signal (When Frontend Exists)

User sees result → thumbs up 👍 or thumbs down 👎 button.

```sql
CREATE TABLE feedback (
  response_id  INTEGER REFERENCES responses(id),
  rating       INTEGER,   -- +1 or -1
  created_at   DATETIME
);
```

### How the Loop Closes

```
Weekly review of 👎 queries
        │
        ├── retrieval failure?    → add to golden dataset as hard negative case
        ├── generation failure?   → review RAG / HyDE prompt
        └── guardrail rejection?  → check if it was a false positive, refine classifier
                │
                ▼
        Golden dataset grows from real failures
                │
                ▼
        MRR re-measured → improvement is now quantified, not guessed
```

---

## Phase 8 — Sarvam AI Integrations

> Sarvam AI is an Indian AI company built specifically for Indic languages.
> With ~$1000 credit, this is a first-class part of the pipeline — not an afterthought.
> It makes Anugamana accessible to the actual audience of the Bhagavad Gita.

### Sarvam Product Map

| Product | What it does | API endpoint |
|---|---|---|
| **Bulbul** | Text-to-speech — Indian languages + Sanskrit | `/text-to-speech` |
| **Saaras** | Speech-to-text — Indian languages | `/speech-to-text` |
| **Mayura** | Translation between Indian languages + English | `/translate` |
| **Sarvam-2B** | LLM fine-tuned on 10 Indic languages | `/chat/completions` |
| **Indic Embeddings** | Embedding model for Indic scripts | `/embeddings` |
| **Transliteration** | Convert between Devanagari ↔ Roman scripts | `/transliterate` |

---

### Integration 1 — Sanskrit Verse Recitation (Bulbul TTS)

**The most impactful, most unique feature of this entire project.**

Western TTS (Google, ElevenLabs, OpenAI) handles Sanskrit poorly — wrong stress, wrong vowel lengths, wrong pronunciation of anusvara and visarga. Sarvam's Bulbul was trained on Indic scripts and produces natural Sanskrit recitation.

**What it enables:**
- Every verse returned includes an audio URL for the Sanskrit/Devanagari recitation
- User hears the verse pronounced correctly — a completely different experience from reading

**Implementation:**
```python
# app/services/sarvam_tts.py
async def get_verse_audio(verse_id: str, devanagari: str) -> str:
    # check audio cache first
    cached = audio_cache.get(verse_id)
    if cached:
        return cached

    audio = await sarvam_client.tts(
        text=devanagari,
        target_language_code="hi-IN",   # Sanskrit routed through Hindi voice
        speaker="meera",                 # female, clear pronunciation
        model="bulbul:v1"
    )
    # save to data/audio_cache/{verse_id}.mp3
    path = save_audio(verse_id, audio)
    audio_cache.set(verse_id, path)
    return path
```

**Audio cache:** TTS is called once per verse and cached permanently to `data/audio_cache/`. 627 verses = 627 audio files generated once at index time, not per request.

**Pre-generation:** Run `scripts/generate_audio.py` once after indexing to pre-generate all verse audio. No TTS latency at query time.

---

### Integration 2 — Multilingual Query Input (Mayura Translation)

**The second most impactful integration — expands the entire user base.**

The Bhagavad Gita's primary audience is Indian. A significant portion of that audience thinks and feels in Hindi, Bengali, Tamil, Telugu, Kannada, or Malayalam — not English. Forcing English-only queries is a massive barrier.

**Supported input languages:**
`hi` (Hindi), `bn` (Bengali), `ta` (Tamil), `te` (Telugu), `kn` (Kannada), `ml` (Malayalam), `gu` (Gujarati), `mr` (Marathi), `pa` (Punjabi), `or` (Odia)

**Pipeline position:** immediately after language detection, before guardrail

```
user query: "मुझे अपने जीवन के उद्देश्य के बारे में मार्गदर्शन चाहिए"
        │
        ▼
[language detect] → "hi" (Hindi)
        │
        ▼
[Mayura translate] → "I need guidance about my life's purpose"
        │
        ▼
[rest of pipeline runs in English internally]
        │
        ▼
[response generated in English by Claude]
        │
        ▼
[Mayura translate response back to Hindi]
        │
        ▼
user receives Hindi guidance
```

The entire retrieval pipeline stays English internally — translation is a thin wrapper at input and output. No pipeline changes required.

---

### Integration 3 — Multilingual Guidance Output (Mayura Translation)

Claude generates guidance in English. Mayura translates it to the user's detected language before returning.

```python
# app/services/sarvam_translate.py
async def translate(text: str, source: str, target: str) -> str:
    if source == target:
        return text
    return await sarvam_client.translate(
        input=text,
        source_language_code=source,
        target_language_code=target,
        model="mayura:v1",
        mode="formal"   # spiritual context → formal register
    )
```

**Response includes both languages:**
```json
{
  "ai_guidance": "You must act without attachment to outcomes...",
  "ai_guidance_translated": "आपको परिणामों की चिंता किए बिना कर्म करना चाहिए...",
  "response_language": "hi"
}
```

---

### Integration 4 — Guidance Audio (Bulbul TTS)

Not just the verse — the Claude-generated guidance can also be read aloud in the user's language.

Unlike verse audio (pre-cached), guidance audio is generated per-request since each response is unique.

```python
async def guidance_to_audio(text: str, language: str) -> str:
    audio = await sarvam_client.tts(
        text=text,
        target_language_code=language,   # e.g. "hi-IN", "ta-IN", "bn-IN"
        speaker="meera",
        model="bulbul:v1"
    )
    # temporary file, returned as base64 or presigned URL
    return encode_audio(audio)
```

Full audio response for the user: verse recitation (Sanskrit) + guidance reading (their language).

---

### Integration 5 — Voice Query Input (Saaras STT)

Users speak their question instead of typing it. Particularly valuable for:
- Older users less comfortable with typing
- Mobile users
- Users more comfortable speaking in Hindi than typing English

```
[user speaks in Hindi via mic]
        │
        ▼
[Saaras STT] → transcript in Hindi
        │
        ▼
[Mayura translate] → English
        │
        ▼
[pipeline runs normally]
```

**API endpoint added:** `POST /search/voice` — accepts audio file, returns same response schema as `/search`.

```python
async def transcribe(audio_bytes: bytes, language: str) -> str:
    return await sarvam_client.stt(
        file=audio_bytes,
        language_code=language,    # detected or user-specified
        model="saaras:v2"
    )
```

---

### Integration 6 — Indic Embeddings for Devanagari (4th Vector Type)

BGE-M3 handles 100+ languages but is a general multilingual model. Sarvam's embedding model is specifically fine-tuned on Indic scripts — it understands Sanskrit/Devanagari nuance at a deeper level.

**Use case:** Add a 4th vector type per verse using Sarvam embeddings on the `devanagari` field. Catches users who search using Devanagari script or Sanskrit terms directly.

```
Verse 2.47
├── gita_verses collection
│   ├── "2.47_meaning"       → BGE-M3 embed(text_for_embedding)
│   ├── "2.47_translation"   → BGE-M3 embed(translation)
│   └── "2.47_devanagari"    → Sarvam embed(devanagari)          ← NEW
│
└── gita_purport collection
    └── purport chunks...
```

**New ChromaDB collection:** `gita_devanagari` — Sarvam embeddings only (different dimension from BGE-M3, must be separate collection).

At query time: if Sarvam detects the query contains Devanagari/Sanskrit, search `gita_devanagari` too and fuse with RRF.

---

### Integration 7 — Hindi Enrichment (Sarvam-2B LLM)

The enrichment phase (Phase 2) generates `meaning_fields` in English only. With Sarvam-2B, generate a parallel Hindi version of the same fields:

```json
{
  "meaning_fields": {
    "situations": "...",    // English
    "teaching":   "...",
    "emotions":   "...",
    "concepts":   "..."
  },
  "meaning_fields_hi": {
    "situations": "...",    // Hindi — generated by Sarvam-2B
    "teaching":   "...",
    "emotions":   "...",
    "concepts":   "..."
  }
}
```

**Why Sarvam-2B over translating the English fields with Mayura:**
Sarvam-2B understands the Gita in Hindi natively. It generates natural Hindi that reflects how a Hindi speaker would describe the verse — not a mechanical translation of the English output. The vocabulary, idioms, and framing are authentically Indic.

**Indexed as separate vectors:** `2.47_meaning_hi` → Sarvam embed(Hindi meaning text)

Hindi queries now match Hindi meaning vectors directly, without any translation step.

---

### Integration 8 — Regional Language Purport Summaries (Mayura)

The full purport (Prabhupada's commentary) is in English. Generate translated summaries for the top Indian languages and store them in `gita_enriched.json`:

```json
{
  "purport_summaries": {
    "en": "...",   // original
    "hi": "...",   // Mayura translation
    "bn": "...",   // Bengali
    "ta": "...",   // Tamil
    "te": "...",   // Telugu
  }
}
```

These are sent to Claude/Sarvam-2B as context when the user's language is detected — the model gets commentary in the user's own language, producing more natural and culturally resonant guidance.

---

### Integration 9 — Transliteration (Roman ↔ Devanagari)

Users may type Sanskrit in Roman script (`"karmanye vadhikaraste"`) or Devanagari (`"कर्मण्येवाधिकारस्ते"`). Transliteration normalises both to the same searchable form before retrieval.

```python
# at query time: if query contains Devanagari, also produce Roman form
# if query is Roman Sanskrit, also produce Devanagari form
# both forms searched against sparse index
roman  = transliterate(query, "devanagari", "roman")
devnag = transliterate(query, "roman", "devanagari")
# both added to sparse search
```

Also used in the frontend — display verse in whichever script the user prefers.

---

### Language Detection Architecture

All Sarvam integrations depend on knowing the query language. Language detection sits at the very top of the pipeline:

```python
# app/services/language_detect.py
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "mr": "Marathi",
    "pa": "Punjabi",
    "sa": "Sanskrit",    # Devanagari
}

async def detect_language(text: str) -> str:
    # Sarvam's language detection or heuristic (Devanagari unicode range check first)
    if contains_devanagari(text):
        return "hi"   # treat as Hindi/Sanskrit
    return await sarvam_client.detect_language(text)
```

---

### Updated Full Query Pipeline (With Sarvam)

```
user query (text or voice)
    │
    ├── voice? → [Saaras STT] → transcript
    │
    ▼
[language detect] → detected_lang
    │
    ├── non-English? → [Mayura] → English query (stored separately for response)
    ├── Sanskrit/Devanagari? → [Transliterate] → both forms for sparse search
    │
    ▼
[input guardrail] — on-topic?
    │
    ▼
[HyDE + query expansion]
    │
    ▼
[hybrid retrieval]
  Dense  → ChromaDB gita_verses + gita_purport   (BGE-M3)
  Sparse → sparse_index                           (BGE-M3)
  Indic  → ChromaDB gita_devanagari               (Sarvam, if Indic query)
    │
    ▼
[RRF fusion → group → rerank → MMR]
    │
    ▼
[RAG generation — Claude in English]
    │
    ├── [Mayura] translate guidance → detected_lang
    ├── [Bulbul] guidance audio → detected_lang
    │
    ▼
[Bulbul] verse audio → Sanskrit (pre-cached, instant)
    │
    ▼
response: verses + translations + guidance + audio
    │
    ▼
[evaluation logging]
```

---

### Updated Response Schema (With Sarvam)

```json
{
  "results": [
    {
      "verse_id":             "2.47",
      "chapter":              2,
      "verse":                47,
      "devanagari":           "कर्मण्येवाधिकारस्ते...",
      "sanskrit":             "karmaṇy evādhikāras te...",
      "translation":          "You have a right to perform...",
      "score":                0.94,
      "verse_audio_url":      "/audio/2.47.mp3",
      "ai_guidance":          "You must act without attachment...",
      "ai_guidance_hi":       "आपको परिणामों की चिंता किए बिना...",
      "guidance_audio_url":   "/audio/guidance_<hash>.mp3"
    }
  ],
  "query_meta": {
    "detected_language":  "hi",
    "original_query":     "मुझे मार्गदर्शन चाहिए",
    "translated_query":   "I need guidance",
    "guardrail":          "relevant",
    "retrieval_ms":       120,
    "rerank_ms":          45,
    "generation_ms":      800,
    "translation_ms":     150,
    "tts_ms":             200
  }
}
```

---

### Priority Matrix

| Integration | Impact | Effort | Credit usage | Build when |
|---|---|---|---|---|
| Bulbul TTS — verse audio | Very high | Low | Low (pre-cached) | Phase 8 |
| Mayura — query translation | Very high | Low | Medium | Phase 8 |
| Mayura — response translation | Very high | Low | Medium | Phase 8 |
| Bulbul TTS — guidance audio | High | Low | Medium (per-request) | Phase 8 |
| Saaras STT — voice queries | High | Medium | Medium | Phase 9 |
| Indic embeddings — 4th vector | Medium | Medium | Low | Phase 9 |
| Hindi enrichment (Sarvam-2B) | Medium | Medium | Medium | Phase 9 |
| Regional purport summaries | Medium | Low | High (627 × N langs) | Phase 9 |
| Transliteration | Low-Medium | Low | Negligible | Phase 8 |

**Phase 8** = build alongside search pipeline (core Sarvam features)
**Phase 9** = add after evaluation confirms baseline pipeline works

---

### New Services Needed

| Module | Responsibility |
|---|---|
| `app/services/language_detect.py` | Detect query language, Devanagari check |
| `app/services/sarvam_tts.py` | Bulbul TTS with audio cache |
| `app/services/sarvam_stt.py` | Saaras STT for voice input |
| `app/services/sarvam_translate.py` | Mayura translation in/out |
| `app/routes/voice.py` | `POST /search/voice` endpoint |
| `scripts/generate_audio.py` | Pre-generate all 627 verse audio files |

---

### New Data Files

| File | Description |
|---|---|
| `data/audio_cache/` | Pre-generated Sanskrit verse audio (627 .mp3 files) |
| `data/gita_multilingual.json` | Purport summaries in regional languages |

---

### Environment Variables to Add

```
SARVAM_API_KEY=...
```

---

```
vedabase.io
    │
    ▼
scripts/scraper.py          →  data/gita_full.json
    │
    ▼
scripts/analyze_gita.py     →  data/gita_analysis.md
    │                           (thematic map, HyDE vocab, eval seeds, guardrail vocab)
    ▼
scripts/enrich.py           →  data/gita_enriched.json
    │                           (meaning_fields EN + HI, text_for_embedding)
    ▼
scripts/indexer.py          →  data/chroma_db/          (BGE-M3 dense)
    │                       →  data/chroma_devanagari/  (Sarvam Indic embeddings)
    │                       →  data/sparse_index.pkl    (BGE-M3 sparse)
    ▼
scripts/generate_audio.py   →  data/audio_cache/        (Bulbul TTS, 627 verse files)
    │
    ▼
── REQUEST PATH ──────────────────────────────────────────────
    │
    ├── voice? → sarvam_stt.py (Saaras) → transcript
    ▼
language_detect.py              detect language
    │
    ├── non-English? → sarvam_translate.py (Mayura) → English query
    ├── Indic script? → transliterate both forms for sparse search
    ▼
guardrail.py                    on-topic check (Claude, max_tokens=5)
    ▼
hyde.py                         HyDE + query expansion (Claude)
    ▼
retrieval.py                    BGE-M3 dense + sparse + Sarvam Indic (parallel)
                                → RRF fusion → group by verse_id
    ▼
reranker.py                     cross-encoder + MMR → top 5
    ▼
rag.py                          Claude RAG (English, faithfulness constraint)
    │
    ├── sarvam_translate.py     translate guidance → user language (Mayura)
    └── sarvam_tts.py           guidance audio in user language (Bulbul)
    ▼
app/routes/search.py            POST /search  →  verses + audio + guidance
app/routes/voice.py             POST /search/voice
    │
    ▼
scripts/evaluate.py             MRR / Recall@5 / NDCG against golden dataset
data/feedback.db                passive logging → feedback loop
```

---

## Scripts to Build (in order)

| # | Script / Module | Phase | Status |
|---|---|---|---|
| 1 | `scripts/scraper.py` | 0 | Done |
| 2 | `scripts/analyze_gita.py` | 1 | To build |
| 3 | `scripts/enrich.py` | 2 | To build |
| 4 | `scripts/indexer.py` | 3 | To rebuild |
| 5 | `scripts/evaluate.py` | 5 | To build |
| 6 | `scripts/generate_audio.py` | 8 | To build |
| 7 | `app/services/guardrail.py` | 6 | To build |
| 8 | `app/services/hyde.py` | 4 | To build |
| 9 | `app/services/retrieval.py` | 4 | To rebuild |
| 10 | `app/services/reranker.py` | 4 | To rebuild |
| 11 | `app/services/rag.py` | 4 | To rebuild |
| 12 | `app/services/language_detect.py` | 8 | To build |
| 13 | `app/services/sarvam_tts.py` | 8 | To build |
| 14 | `app/services/sarvam_stt.py` | 9 | To build |
| 15 | `app/services/sarvam_translate.py` | 8 | To build |
| 16 | `app/routes/search.py` | 4 | To update |
| 17 | `app/routes/voice.py` | 9 | To build |

---

## Key Dependencies to Add

```
beautifulsoup4>=4.12.0      # scraper (already added)
FlagEmbedding>=1.2.0        # BGE-M3 (dense + sparse)
chromadb>=0.5.0             # vector store
sentence-transformers>=3.0  # cross-encoder reranker
sarvamai>=0.1.0             # Sarvam AI (TTS, STT, translation, LLM)
langdetect>=1.0.9           # fallback language detection
```

---

## Data Files (all git-ignored except .example)

| File | Description |
|---|---|
| `data/gita_full.json` | Raw scraped verses |
| `data/gita_analysis.md` | Gita thematic analysis — informs all prompts |
| `data/gita_enriched.json` | Verses + enriched meaning fields (English + Hindi) |
| `data/gita_multilingual.json` | Purport summaries in regional Indian languages |
| `data/chroma_db/` | ChromaDB dense vector store (BGE-M3) |
| `data/chroma_devanagari/` | ChromaDB Indic vector store (Sarvam embeddings) |
| `data/sparse_index.pkl` | BGE-M3 sparse index for keyword retrieval |
| `data/audio_cache/` | Pre-generated Sanskrit verse audio (627 .mp3 files) |
| `data/golden_dataset.json` | Hand-curated (query, expected_verse) eval pairs |
| `data/eval_results.json` | MRR / Recall@5 history per pipeline version |
| `data/feedback.db` | SQLite: query logs + user feedback |
| `data/scrape_checkpoint.json` | Scraper resume state (temporary) |

---

## Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| Embedding model | `BAAI/bge-m3` | Free, multilingual, dense+sparse in one model, 1024-dim, top MTEB scores |
| Vector store | ChromaDB + sparse index on disk | Zero infra, pure Python, swappable |
| What to embed | Synthetic meaning fields + translation + semantic purport chunks | Raw fields fail on emotional/abstract queries |
| Enrichment approach | Chapter-aware prompts + structured 4-field JSON | Gita too diverse for one generic prompt |
| Prompt engineering order | Phase 1 analysis → then write prompts | Can't engineer prompts for a domain you haven't studied |
| Vectors per verse | 3 types: meaning, translation, purport chunks | Each catches different query patterns |
| Purport chunking | Paragraph-based semantic chunks (not fixed word count) | Prabhupada's paragraphs are already semantic units |
| Parent-child retrieval | Child chunk for retrieval, 3-para window sent to Claude | Precision in finding + context in generating |
| Hybrid search | BGE-M3 dense + sparse, fused with RRF | Dense for meaning, sparse for exact Sanskrit/verse terms |
| Query transformation | HyDE + query expansion | HyDE closes vocabulary gap between casual queries and scholarly text |
| HyDE prompt | Gita-commentary style, derived from Phase 1 analysis | Hypothetical must embed in same space as indexed corpus |
| Reranking | Cross-encoder + MMR | Cross-encoder for accuracy, MMR for chapter diversity |
| Evaluation | Golden dataset + MRR@5 + LLM-as-judge faithfulness | Can't know if pipeline is good without measuring it |
| Guardrails | Input classifier (fast Claude call) + output constraint in prompt | Prevent garbage retrieval; ensure faithful generation |
| Feedback loop | Passive SQLite logging → weekly review → golden dataset growth | Continuous improvement grounded in real failures |
| Sanskrit TTS | Sarvam Bulbul (pre-cached per verse) | Western TTS mispronounces Sanskrit; Sarvam trained on Indic scripts |
| Multilingual input | Sarvam Mayura (translate query → English → pipeline) | Gita's audience is Indian; forcing English is a barrier |
| Multilingual output | Sarvam Mayura (translate Claude guidance → user's language) | Response should be in the user's language |
| Guidance audio | Sarvam Bulbul (per-request, user's language) | Full audio experience: verse in Sanskrit + guidance in their language |
| Voice input | Sarvam Saaras STT | Accessibility for users more comfortable speaking than typing |
| Indic embeddings | Sarvam embeddings on Devanagari field (4th vector type) | Better Sanskrit/Devanagari query matching than general multilingual model |
| Hindi enrichment | Sarvam-2B generates Hindi meaning_fields natively | Authentic Hindi framing vs mechanical translation of English output |
| Language detection | Sarvam detect + Devanagari unicode heuristic | Must know language before any translation or routing decision |
