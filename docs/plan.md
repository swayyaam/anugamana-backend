# Anugamana — RAG Pipeline Plan

> Full original plan archived at `docs/archive/plan_v1_20260510.md`

---

## Current State (May 2026)

Phases 0–7 are complete. The core pipeline is fully built and working end-to-end.

| Phase | What | Status |
|---|---|---|
| 0 | Scraping — 700 verses | ✅ Done |
| 1 | Corpus analysis — 18 chapter analyses | ✅ Done |
| 2 | Enrichment — 700 × meaning_fields | ✅ Done |
| 3 | Indexing — 2603 vectors + sparse index | ✅ Done |
| 4 | Search pipeline — guardrail, HyDE, retrieval, rerank, RAG | ✅ Done |
| 5 | Evaluation — MRR@5, Recall@5, NDCG@5 | ✅ Done |
| 6 | Guardrails — input classifier + output faithfulness | ✅ Done |
| 7 | Feedback loop — SQLite logging, LLM judge, metrics endpoint | ✅ Done |
| 8 | Sarvam integrations | 🔲 Next |
| 9 | Sarvam advanced (voice, Indic embeddings, Hindi enrichment) | 🔲 After 8 |

---

## Immediate Action Items

- [ ] Create `data/golden_manual.json` — 40 hand-curated hard query-verse pairs
- [ ] Run `python scripts/build_dataset.py && python scripts/evaluate.py`
- [ ] Get MRR@5 numbers for the paper (baseline vs no_hyde vs full)
- [ ] Get `SARVAM_API_KEY` and add to `.env`

---

## Phase 8 — Sarvam AI Integrations 🔲 NEXT

> Sarvam AI is an Indian AI company built specifically for Indic languages.
> Makes Anugamana accessible to the actual audience of the Bhagavad Gita.

### Sarvam Product Map

| Product | What it does | API endpoint |
|---|---|---|
| **Bulbul** | Text-to-speech — Indian languages + Sanskrit | `/text-to-speech` |
| **Saaras** | Speech-to-text — Indian languages | `/speech-to-text` |
| **Mayura** | Translation between Indian languages + English | `/translate` |
| **Sarvam-2B** | LLM fine-tuned on 10 Indic languages | `/chat/completions` |
| **Indic Embeddings** | Embedding model for Indic scripts | `/embeddings` |
| **Transliteration** | Convert between Devanagari ↔ Roman scripts | `/transliterate` |

### Priority Matrix

| Integration | Impact | Effort | Build in |
|---|---|---|---|
| Bulbul TTS — verse audio (pre-cached) | Very high | Low | Phase 8 |
| Mayura — query translation | Very high | Low | Phase 8 |
| Mayura — response translation | Very high | Low | Phase 8 |
| Bulbul TTS — guidance audio (per-request) | High | Low | Phase 8 |
| Transliteration (Roman ↔ Devanagari) | Medium | Low | Phase 8 |
| Saaras STT — voice queries | High | Medium | Phase 9 |
| Indic embeddings — 4th vector type | Medium | Medium | Phase 9 |
| Hindi enrichment (Sarvam-2B) | Medium | Medium | Phase 9 |
| Regional purport summaries | Medium | Low | Phase 9 |

### Services to Build (Phase 8)

| Module | Responsibility |
|---|---|
| `app/services/language_detect.py` | Detect query language + Devanagari unicode check |
| `app/services/sarvam_tts.py` | Bulbul TTS — verse audio (cached) + guidance audio (per-request) |
| `app/services/sarvam_translate.py` | Mayura — translate query in + guidance out |
| `scripts/generate_audio.py` | Pre-generate all 700 verse audio files (run once) |

### Services to Build (Phase 9)

| Module | Responsibility |
|---|---|
| `app/services/sarvam_stt.py` | Saaras STT for voice input |
| `app/routes/voice.py` | `POST /search/voice` endpoint |

### Environment Variable to Add

```
SARVAM_API_KEY=...
```

### Updated Full Query Pipeline (With Sarvam)

```
user query (text or voice)
    │
    ├── voice? → [Saaras STT] → transcript
    │
    ▼
[language detect] → detected_lang
    │
    ├── non-English? → [Mayura] → English query
    ├── Devanagari? → [Transliterate] → both forms for sparse search
    │
    ▼
[guardrail] — on-topic?
    │
    ▼
[HyDE + query expansion]
    │
    ▼
[hybrid retrieval] — BGE-M3 dense + sparse + Sarvam Indic (if Indic query)
    │
    ▼
[RRF → rerank → MMR]
    │
    ▼
[RAG — Claude in English]
    │
    ├── [Mayura] translate guidance → detected_lang
    ├── [Bulbul] guidance audio → detected_lang
    ▼
[Bulbul] verse audio → Sanskrit (pre-cached)
    │
    ▼
response: verses + audio + guidance (+ translated guidance)
    │
    ▼
[feedback logging]
```

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

### Integration Details

#### 1 — Sanskrit Verse Recitation (Bulbul TTS)
Pre-generate audio for all 700 verses once. Cached to `data/audio_cache/{verse_id}.mp3`.
No TTS latency at query time.

```python
# app/services/sarvam_tts.py
async def get_verse_audio(verse_id: str, devanagari: str) -> str:
    cached = audio_cache.get(verse_id)
    if cached:
        return cached
    audio = await sarvam_client.tts(
        text=devanagari,
        target_language_code="hi-IN",
        speaker="meera",
        model="bulbul:v1"
    )
    path = save_audio(verse_id, audio)
    audio_cache.set(verse_id, path)
    return path
```

#### 2+3 — Multilingual Query Input + Output (Mayura)
Translation is a thin wrapper at input and output. Pipeline stays English internally.

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
        mode="formal"
    )
```

#### 4 — Language Detection

```python
# app/services/language_detect.py
SUPPORTED_LANGUAGES = {
    "en": "English", "hi": "Hindi", "bn": "Bengali",
    "ta": "Tamil",   "te": "Telugu", "kn": "Kannada",
    "ml": "Malayalam", "gu": "Gujarati", "mr": "Marathi",
    "pa": "Punjabi", "sa": "Sanskrit",
}

async def detect_language(text: str) -> str:
    if contains_devanagari(text):
        return "hi"
    return await sarvam_client.detect_language(text)
```

#### 5 — Voice Input (Saaras STT) — Phase 9

```python
# app/services/sarvam_stt.py
async def transcribe(audio_bytes: bytes, language: str) -> str:
    return await sarvam_client.stt(
        file=audio_bytes,
        language_code=language,
        model="saaras:v2"
    )
```

#### 6 — Indic Embeddings (4th Vector Type) — Phase 9

Add `gita_devanagari` ChromaDB collection using Sarvam embeddings on the `devanagari` field.
At query time: if Devanagari detected, search this collection and fuse with RRF.

#### 7 — Hindi Enrichment (Sarvam-2B) — Phase 9

Run Sarvam-2B to generate Hindi `meaning_fields` natively (not translated from English).
Index as `2.47_meaning_hi` → Sarvam embed(Hindi meaning text).
Hindi queries match Hindi vectors directly without translation step.

---

## New Data Files (Phase 8+)

| File | Description |
|---|---|
| `data/audio_cache/` | Pre-generated Sanskrit verse audio (700 .mp3 files) |
| `data/gita_multilingual.json` | Purport summaries in regional Indian languages |
| `data/chroma_devanagari/` | ChromaDB Indic vector store (Sarvam embeddings) |

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
