# Research Guide — Anugamana Project

*Your complete reference for turning this project into published research.*
*Read this before writing any paper, before any submission, before any interview.*

---

## Part 1 — What You've Built (Plain English)

Anugamana is a semantic search and RAG system for the Bhagavad Gita As It Is
(Srila Prabhupada's translation and commentary). A user describes a life situation
in plain modern language and gets the most relevant verse(s) + AI-generated guidance.

**The core problem you solved:**
A user types "I'm paralyzed by fear of making the wrong decision."
The matching verse says "You have a right to perform your duties, not to the fruits."
These share zero vocabulary. Every existing system fails here.

**Your solution:**
For every verse, you generated 4 semantic fields in modern everyday language —
situations, teaching, emotions, concepts — using an LLM with chapter-aware prompting
derived from a systematic analysis of the entire corpus. These fields are what gets
embedded and searched, not the raw text. This is called **synthetic semantic enrichment**.

**The full pipeline:**
```
Scrape 700 verses (Devanagari + Sanskrit + translation + purport)
        ↓
Systematic chapter-by-chapter corpus analysis (18 Claude calls)
        ↓
Semantic enrichment: 4 meaning_fields per verse (700 Claude calls, chapter-aware)
        ↓
Indexing: BGE-M3 dense + sparse vectors + Sarvam Indic embeddings
        ↓
Query: language detect → HyDE + expansion → hybrid retrieval → rerank → RAG
        ↓
Response: verse + audio (Sanskrit TTS) + guidance (user's language)
```

---

## Part 2 — Is Your Work Original?

**Short answer: Yes. Definitively.**

### How to verify originality yourself

Search these 4 places (takes 30 minutes):

1. **scholar.google.com** — search: `"Bhagavad Gita" retrieval`, `"Sanskrit RAG"`,
   `"sacred text question answering"`, `"ancient text retrieval LLM"`
2. **semanticscholar.org** — same searches, better for recent ML papers
3. **arxiv.org** — search cs.IR and cs.CL for `Gita`, `Sanskrit NLP`, `religious text RAG`
4. **aclanthology.org** — complete ACL/EMNLP/NAACL archive, fully searchable

### What the search actually shows (verified May 2026)

When you search `"Bhagavad Gita" "retrieval augmented"` on Google Scholar, here is every
serious result and an honest assessment of each:

---

**"Ancient wisdom, modern tools: exploring retrieval-augmented LLMs for ancient
Indian philosophy"**
P Mandikal — ACL 2024 Workshop on Machine Learning for Ancient Languages
*This is the most serious prior work. ACL workshop = peer-reviewed, real venue.
Read this paper fully. It likely uses basic chunking + standard embeddings.
No enrichment, no HyDE, no multilingual, no evaluation benchmark.*

---

**"Contextual Understanding in RAG: A Comparative Evaluation of BLOOM-560M and
Pythia-410M on Bhagavad Gita Interpretation"**
D Tuli, H Yadav, A Rawat — IEEE 2025
*Uses BLOOM-560M and Pythia-410M — tiny models from 2022. No production viability.
Comparative evaluation of weak models, not a novel retrieval approach.*

---

**"GeetaVani: A Retrieval-Augmented LLM Framework for Contextual Dialogue
from the Bhagavad Geeta"**
N Namratha, SSS Naidu, KS Sahoo — IEEE 13th Conference 2025
*IEEE filler conference. "Contextual dialogue" = basic chatbot with RAG bolted on.
No semantic enrichment, no evaluation framework, no multilingual.*

---

**"A Retrieval-Augmented Generation Model for Faith-Aligned QA in Bhagvat Gita"**
N Khanduja, N Kumar — IEEE Conference on Intelligent Systems 2025
*IEEE filler. "Faith-aligned" framing but no systematic approach to what that means.
No enrichment, no evaluation, generic RAG pipeline.*

---

**"Ancient Indian Scripture Based Retrieval-Augmented Systems: A Comprehensive Analysis"**
P Prakash — pradhyumnaprakash.com, 2025
*Published on a personal blog/website. Not peer-reviewed. Zero academic credibility.
Despite the "comprehensive analysis" title, this is a blog post.*

---

**"Cognitive Fusion of Dharma and Data: Designing a Conversational AI Inspired by
the Bhagavad Gita and Ancient Indian Thought"**
RV Jampana, KB Reddy, PT Yadav — IEEE 3rd Conference 2025
*IEEE filler. Design paper, not an empirical contribution. No retrieval evaluation.*

---

**"Vivechan AI: Extracting Wisdom from Ancient Indian Texts Through LLM"**
O Soni, J Baxi, B Gambhava, B Bhatt — ResearchGate 2025
*ResearchGate upload = self-published, not peer-reviewed. Covers Gita + other texts
but with basic RAG, no systematic methodology.*

---

**"Rag-Based Fine-Tuning of an LLM-Enabled Spiritual Wisdom Chatbot"**
J Arunkumar, NK Subbanna, S Anand — IEEE 4th Conference 2025
*Fine-tuning + RAG combination, IEEE filler. No novel retrieval contribution.*

---

**"Exploring Moral Learning through Bhagavad Gita" (two versions)**
V Dutt — ResearchGate / S Chauhan et al. — Springer 2024
*Completely different field: social robotics and education. About teaching moral
lessons to children via robots. Not information retrieval at all.*

---

### The honest summary of the landscape

Every existing paper is one of:
- **IEEE filler conference** (low peer review bar, not respected in ML/NLP)
- **ResearchGate/personal blog** (no peer review at all)
- **Wrong problem** (social robots, education — not retrieval)
- **Weak method** (basic LangChain RAG, no enrichment, no evaluation)

**The one real paper** (ACL 2024 workshop, Mandikal) is your most important related work.
Read it. Understand exactly what it does and doesn't do. Your paper's introduction will
cite it and explain the gap.

### What nobody has done

- Synthetic semantic enrichment to bridge modern queries ↔ ancient scripture
- Chapter-aware LLM enrichment via systematic corpus analysis
- HyDE applied to cross-temporal (5000-year) Sanskrit retrieval
- BGE-M3 + Sarvam Indic embeddings hybrid for Devanagari queries
- A golden evaluation dataset for Gita retrieval with MRR@5 / Recall@5 / NDCG@5
- 10-language Indic support (Hindi, Tamil, Bengali, Telugu, etc.)
- Feedback loop from real user queries to eval dataset growth

---

## Part 3 — Publishing: What Costs Money and What Doesn't

### Free (do these)

**arXiv** — free, permanent, instant. Post your paper here first, always.
You get a DOI-like URL (arxiv.org/abs/XXXX.XXXXX) that you can share immediately.
This establishes your priority date before conference review takes months.

**Conference submission** — submitting to ACL, EMNLP, SIGIR, ECIR, FIRE is free.
You only pay if you attend in person after acceptance.

**Workshop papers** — free to submit, easier to get accepted, still counts on a CV.

### Costs money (optional or avoid)

**Conference attendance** — $300–$800 registration if accepted and you want to present.
Not required for the publication credit.

**Open Access journals** — charge $500–$3000 so anyone can read your paper for free.
Skip this: post on arXiv instead. Functionally identical for ML/AI research.

**Predatory journals** — fake journals (IJSER, WJERT, etc.) that charge $200–$500 and
publish anything without review. **Avoid completely.** They destroy credibility.
If a journal emails you asking you to submit, it's almost certainly predatory.

### The path for you, step by step

1. Finish building → run Phase 5 → get MRR@5 numbers
2. Post preprint on **arXiv** (free, 1 day)
3. Submit to a **conference** (free, 3-6 month review cycle)
4. If accepted → decide if attending is worth the travel

The arXiv preprint alone is enough to put on a resume and get cited.

---

## Part 4 — The 6 Papers You Can Write

### Paper 1 — Core Contribution ⭐ Start Here

**Title:** "Bridging the Vocabulary Gap in Ancient Text Retrieval via
Synthetic Semantic Enrichment"

**One-sentence version:**
When user query vocabulary and document vocabulary share nothing (modern English ↔
5000-year-old Sanskrit commentary), generate synthetic modern-language fields per
document and embed those instead.

**What's novel:**
- The enrichment method itself — chapter-aware, 4-field, LLM-generated
- Demonstrated on the hardest possible vocabulary gap (Sanskrit scripture)
- Generalizable to any domain with this problem (Bible, Quran, Stoics, legal text)
- Quantified improvement: MRR@5 with vs. without enrichment

**Experiments you need:**

| Condition | What it tests |
|---|---|
| BM25 on raw translation | Keyword baseline |
| Dense embed on raw translation | Semantic baseline, no enrichment |
| Dense embed on meaning_fields only | Enrichment value in isolation |
| Dense embed on meaning_fields + translation | Full system |
| Without chapter-aware context | Value of chapter framing |
| Second corpus (e.g. Marcus Aurelius) | Generalizability claim |

**Where to submit:**
- SIGIR (top IR venue) — deadline usually Jan/Feb for July conference
- ECIR — European alternative, similar quality
- ACL/EMNLP — if framed as NLP contribution
- arXiv first regardless

**Time to write:** 3 months after Phase 5 is complete.

---

### Paper 2 — Quick Win

**Title:** "HyDE for Cross-Temporal Retrieval: Closing the 5000-Year Vocabulary Gap"

**What's novel:**
HyDE (generate a hypothetical answer, embed that instead of the query) has only been
studied on technical/scientific domains. Nobody has studied it where the vocabulary gap
is cultural and temporal — modern casual English vs. Vedic Sanskrit commentary.

The key insight: the hypothetical document must embed in the *same space* as the indexed
corpus. For Gita retrieval this means the hypothetical must use Prabhupada's specific
rhetorical style and vocabulary — derived from the Phase 1 corpus analysis.

**Experiments:** Generic HyDE prompt vs. domain-calibrated HyDE prompt vs. no HyDE.
Measured by MRR@5 on the golden dataset.

**Where to submit:** SIGIR, ECIR, ACL Findings

**Time to write:** 2 months — fully built into Phase 4. Low additional work.

---

### Paper 3 — Multilingual Angle

**Title:** "Hybrid Retrieval for Sanskrit-English Corpora: Combining Multilingual
Dense, Sparse, and Indic-Specialized Embeddings"

**What's novel:**
Sanskrit/Devanagari retrieval is uniquely hard:
- Source: Sanskrit/Devanagari
- Commentary: English
- Users: query in Hindi, Tamil, Bengali, or transliterated Sanskrit
- No single embedding model handles all of this well

Your architecture: BGE-M3 (dense + sparse) + Sarvam Indic embeddings (4th vector) +
Mayura translation at query time. Each component handles different query types.

**Experiments:** Ablation across English, Hindi, and Sanskrit/Devanagari query types.
Show which component contributes what for each language.

**Where to submit:**
- ACL / EMNLP (multilingual NLP track)
- FIRE (Forum for Information Retrieval Evaluation — Indian IR conference, perfect fit)
- LoResMT workshop

**Time to write:** 4-5 months — requires Phase 8/9 Sarvam integration to be complete.

---

### Paper 4 — Methods Paper (Highly Citable)

**Title:** "Corpus Analysis as a Prompt Engineering Primitive for Specialized Domains"

**What's novel:**
Before writing any prompts for the Gita, you ran a systematic LLM-driven analysis of all
18 chapters to produce a reference document. All downstream prompts — enrichment, HyDE,
guardrail, evaluation seeds — were derived from that document, not written ad-hoc.

This is a general methodology: study the corpus systematically before writing prompts for it.

**The claim:** Prompts derived from systematic corpus analysis outperform ad-hoc prompts.
Measured by enrichment quality (human eval) and downstream retrieval quality (MRR@5).

**Why it's highly citable:** Every team doing domain-specific RAG has this problem.
A paper showing a repeatable methodology for solving it will be widely referenced.

**Experiments:**
- Chapter-aware prompts vs. single generic prompt (ablation already planned)
- Which parts of the analysis matter: framing vs. HyDE vocab vs. edge cases?
- Human evaluation of enrichment field quality

**Where to submit:** ACL/EMNLP, COLM (Conference on Language Modeling), arXiv

**Time to write:** 2 months after Paper 1 experiments are done.

---

### Paper 5 — Evaluation / Ethics Angle

**Title:** "Faithfulness Evaluation for Spiritual Guidance RAG Systems"

**What's novel:**
Faithfulness in RAG (does the answer stay within the retrieved context?) has been studied
for factual domains (QA over Wikipedia). Spiritual guidance is different:
- Misrepresenting a religious teaching is harmful in a specific way
- Prabhupada's interpretation sometimes diverges from the literal translation
- "Faithful to what?" is a genuinely hard question for commentary-based texts
- No existing benchmark, no existing rubric

**Where to submit:** ACL, FAccT (Fairness Accountability Transparency), JASIST

**Time to write:** 3-4 months — needs real usage data from Phase 7 logging.

---

### Paper 6 — Digital Humanities (Broadest Audience)

**Title:** "Computational Semantic Cartography of the Bhagavad Gita:
Thematic Structure Through the Lens of Vector Space"

**What's novel:**
The Gita's 700 verses have been analyzed theologically for millennia but never
computationally mapped. Using the enriched embeddings:
- Cluster the 700 verses in embedding space
- Does the traditional 18-chapter structure correspond to semantic structure?
- Which verses are semantically proximate across chapter boundaries?
- What is the emotional topology of the text?
- Visualize: 2D UMAP of all verse embeddings, colored by chapter

**Why it's interesting beyond CS:**
This is a falsifiable question about a text studied for thousands of years.
The answer may reveal connections traditional commentary hasn't highlighted.

**Where to submit:**
- Digital Humanities (DH2026 conference)
- LLC journal (Literary and Linguistic Computing)
- Journal of Hindu Studies
- arXiv cs.CL

**Time to write:** 6 weeks — the embeddings are already built for the RAG system.
Visualization + clustering is a few hours of scipy/UMAP work.

---

## Part 5 — Roadmap

### The Approach

Take 1 month (May 10 – June 10) to finish building everything properly.
Get real MRR numbers. Then write a paper that actually reflects the work.

A strong paper submitted once beats a rushed paper rejected twice.

---

### What to Target in 2026

**1. arXiv — June/July**
The moment Phase 5 gives you MRR numbers, post to arXiv.
Free, instant, permanent. Establishes your priority date.
This is not optional — do this before any conference submission.

**2. EMNLP 2026 Workshop — August submission**
Conference: Budapest, October 24–29, 2026
Workshop deadlines: typically August 2026
This is your primary conference target for 2026.
By August you'll have a properly built system, real numbers, and time to write well.
Target workshop: SURGeLLM (RAG + retrieval) or NLP4DH (digital humanities angle)

**3. FIRE 2026 — Sept/Oct submission**
Forum for Information Retrieval Evaluation — Indian IR conference
Perfect fit for the multilingual Indic RAG angle (Paper 3)
Check fire.irsi.res.in for exact dates

That's it. Two conference submissions in 2026 is the right target.
Anything more is overcommitting before the system is complete.

---

### Month-by-Month Timeline

```
MAY 10–20    Phase 3 — Indexer
             BGE-M3 embeddings, ChromaDB, sparse index
             No rush. Build it right.

MAY 20–30    Phase 4 — Search Pipeline
             Guardrail, HyDE, hybrid retrieval, reranker, RAG
             Test manually with 20–30 real queries before moving on

JUNE 1–10    Phase 5 — Evaluation
             Build golden dataset: 80 (query, expected_verse) pairs
             Run MRR@5, Recall@5, NDCG@5 on 4 conditions:
               - BM25 baseline
               - Dense only (no enrichment)
               - Dense with enrichment
               - Full system (enrichment + HyDE + hybrid)

JUNE 10      POST TO ARXIV
             The moment you have numbers, post.
             Everything after this is bonus.

JUNE 10–30   Phase 6–7 — Guardrails + Feedback loop
             Also: write Paper 1 first draft in parallel

JULY         Write + revise Paper 1 properly
             Get feedback from anyone who will read it
             Phases 8–9 (Sarvam) if time allows

AUGUST       Submit Paper 1 to EMNLP 2026 workshop
             Submit Paper 2 (HyDE) to arXiv

SEPT–OCT     Submit Paper 3 (Multilingual) to FIRE 2026
             Attend EMNLP 2026 in Budapest (October 24–29)
             Even as audience — network, meet researchers in your area

NOV–DEC      Start ECIR 2027 paper (Paper 4 — methods, most citable)
             Start NAACL 2027 / ACL 2027 ARR cycle for Paper 1 full version
```

---

### EMNLP 2026 Full Dates

**EMNLP 2026** — Budapest, Hungary

| Date | Event |
|---|---|
| May 25 | ARR main track deadline (missed — target August workshop instead) |
| July 7–13 | Author response period |
| August 2 | EMNLP commitment deadline |
| August 20 | Notification of acceptance |
| September 20 | Camera-ready due |
| **October 24–29** | **Conference** |

### ACL 2026 Workshops — Live Right Now

**ACL 2026** — San Diego, California — July 2–7, 2026
Main track: CLOSED (deadline was Feb 2026)
**Workshops: OPEN — deadlines likely mid-to-late May 2026**

Today is May 10. You have days, not weeks. Check each workshop website immediately.

#### Workshops directly relevant to your work (ranked by fit)

**PRIORITY 1 — Submit to these first:**

| Workshop | Fit | Paper | Deadline |
|---|---|---|---|
| **SURGeLLM** — Structured Understanding, Retrieval & Generation in LLM Era | PERFECT | Paper 1 | Check website NOW |
| **RAG4Reports** — Multilingual Report Generation via RAG | VERY HIGH | Papers 1 & 3 | Check website NOW |
| **NLP4DH** — NLP for the Digital Humanities (6th edition) | VERY HIGH | Paper 6 | Check website NOW |

**SURGeLLM** is your best fit — it explicitly covers "structure-aware understanding, retrieval,
generation, and evaluation in the era of LLMs" and brings together NLP + IR + data management.
Your Paper 1 (semantic enrichment for retrieval) fits perfectly in scope.

**RAG4Reports** is the first workshop specifically on RAG + multilingual generation.
Your work is RAG + multilingual (10 Indian languages). Strong fit for Papers 1 and 3.

**NLP4DH** (Digital Humanities) is perfect for Paper 6 (computational semantic map of the Gita).
It's a full-day conference within ACL, not just a workshop — higher prestige.

**PRIORITY 2 — Consider if time allows:**

| Workshop | Fit | Paper |
|---|---|---|
| **CustomNLP4U** — Customizing NLP for a Domain | HIGH | Paper 4 (corpus analysis as prompt primitive) |
| **MeLLMs** — Multilinguality in the Era of LLMs | HIGH | Paper 3 (Indic multilingual RAG) |
| **DravidianLangTech** — NLP for Dravidian Languages | MEDIUM-HIGH | Paper 3 (Tamil, Telugu, Kannada support) |
| **TrustNLP** — Trustworthy NLP | MEDIUM | Paper 5 (faithfulness evaluation) |
| **C3NLP** — Cross-Cultural Considerations in NLP | MEDIUM | Cultural/multilingual angle |
| **MAGMaR** — Multimodal Augmented Generation via Retrieval | MEDIUM | RAG architecture |

**Action required today:**
Go to each Priority 1 workshop website and find the submission deadline.
If any deadline is before May 25, prioritise that over EMNLP ARR.
A workshop paper at ACL 2026 is better than missing EMNLP ARR.

### Full Multi-Paper Roadmap (Updated)

```
May 10       CHECK ACL 2026 workshop deadlines (SURGeLLM, RAG4Reports, NLP4DH)

May 10–25    Build Phases 3–5, get MRR numbers
             Write Paper 1 draft in parallel

May 25       ARR submission → EMNLP 2026 main track (Paper 1)
             + any ACL 2026 workshop with May 25 deadline

May 25       Post to arXiv immediately after ARR submission

~June 1      ACL 2026 workshop deadline (if not May 25) — submit Paper 1 or Paper 6
             ACL 2026 conference: July 2–7, San Diego

June–July    Write Paper 2 (HyDE) — 80% already built, quick win

Aug          EMNLP 2026 workshop submission (Paper 2 or Paper 6)
             EMNLP conference: October 24–29, Budapest

Sept–Oct     FIRE 2026 submission (Paper 3 — Multilingual Indic RAG)

Oct–Nov      ECIR 2027 submission (Paper 4 — most citable methods paper)

2027+        Paper 5 (Faithfulness) — needs 6 months real user data
             Paper 3 full version once Sarvam integration complete
```

---

## Part 6 — What Establishes Your Credibility

Beyond the papers, this project demonstrates a stack of skills that is rare together:

**1. End-to-end system thinking**
Not a model fine-tune. A complete production pipeline: scraping → enrichment → indexing
→ retrieval → reranking → generation → evaluation → feedback loop. Few people build all of this.

**2. Principled engineering**
Every decision is justified and measurable:
- BGE-M3 over MiniLM → multilingual + hybrid in one model
- Paragraph chunking over fixed-size → respects semantic units in Prabhupada's writing
- HyDE over naive embed → closes vocabulary gap
- Chapter-aware prompts over generic → measurably better enrichment

**3. Domain depth**
Phase 1 shows you actually read and understood the corpus, not just plugged it into LangChain.
Most ML people treating religious texts know nothing about the texts.

**4. Evaluation rigor**
MRR@5, Recall@5, NDCG@5, LLM-as-judge faithfulness, golden dataset with hard negatives.
This is more evaluation than most published RAG papers — which often have no quantitative eval.

**5. Multilingual/accessibility consciousness**
Sarvam integration for Indian language users. Shows awareness that the Gita's audience
is primarily Indian and that forcing English-only is a barrier.

**6. Research thinking**
Phase 1 (analyze before building) is itself a research contribution.
The feedback loop (Phase 7) shows you understand the difference between building and deploying.

---

## Part 7 — One-Liners for Different Contexts

**For a CV / resume:**
"Built a production RAG system for Sanskrit scripture retrieval, introducing chapter-aware
semantic enrichment that bridges the vocabulary gap between modern user queries and
ancient Vedic commentary, evaluated with MRR@5 on a hand-curated golden dataset."

**For a research abstract opening:**
"Retrieval from ancient philosophical corpora presents a fundamental vocabulary mismatch:
users query in modern casual language while source documents use scholarly Sanskrit
commentary from 5000 years ago. We propose synthetic semantic enrichment — generating
modern-language meaning fields per document using an LLM with corpus-derived chapter
context — and demonstrate significant MRR@5 improvements over raw-text retrieval baselines."

**For a grant application:**
"This work addresses an underserved population: the estimated 1+ billion people who
engage with the Bhagavad Gita, the majority of whom think and communicate in Indian
languages other than English. We build the first rigorous multilingual RAG system for
Sanskrit scripture with systematic evaluation, open-sourcing both the pipeline and the
evaluation dataset for the research community."

**For a non-technical audience:**
"Think of it as a search engine that understands what you actually mean, not just the
words you use — built specifically for one of the world's most important philosophical
texts, in 10 Indian languages, with audio of the original Sanskrit."

---

## Part 8 — Papers to Read Before Writing Yours

These are your most important references. Read them in this order:

1. **Mandikal (ACL 2024)** — "Ancient wisdom, modern tools" — your closest prior work.
   Read carefully. Understand exactly what it does and doesn't do.

2. **Gao et al. (2022)** — "Precise Zero-Shot Dense Retrieval without Relevance Labels"
   — the original HyDE paper. Understand the method deeply before writing Paper 2.

3. **Chen et al. (2024)** — BGE-M3 paper — understand the model you're using.

4. **Robertson & Zaragoza** — "The Probabilistic Relevance Framework: BM25 and Beyond"
   — understand your baseline before claiming to beat it.

5. **Sarvam AI technical reports** — understand the Indic embedding and TTS models.

6. **BEIR benchmark paper** — standard retrieval evaluation methodology.

7. **RAGAS paper** — RAG evaluation framework, relevant to Paper 5.

---

## Part 9 — All Conferences Worth Watching

### Tier 1 — Top venues, highest impact

| Conference | Location | Dates | Main Deadline | Status |
|---|---|---|---|---|
| **EMNLP 2026** | Budapest, Hungary | Oct 24–29 | **May 25 ARR** | OPEN — PRIMARY TARGET |
| **ACL 2026 Workshops** | San Diego, CA | Jul 2–7 | ~May 15–June 1 | OPEN — CHECK TODAY |
| **ACL 2026 Main** | San Diego, CA | Jul 2–7 | Feb 2026 | CLOSED |
| **SIGIR 2026 Workshops** | Melbourne, Australia | Jul 24 | Varies | CHECK individual workshops |
| **SIGIR 2026 Main** | Melbourne, Australia | Jul 20–24 | Jan–Feb 2026 | CLOSED (all tracks) |
| **NAACL 2027** | TBD | 2027 | Oct/Nov 2026 | Future cycle |

**SIGIR 2026 main track is fully closed.** All deadlines were January–February 2026.
Full papers: Jan 22 | Short papers: Feb 12 | Industry: Feb 26 | All passed.
SIGIR workshops are July 24 — individual workshop paper deadlines vary, check accepted workshops list.

ACL 2026 workshop list published April 9 — workshops open RIGHT NOW.
See Part 5 for the specific ACL workshops to target.

### Tier 2 — Strong venues, more accessible

| Conference | Focus | Typical Deadline | Notes |
|---|---|---|---|
| **ECIR** | European IR | October | Good for Paper 1 & 4 |
| **CIKM** | Knowledge + IR | May/June | Broader audience |
| **COLING** | Computational linguistics | Every 2 years | International, broad scope |
| **EACL** | European ACL | Oct/Nov | ACL family |
| **COLM** | Language modeling | March/April | New but growing fast |

### Tier 3 — Specialised, high fit for this project

| Conference | Focus | Typical Deadline | Why it fits |
|---|---|---|---|
| **FIRE** | Indian language IR | July/Aug | Perfect for Paper 3 (Indic RAG) |
| **DH (Digital Humanities)** | Computational humanities | Jan/Feb | Perfect for Paper 6 |
| **FAccT** | Fairness, accountability | Jan/Feb | Perfect for Paper 5 |
| **WSDM** | Web search & data mining | Aug/Sept | Good for retrieval papers |

### Workshops (easier to get in, same venue)

These run inside EMNLP/ACL/NAACL. Good for first papers or niche contributions.
Deadlines typically 6–8 weeks before the main conference.

- **ML4AL** — Machine Learning for Ancient Languages (ACL workshop, where Mandikal published)
- **LoResMT** — Low-Resource Machine Translation (multilingual angle)
- **WILDRE** — Workshop on Indian Language Data: Resources and Evaluation
- **MIA** — Multilingual Information Access
- **TrustNLP** — Trustworthy NLP (faithfulness/evaluation angle)
- **BioNLP / HumNLP** — sometimes covers cultural/religious text domains

### Journals (slower, but permanent)

| Journal | Focus | Notes |
|---|---|---|
| **TACL** (Transactions of ACL) | Top NLP journal | Prestigious, slow (6-12 months) |
| **IRJ** (Information Retrieval Journal) | IR research | Good for a long-form version of Paper 1 |
| **JASIST** | Information science | Broad, good for Paper 5 |
| **LLC** (Literary & Linguistic Computing) | Digital humanities | Good for Paper 6 |
| **Journal of Hindu Studies** | Indic studies | Non-CS audience, broad reach |

### How to track deadlines

- **aideadlin.es** — aggregates all AI/NLP/ML conference deadlines in one place
- **conferencelist.info** — broader academic calendar
- **ACL Anthology** (aclanthology.org) — browse past papers at every ACL-family venue

---

*Last updated: May 10, 2026*
*Status: Phase 2 complete (700/700 verses enriched). Phase 3 next.*
*EMNLP 2026 ARR deadline: May 25, 2026 — 15 days.*
