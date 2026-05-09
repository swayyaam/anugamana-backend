# Anugamana — Business Plan

> "Anugamana" (अनुगमन) — following, going after, guiding along.
> An AI-powered spiritual guidance platform built on the Bhagavad Gita As It Is.

*All prices in Indian Rupees (₹). API costs kept in USD where billed in USD, with ₹ equivalent at ~₹94/dollar.*

---

## The Real Goal (Developer's Honest Perspective)

Cover running costs. Make enough extra to stay motivated to keep building.
That's it. No VC pitch, no hockey stick. Just a sustainable indie product.

**Target:** Cover ₹7,500/month in costs + earn ₹15,000–20,000/month extra = **~₹22,500–27,500/month total**
**How many paid users needed:** ~75–90 users at ₹299/month.

That's it. 75–90 people who genuinely find value. Completely achievable for a niche spiritual app with a dedicated community like ISKCON.

---

## What Is This Product

A freemium app where users describe a life situation, an emotion, or a question in natural language — in any Indian language or English — and receive the most relevant Bhagavad Gita verses with AI-generated personalised guidance, Sanskrit audio recitation, and translated responses in their own language.

**The core experience:**
```
User: "I keep failing at work and feel like giving up"
        ↓
Anugamana finds: Bg 2.47, Bg 3.19, Bg 18.48
        ↓
Plays: Sanskrit verse audio
        ↓
Shows: Translation + Prabhupada's commentary
        ↓
Claude says: "You are not defined by outcomes. The Gita teaches..."
        ↓
User's language: Hindi / Tamil / Bengali / English
```

No other Gita app does this. Most are glorified PDFs.

---

## Target Audience

### Primary
| Segment | Size | Willingness to Pay |
|---|---|---|
| ISKCON devotees (global) | ~1M active practitioners | High — deep connection to Prabhupada's Gita |
| Hindu diaspora (US, UK, Canada, Australia) | ~10M | High — disposable income, spiritual interest |
| Yoga practitioners (non-Indian) | ~5M | Medium — spiritually curious, English-first |

### Secondary
| Segment | Notes |
|---|---|
| Indian domestic users | Large volume, moderate price sensitivity |
| Students of Indian philosophy | Academia, universities |
| Therapists / life coaches | Incorporating spiritual frameworks |
| Temple and yoga studio operators | Institutional use |

### Key Insight
The ISKCON community is global, tightly networked, and has deep loyalty to Srila Prabhupada's specific translation. They will pay for a tool that serves the text they revere. The multilingual angle (Hindi, Bengali, Tamil, Telugu) opens up the Indian domestic market which no competitor has touched with AI.

---

## Competitive Landscape

| Product | What it does | What it lacks |
|---|---|---|
| Bhagavad Gita App (iOS/Android) | Browse verses, keyword search | No AI, no audio, no personalisation |
| vedabase.io | Full text online | Web only, no guidance, no voice, no AI |
| Generic GPT wrappers | ChatGPT prompted with Gita | Not specific, no semantic search, hallucination-prone |
| Shloka / other Gita apps | Verse display, basic search | No AI guidance, English only |

**Anugamana's moat:**
1. Purpose-built semantic search — not a generic LLM wrapper
2. Authentic Sanskrit TTS via Sarvam Bulbul — no competitor has this
3. Multilingual RAG — Hindi/Tamil/Bengali guidance, first in market
4. Grounded, faithful AI guidance — explicitly constrained to the text
5. Production-grade retrieval (BGE-M3, HyDE, reranking) — search quality competitors cannot match

---

## Real Monthly Costs (What You're Actually Paying)

*API services are billed in USD — converted to ₹ at ₹84/dollar.*

| Cost Item | Monthly (USD) | Monthly (₹) | Notes |
|---|---|---|---|
| Render hosting (Standard, 2GB RAM) | $25–50 | ₹2,350–4,700 | BGE-M3 needs at least 2GB RAM |
| Claude API | $10–30 | ₹940–2,820 | ~₹0.19/search; depends on usage |
| Sarvam API | $0 | ₹0 | Covered by $1,000 credit (~18 months) |
| Domain + SSL | $1 | ₹94 | ~₹1,128/year |
| Monitoring (Sentry free tier) | $0 | ₹0 | Free tier sufficient early on |
| **Total** | **~$36–81/month** | **~₹3,400–7,600/month** | Round up to **₹7,500/month** to be safe |

**After Sarvam credit runs out** (~18 months), add ₹1,900–3,800/month for TTS + translation.
Budget for that now, plan before it hits.

---

## The Only Number That Matters

```
Monthly target = costs (₹7,500) + motivation money (₹15,000–20,000)
              = ₹22,500–27,500/month

At ₹299/month per paid user:  need 76–92 users

Call it: ~80 paid users at ₹299/month = costs covered + ₹16,420/month extra
```

This is a realistic 6–12 month goal for a niche app with a dedicated community.

---

## Freemium Model — Kept Simple

Two tiers only. Don't over-engineer this early.

### The Principle
The sacred text is always free. The AI, audio, and depth are what users pay for.

---

### Tier 1 — Seva (Free · Forever)
> *Seva = selfless service*

**Goal:** Let anyone experience the quality of the search. Make them want the AI guidance.

| Feature | Limit |
|---|---|
| Semantic search | 5 searches/day |
| Results per search | Top 3 verses |
| Translation + Sanskrit + Devanagari | ✅ Always free |
| AI-generated guidance | ❌ |
| Sanskrit audio recitation | ❌ |
| Multilingual support | ❌ |
| Bookmarks | ❌ |
| Daily verse | ❌ |

5 searches/day is enough to experience the product. Not enough to rely on it daily — that nudge matters.

---

### Tier 2 — Sadhana (₹299/month)
> *Sadhana = daily spiritual practice*

**Annual: ₹2,499/year** *(save ₹1,089 — ~30% off)*

**Goal:** Everything a serious daily practitioner needs. One price, no artificial limits.

| Feature | Included |
|---|---|
| Semantic search | ✅ Unlimited |
| Results per search | ✅ Top 5 verses |
| Translation + Sanskrit + Devanagari | ✅ |
| AI-generated guidance (Claude) | ✅ |
| Sanskrit audio recitation (Sarvam Bulbul) | ✅ All verses |
| Multilingual support | ✅ Hindi + 1 more language |
| Bookmarks / saved verses | ✅ Unlimited |
| Daily verse + guidance | ✅ |
| Reflection journal | ✅ |
| Advanced filters (chapter, emotion, theme) | ✅ |
| Voice input (Saaras STT) | ✅ |
| Guidance audio in your language | ✅ |

**One price. Everything in.** No "you need the next tier for that" friction.

---

### When to Add a Third Tier

Only when real usage data shows a segment wanting something specific — institutional use, API access, multi-seat. Don't create it until someone asks for it.

**Possible future tier — Ashram (temples, yoga studios, institutions):**
Multi-seat, admin dashboard, custom branding, API access, annual contract.
Build this when a temple actually requests it, not before.

---

## Breakeven Math

*Based on ₹299/month Sadhana tier. Costs = ₹7,500/month.*

| Paid users | Monthly revenue | After costs | What it means |
|---|---|---|---|
| 26 | ₹7,774 | ₹274 | Break-even point |
| 40 | ₹11,960 | ₹4,460 | Costs covered + small extra |
| 60 | ₹17,940 | ₹10,440 | Costs covered + decent extra |
| **80** | **₹23,920** | **₹16,420** | **Sweet spot — sustainable + motivating** |
| 100 | ₹29,900 | ₹22,400 | Strong side income |
| 150 | ₹44,850 | ₹37,350 | Serious part-time income |
| 200 | ₹59,800 | ₹52,300 | Full freelance-replacement territory |

**First milestone to aim for:** 80 paid users.
That's the point where costs are covered and you have real motivation money coming in every month.

---

## Additional Revenue (Low Effort)

### 1. Annual Plan
Offer annual at ~27% off. You get upfront cash, user gets savings, both win.
```
Monthly plan: ₹299/month = ₹3,588/year
Annual plan:  ₹2,499/year  (saves user ₹1,089)
```

### 2. Dana (Voluntary Donation) on Free Tier
A quiet "support the project" button. No nag, no pressure. Some devotees will feel spiritually motivated to contribute.
```
"This service is offered freely (seva). If it has helped you,
 consider making a small dana offering to keep it running."
→ One-time ₹200 donation option
```
Even 15 donations/month = ₹3,000 extra with zero effort.

### 3. API Access (Future)
Once the pipeline is proven, charge other developers to query it.
```
₹1,000 / 10,000 API calls
Wellness apps, meditation apps, coaching platforms could use this.
```

### 4. Ashram Tier (Future, When Demand Exists)
Temples, yoga studios, coaching institutes — multi-seat institutional plans.
Priced at ₹2,000–5,000/month per institution depending on seats.
Even 5 institutional clients = ₹10,000–25,000/month additional.

---

## Cost Structure (Detailed)

*API costs billed in USD — shown with ₹ equivalent.*

### Variable costs per search request

| API call | Cost (USD) | Cost (₹) | When triggered |
|---|---|---|---|
| Claude guardrail check | ~$0.0001 | ~₹0.009 | Every search |
| Claude HyDE generation | ~$0.0005 | ~₹0.047 | Every search |
| Claude RAG guidance | ~$0.002 | ~₹0.19 | Sadhana tier only |
| Sarvam Mayura translation | ~$0.002 | ~₹0.19 | Non-English queries |
| Sarvam Saaras STT | ~$0.01/min | ~₹0.94/min | Voice queries only |
| Sarvam Bulbul guidance TTS | ~$0.005 | ~₹0.47 | Per guidance response |
| **Total per Sadhana search** | **~$0.003–0.005** | **~₹0.28–0.47** | |

**Verse audio is a one-time cost** — 627 files generated once (~$3 / ₹282 total), cached forever. Zero per-request cost for Sanskrit recitation.

### Cost per paid user per month
Assuming a paid user does ~50 searches/month:
```
50 searches × ₹0.38 average = ₹19/user/month in API costs
User pays ₹299/month → ₹280 gross margin per user (~93.6%)
```
Margins are very healthy. API costs stay tiny relative to revenue even at scale.

### Fixed costs (monthly)
| Item | USD | ₹ |
|---|---|---|
| Render Standard hosting | $25–50 | ₹2,350–4,700 |
| Domain + monitoring | $1–10 | ₹94–940 |
| Sarvam credit (covers ~18 months) | $0 | ₹0 |
| **Total fixed** | **~$26–60** | **~₹2,444–5,640** |

---

## Legal Considerations ⚠️

### BBT Copyright — Most Important Risk

Srila Prabhupada's translation and purports (*Bhagavad Gita As It Is*) are **copyrighted by the Bhaktivedanta Book Trust International (BBT)**, not public domain.

The current scraper pulls from vedabase.io — the BBT's official website. Building a commercial product on this content without a license is a legal risk.

| Option | Notes |
|---|---|
| **License from BBT** | Contact BBT directly. This is the cleanest path. ISKCON has a history of licensing their content for educational use. |
| **Partnership with ISKCON/BBT** | Offer BBT a free Ashram tier or revenue share in exchange for a content license. The app serves their mission — spreading Gita knowledge. |
| **Use public domain translations** | The Sanskrit is ancient and public domain. Other translations exist. But the audience specifically values Prabhupada's version. |

**Recommended path:** Approach BBT/ISKCON early. Frame it as a technology partnership. They want the Gita to reach more people — you've built the tool to do that.

---

## Go-to-Market Strategy

### Phase 1 — ISKCON Community (Months 1–3)
- Share in ISKCON Facebook groups, WhatsApp satsang groups, Telegram channels
- Free tier for all ISKCON devotees who sign up via referral
- Direct outreach to 5–10 ISKCON temple presidents for pilot Ashram accounts
- Content angle: "We built an AI tool for Prabhupada's Gita" — this community self-distributes

### Phase 2 — Indian Diaspora (Months 4–6)
- Hindi/English social content (Instagram Reels, YouTube Shorts): verse + guidance clips
- Partner with Hindu cultural organisations, yoga studios
- Collaborate with Indian spiritual YouTube channels

### Phase 3 — Indian Domestic (Months 7–12)
- Full Hindi UI + multilingual guidance live
- Regional language support (Tamil, Telugu, Bengali)
- App Store / Play Store listing with regional language metadata
- Partner with Indian spiritual influencers

---

## Product Differentiation Summary

| What competitors offer | What Anugamana offers |
|---|---|
| Keyword search | Semantic search — finds verses by meaning, not words |
| Browse by chapter | "I feel lost" → right verse instantly |
| English only | 10 Indian languages |
| No audio | Sanskrit recitation with correct pronunciation |
| No AI | Claude guidance grounded faithfully in the text |
| Web/app only | Voice input — speak in Hindi, get answer in Hindi |
| One-size fits all | The verse finds you based on your situation |

---

## North Star Metric

**Weekly Active Seekers (WAS)** — users who perform at least one meaningful search per week.

Better than DAU because spiritual practice aligns to weekly rhythms (satsang, Sunday programs). A weekly active user is demonstrating genuine, recurring value from the product.

---

## Name & Positioning

**Anugamana** — following the path, being guided along.

**Tagline options:**
- *"The Gita knows what you're going through."*
- *"Ancient wisdom. Your language. Your situation."*
- *"Ask the Gita."*

The product should never feel like a chatbot or a tech product. It should feel like a wise guide who knows the scripture perfectly and speaks to you directly.
