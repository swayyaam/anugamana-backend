"""
Affective state classification, used as a retrieval signal.

Why this is not a feature bolt-on
---------------------------------
The enrichment gives every verse an `emotions` field written in modern everyday
language, and that field is already inside `text_for_embedding`. But a dense
similarity search blends it with three other fields, so a query whose defining
property is its *emotional* state competes against situational and conceptual
vocabulary for the same similarity budget.

The hypothesis this module exists to test: for affective queries — which are the
queries this whole project is about — an explicit emotion-matching arm added to
RRF fusion retrieves better than semantic similarity alone. That is condition C12
in the ablation grid, and it is a falsifiable claim, not a UI nicety.

The taxonomy
------------
Derived from the corpus rather than invented. Term frequencies across all 700
enriched `emotions` fields:

    anxiety 278 · frustration 239 · relief 198 · exhaustion 163 · fear 160
    shame 133 · despair 131 · doubt 115 · confusion 108 · dread 75
    desperation 73 · loneliness 77

Each label is then anchored to the Gita's own affective vocabulary, which is what
makes this defensible for the domain: chapter 1 is literally *Arjuna-viṣāda-yoga*,
the yoga of Arjuna's despair. The text has a theory of these states already; we
are aligning to it, not imposing a generic sentiment schema.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import structlog
from anthropic import AsyncAnthropic

from app.config import (
    ANTHROPIC_API_KEY,
    LLM_MODEL,
    SARVAM_CHAT_MODEL,
    SARVAM_ENABLED,
)
from app.services.sarvam.client import SarvamError, SarvamUnavailable
from app.services.sarvam.client import client as sarvam_client

logger = structlog.get_logger(__name__)

_claude = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


@dataclass(frozen=True)
class EmotionLabel:
    key: str
    gita_term: str
    gloss: str
    #: Text embedded to form the emotion arm's query vector. Written in the same
    #: register as the enrichment's `emotions` field so the two land close.
    probe: str


TAXONOMY: dict[str, EmotionLabel] = {
    label.key: label
    for label in [
        EmotionLabel(
            "anxiety", "cintā",
            "anticipatory dread about outcomes not yet arrived",
            "the anxiety of obsessing over how something will turn out, "
            "restless worry about consequences you cannot control",
        ),
        EmotionLabel(
            "despair", "viṣāda",
            "collapse of the will to act; Arjuna's state in chapter 1",
            "the despair of feeling that nothing you do matters, "
            "wanting to put down your responsibilities and withdraw entirely",
        ),
        EmotionLabel(
            "grief", "śoka",
            "sorrow at loss, especially of people",
            "grief after losing someone, mourning, the ache of absence "
            "and of things that cannot be undone",
        ),
        EmotionLabel(
            "fear", "bhaya",
            "dread of a specific threat, including death",
            "fear of what is coming, dread of failure, of death, "
            "of losing what you have",
        ),
        EmotionLabel(
            "doubt", "saṁśaya",
            "inability to settle on what is true or right",
            "doubt about whether any of this is true, second-guessing your path, "
            "being unable to commit to a decision",
        ),
        EmotionLabel(
            "confusion", "moha",
            "delusion; not seeing the situation as it is",
            "confusion about what is actually happening, feeling lost, "
            "unable to tell what is real from what you have told yourself",
        ),
        EmotionLabel(
            "anger", "krodha",
            "hostility arising from thwarted desire",
            "anger at someone who wronged you, resentment, "
            "irritation that keeps returning",
        ),
        EmotionLabel(
            "craving", "kāma",
            "desire and attachment to a particular outcome",
            "wanting something badly, craving recognition or possession, "
            "being unable to let go of what you want",
        ),
        EmotionLabel(
            "frustration", "aśānti",
            "agitation from repeated blocked effort",
            "frustration at effort that goes nowhere, "
            "trying repeatedly and getting the same result",
        ),
        EmotionLabel(
            "exhaustion", "glāni",
            "depletion; the weariness that precedes giving up",
            "exhaustion from carrying something too long, burnout, "
            "having nothing left to give",
        ),
        EmotionLabel(
            "shame", "lajjā",
            "self-directed judgment about one's own worth or conduct",
            "shame about what you have done or failed to do, "
            "feeling unworthy, wanting to hide",
        ),
        EmotionLabel(
            "loneliness", "eka-bhāva",
            "isolation; being unaccompanied in one's situation",
            "loneliness, feeling that nobody understands your situation, "
            "being surrounded by people and still alone",
        ),
        EmotionLabel(
            "equanimity", "samatva",
            "the settled state the text points toward",
            "steadiness, acceptance, peace that does not depend on outcomes, "
            "relief after letting something go",
        ),
        EmotionLabel(
            "seeking", "jijñāsā",
            "intellectual longing rather than distress",
            "wanting to understand something, philosophical curiosity, "
            "asking what the nature of things really is",
        ),
    ]
}

LABELS = tuple(TAXONOMY)

_SYSTEM = f"""\
You classify the emotional state expressed in a message, for a system that
retrieves passages from the Bhagavad Gita.

Choose from exactly these labels:
{", ".join(LABELS)}

Rules:
- primary: the single dominant state. If the message is a detached intellectual
  question with no distress, use "seeking".
- secondary: at most two further labels genuinely present, else an empty list.
- intensity: 1 (mentioned in passing) to 5 (acute and overwhelming).
- Judge only what the message expresses. Do not infer a backstory.

Respond with valid JSON only:
{{"primary": "...", "secondary": ["..."], "intensity": 1-5}}\
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class EmotionResult:
    primary: str | None = None
    secondary: list[str] = field(default_factory=list)
    intensity: int | None = None
    backend: str = "none"

    @property
    def detected(self) -> bool:
        return self.primary in TAXONOMY

    def probe_text(self) -> str:
        """Text to embed as the emotion arm's query vector."""
        if not self.detected:
            return ""
        parts = [TAXONOMY[self.primary].probe]
        parts += [
            TAXONOMY[key].probe for key in self.secondary if key in TAXONOMY
        ]
        return " ".join(parts)

    def as_dict(self) -> dict:
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "intensity": self.intensity,
            "gita_term": TAXONOMY[self.primary].gita_term if self.detected else None,
            "backend": self.backend,
        }


def _parse(text: str) -> dict:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"no JSON in emotion classifier output: {text[:120]!r}")
    return json.loads(match.group(0))


def _build(payload: dict, backend: str) -> EmotionResult:
    primary = str(payload.get("primary", "")).strip().lower()
    if primary not in TAXONOMY:
        raise ValueError(f"label {primary!r} is not in the taxonomy")

    secondary = [
        str(s).strip().lower()
        for s in (payload.get("secondary") or [])
        if str(s).strip().lower() in TAXONOMY
    ][:2]

    intensity = payload.get("intensity")
    intensity = int(intensity) if isinstance(intensity, (int, float)) else None
    if intensity is not None:
        intensity = max(1, min(5, intensity))

    return EmotionResult(primary, secondary, intensity, backend)


async def _classify_sarvam(query: str) -> EmotionResult:
    response = await sarvam_client.post(
        "/chat/completions",
        {
            "model": SARVAM_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query},
            ],
            "temperature": 0.0,
            "max_tokens": 120,
        },
    )
    content = response["choices"][0]["message"]["content"]
    return _build(_parse(content), "sarvam")


async def _classify_claude(query: str) -> EmotionResult:
    response = await _claude.messages.create(
        model=LLM_MODEL,
        max_tokens=120,
        system=_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )
    return _build(_parse(response.content[0].text.strip()), "claude")


async def classify(query: str, *, prefer: str = "auto") -> EmotionResult:
    """
    Classify the query's affective state.

    `prefer` selects the backend: "sarvam", "claude", or "auto" (Sarvam when
    configured, else Claude). The backend used is recorded on the result, because
    a result produced by a different classifier is not comparable and must not be
    silently pooled in an evaluation.
    """
    order: list[str] = []
    if prefer == "auto":
        order = ["sarvam", "claude"] if SARVAM_ENABLED else ["claude"]
    else:
        order = [prefer]

    for backend in order:
        try:
            if backend == "sarvam":
                return await _classify_sarvam(query)
            return await _classify_claude(query)
        except (SarvamUnavailable, SarvamError) as e:
            logger.info("emotion_sarvam_unavailable", error=str(e))
            continue
        except Exception as e:
            logger.warning("emotion_classify_failed", backend=backend, error=str(e))
            continue

    return EmotionResult()
