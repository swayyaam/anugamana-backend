"""
Crisis routing — the branch that must run before retrieval (audit S-01).

Why this exists
---------------
The topical guardrail is deliberately generous: its prompt tells the model that
"personal struggles, existential questions, and moral dilemmas all count". A user
in crisis therefore classifies as *relevant*, and the pipeline does exactly what
it was built to do — retrieves verses about the soul being eternal and the body
being temporary, then generates warm, personalised guidance grounded in them.

That is a reachable and genuinely harmful output. Crisis queries must never reach
retrieval or generation.

Design
------
Two stages, cheap first:
  1. A lexical prefilter over unambiguous risk phrasings. Zero latency, zero cost,
     catches the blunt cases, and works when the API is down.
  2. An LLM classifier for everything the prefilter is unsure about, with an
     explicit instruction to err toward caution.

Failure mode is deliberately asymmetric: the topical guardrail fails *open* (a
false off-topic call merely annoys someone), this one fails *safe* on the
prefilter and *open* on the model call — a lexical hit is never overridden by an
API result, and an API outage cannot suppress a lexical hit.

The response is a fixed string. It is never LLM-generated, so it cannot drift.
"""

import re

import structlog
from anthropic import AsyncAnthropic

from app.config import ANTHROPIC_API_KEY, LLM_MODEL

logger = structlog.get_logger(__name__)

_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# Unambiguous risk phrasings. Kept deliberately narrow — the model stage handles
# nuance. Every pattern here should be one a reasonable person would escalate on.
_CRISIS_PATTERNS = [
    r"\bkill(?:ing)?\s+my\s?self\b",
    r"\bkms\b",
    r"\bend(?:ing)?\s+(?:my|it)\s+(?:life|all)\b",
    r"\btake\s+my\s+own\s+life\b",
    r"\bsuicid(?:e|al)\b",
    r"\bwant\s+to\s+die\b",
    r"\bwish\s+i\s+(?:was|were)\s+dead\b",
    r"\bbetter\s+off\s+(?:dead|without\s+me)\b",
    r"\bdon'?t\s+want\s+to\s+(?:live|be\s+alive|exist)\b",
    r"\bno\s+reason\s+to\s+(?:live|go\s+on)\b",
    r"\bharm(?:ing)?\s+my\s?self\b",
    r"\bhurt(?:ing)?\s+my\s?self\b",
    r"\bcut(?:ting)?\s+my\s?self\b",
    r"\boverdos(?:e|ing)\b",
    r"\bhang\s+my\s?self\b",
    r"\bjump\s+(?:off|from)\s+(?:a\s+)?(?:bridge|building|roof|terrace)\b",
]
_CRISIS_RE = re.compile("|".join(_CRISIS_PATTERNS), re.IGNORECASE)

CRISIS_SYSTEM = """\
You are a safety classifier for a spiritual guidance application.

Decide whether the user's message indicates risk of suicide, self-harm, or an
acute mental-health crisis — including indirect expressions such as hopelessness
about continuing to exist, wanting to disappear permanently, or planning to hurt
themselves.

Do NOT flag: ordinary grief, sadness, burnout, existential or philosophical
questions about death and the soul, questions about the Bhagavad Gita's teachings
on mortality, or someone asking about a character in the text.

When genuinely uncertain, answer crisis — a cautious false positive is far less
costly than a missed one.

Reply with exactly one word: crisis   or   safe\
"""

#: Fixed response. Never model-generated.
CRISIS_RESPONSE = (
    "It sounds like you may be going through something very painful right now, and "
    "I don't want to answer that with a verse.\n\n"
    "Please talk to someone who can properly support you:\n\n"
    "• India — Tele-MANAS: 14416 or 1-800-891-4416 (24/7, free, 20 languages)\n"
    "• India — AASRA: +91 91529 87821 (24/7)\n"
    "• India — Vandrevala Foundation: 1860 266 2345 or +91 99996 66555 (24/7)\n"
    "• US — 988 Suicide & Crisis Lifeline: call or text 988\n"
    "• UK & ROI — Samaritans: 116 123\n"
    "• Elsewhere — findahelpline.com lists services by country\n\n"
    "If you are in immediate danger, please contact your local emergency services.\n\n"
    "You deserve real support from a person, not an algorithm. Anugamana will still "
    "be here afterwards."
)


def lexical_crisis_check(query: str) -> bool:
    """Fast, offline, dependency-free. True when escalation is unambiguous."""
    return bool(_CRISIS_RE.search(query))


async def classify(query: str) -> str:
    """
    Returns "crisis" or "safe".

    A lexical hit short-circuits and is never overridden. The model stage runs
    only on lexically-clean queries and fails open, so an API outage degrades to
    the prefilter rather than disabling safety entirely.
    """
    if lexical_crisis_check(query):
        logger.warning("crisis_detected", stage="lexical")
        return "crisis"

    try:
        response = await _client.messages.create(
            model=LLM_MODEL,
            max_tokens=5,
            system=CRISIS_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        verdict = response.content[0].text.strip().lower()
        if "crisis" in verdict:
            logger.warning("crisis_detected", stage="model")
            return "crisis"
        return "safe"
    except Exception as e:
        logger.warning("crisis_classifier_failed", error=str(e))
        return "safe"
