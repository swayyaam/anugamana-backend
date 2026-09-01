"""
Topical guardrail: is this query in scope for the Gita pipeline?

Runs *after* the crisis check in app/services/safety.py, never instead of it —
this classifier is deliberately generous about personal distress, which is
exactly why it must not be the only thing standing between a user in crisis and
the retrieval pipeline.

Single Claude call, max_tokens=5. Fails open: an API outage must never block a
legitimate user.

Fixed 2026-08-31 (audit E-04): this coroutine used to call the *synchronous*
Anthropic client and was awaited directly in the request handler, so one
in-flight classification stalled every other request in the process.
"""

import re

import structlog
from anthropic import AsyncAnthropic

from app.config import ANTHROPIC_API_KEY, LLM_MODEL

logger = structlog.get_logger(__name__)

_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

GUARDRAIL_SYSTEM = """\
You are a classifier for a Bhagavad Gita spiritual guidance system.

Decide if the user's query is related to any of:
spiritual guidance, life situations, dharma, karma, duty, the soul, emotions,
philosophy, meditation, devotion, attachment, liberation, the Bhagavad Gita,
Hindu philosophy, yoga, Sanskrit terms, or any theme addressed in the Gita.

Be generous — personal struggles, existential questions, and moral dilemmas all count.
Only reject queries that are clearly unrelated: coding help, weather, sports scores, etc.

Reply with exactly one word: relevant   or   off_topic\
"""


#: Terms that essentially never occur in a genuine request for spiritual
#: guidance. Deliberately narrow and unambiguous: this list runs BEFORE the
#: model and its verdict is final, so a false positive turns a real user away.
#: Nothing here is a word someone would use to describe their life — no
#: "work", "code" (as in moral code), "practice", "energy", "balance".
_OFF_TOPIC_PATTERNS = [
    r"\bcss\b", r"\bhtml\b", r"\bjavascript\b", r"\btypescript\b",
    r"\bpython\b(?!\s+(?:of|as)\b)", r"\bjava\b", r"\bsql\b", r"\bregex\b",
    r"\bdiv\b", r"\bapi\b", r"\bendpoint\b", r"\bcompiler?\b",
    r"\bnpm\b", r"\bdocker\b", r"\bkubernetes\b", r"\bgit(hub)?\b",
    r"\bstack\s?overflow\b", r"\bsyntax\s+error\b", r"\bnull\s?pointer\b",
    r"\bweather\s+(?:today|tomorrow|forecast)\b", r"\bcricket\s+score\b",
    r"\bfootball\s+score\b", r"\bstock\s+price\b", r"\bbitcoin\b",
    r"\brecipe\s+for\b", r"\bflight\s+to\b", r"\bhotel\s+in\b",
]
_OFF_TOPIC_RE = re.compile("|".join(_OFF_TOPIC_PATTERNS), re.IGNORECASE)


def lexical_off_topic_check(query: str) -> bool:
    """
    Offline, unambiguous off-topic detection.

    The model guardrail fails OPEN — an API outage must never block a real user.
    The consequence is that while the API is down, "how do I center a div in
    CSS" is answered with verses, which is what happened in practice. This
    prefilter catches the blatant cases without a network call, so the failure
    mode degrades to "obvious rubbish is still rejected" rather than
    "everything is accepted".
    """
    return bool(_OFF_TOPIC_RE.search(query))


async def classify(query: str) -> str:
    """
    Returns 'relevant' or 'off_topic'.

    Fails open on API error, except where the offline prefilter has already
    settled it.
    """
    if lexical_off_topic_check(query):
        logger.info("off_topic_detected", stage="lexical")
        return "off_topic"

    try:
        response = await _client.messages.create(
            model=LLM_MODEL,
            max_tokens=5,
            system=GUARDRAIL_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        result = response.content[0].text.strip().lower()
        return "relevant" if "relevant" in result else "off_topic"
    except Exception as e:
        logger.warning("guardrail_failed", error=str(e))
        # Fail open: an outage of ours must not turn away someone with a real
        # question. The lexical prefilter above already handled the blatant
        # cases, so this is only the genuinely ambiguous middle.
        return "relevant"
