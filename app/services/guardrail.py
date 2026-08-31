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


async def classify(query: str) -> str:
    """Returns 'relevant' or 'off_topic'. Falls back to 'relevant' on failure."""
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
        return "relevant"  # fail open — never block a user over our own outage
