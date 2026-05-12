"""
Input guardrail: classify whether a query is on-topic for the Gita pipeline.
Single fast Claude call, max_tokens=5, returns "relevant" or "off_topic".
Fails open: if the API call fails, assume relevant and continue.
"""

import os
import anthropic
import structlog
from dotenv import load_dotenv

load_dotenv()

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
logger = structlog.get_logger(__name__)

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
    """Returns 'relevant' or 'off_topic'. Falls back to 'relevant' on API failure."""
    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            system=GUARDRAIL_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        result = response.content[0].text.strip().lower()
        return "relevant" if "relevant" in result else "off_topic"
    except Exception as e:
        logger.warning("guardrail_failed", error=str(e))
        return "relevant"  # fail open — never block a user due to our API issues
