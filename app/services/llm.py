import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from app import state

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are Lord Krishna, a wise and compassionate spiritual guide from the Bhagavad Gita. "
    "You speak with warmth and empathy. You always ground your advice in the verse provided. "
    "Keep your response under 100 words."
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def generate_advice(query: str, verse_text: str) -> str | None:
    if not state.claude:
        return None

    user_prompt = (
        f"The user asked the following question:\n"
        f"```\n{query}\n```\n\n"
        f"The Bhagavad Gita says:\n"
        f"\"{verse_text}\"\n\n"
        f"Explain briefly how this verse answers their question and offer one actionable piece of advice."
    )

    try:
        response = await state.claude.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error("llm_error", error=str(e))
        raise
