"""
LLM-as-judge: async faithfulness check on generated guidance.

Runs after the response is returned — never blocks the user.
Scores: {"faithful": bool, "relevant": bool, "score": 1-5}
"""

import json
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

JUDGE_SYSTEM = """\
You are evaluating AI-generated spiritual guidance for the Bhagavad Gita.

Given: a user question, a verse, and the AI's guidance response.
Score on two dimensions:

faithful: true if the guidance stays within what the verse and commentary say.
          false if it adds outside information or misrepresents the teaching.

relevant: true if the guidance actually addresses the user's question.
          false if it is generic or misses the point of the question.

score: 1-5 overall quality (1 = poor, 5 = excellent)

Respond with valid JSON only, no explanation:
{"faithful": true/false, "relevant": true/false, "score": 1-5}\
"""


def judge(query: str, verse_id: str, translation: str, guidance: str) -> dict:
    """Returns {"faithful": bool, "relevant": bool, "score": int}."""
    user_msg = (
        f"User question: {query}\n\n"
        f"Verse {verse_id}: {translation}\n\n"
        f"AI guidance: {guidance}"
    )

    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return json.loads(response.content[0].text.strip())
    except Exception:
        return {"faithful": None, "relevant": None, "score": None}
