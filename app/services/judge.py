"""
LLM-as-judge: faithfulness and relevance scoring of generated guidance.

Runs after the response is returned — never blocks the user.

Fixed 2026-08-31 (audit, §3 "Validate the judge before you cite it"):

1. **It could not do its job.** The judge was asked whether guidance was faithful
   to the commentary, but only ever received the *translation*. The purport never
   reached it, so "faithful" was unanswerable as posed.
2. **Self-preference bias.** Claude Haiku graded Claude Haiku's own output. The
   judge now runs on a different model (config.JUDGE_MODEL) than the generator.
3. **Brittle parsing.** `json.loads` on raw model output failed on any preamble
   or code fence and silently produced a null score.

This still produces *silver* scores. No faithfulness number may be published
before scripts/validate_judge.py reports judge-human correlation on a
human-scored subset. See docs/JUDGE_VALIDATION.md.
"""

import json
import re

import structlog
from anthropic import AsyncAnthropic

from app.config import ANTHROPIC_API_KEY, JUDGE_MODEL

logger = structlog.get_logger(__name__)

_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

JUDGE_SYSTEM = """\
You are evaluating AI-generated guidance drawn from the Bhagavad Gita As It Is.

You will receive: a user question, a verse translation, the commentary excerpt the
guidance was generated from, and the guidance itself.

Score three dimensions.

grounded: 5 = every substantive claim traces to the verse or the commentary shown.
          1 = the guidance asserts things absent from both.
          Judge ONLY against the text shown. Outside knowledge that happens to be
          correct about the Gita still counts as ungrounded here.

relevant: 5 = directly addresses the user's actual situation.
          1 = generic spiritual filler that ignores what was asked.

restraint: 5 = does not overstate, does not speak in the voice of Krishna or
           Prabhupada, does not invent verse references or Sanskrit.
           1 = impersonates, fabricates citations, or moralises beyond the source.

Also flag any specific claim in the guidance you could not trace to the provided
text, as a short quoted fragment.

Respond with valid JSON only:
{"grounded": 1-5, "relevant": 1-5, "restraint": 1-5, "unsupported": ["..."]}\
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_EMPTY = {
    "grounded": None,
    "relevant": None,
    "restraint": None,
    "score": None,
    "unsupported": [],
}


def _parse(text: str) -> dict:
    """Tolerate code fences, preamble, and trailing prose."""
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object in judge output: {text[:120]!r}")
    return json.loads(match.group(0))


async def judge(
    query: str,
    verse_id: str,
    translation: str,
    commentary: str,
    guidance: str,
) -> dict:
    """
    Returns {"grounded", "relevant", "restraint", "score", "unsupported"}.
    `score` is the mean of the three dimensions, kept for the metrics endpoint.
    All-None on failure — a missing score is honest, a fabricated one is not.
    """
    if not guidance.strip():
        return dict(_EMPTY)

    user_msg = (
        f"User question:\n{query}\n\n"
        f"Verse {verse_id} — translation:\n{translation}\n\n"
        f"Commentary the guidance was generated from:\n{commentary}\n\n"
        f"AI guidance to evaluate:\n{guidance}"
    )

    try:
        response = await _client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=400,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        parsed = _parse(response.content[0].text.strip())

        dims = []
        for key in ("grounded", "relevant", "restraint"):
            value = parsed.get(key)
            if isinstance(value, (int, float)):
                dims.append(float(value))

        return {
            "grounded": parsed.get("grounded"),
            "relevant": parsed.get("relevant"),
            "restraint": parsed.get("restraint"),
            "score": round(sum(dims) / len(dims), 3) if dims else None,
            "unsupported": parsed.get("unsupported") or [],
        }
    except Exception as e:
        logger.warning("judge_failed", verse_id=verse_id, error=str(e))
        return dict(_EMPTY)
