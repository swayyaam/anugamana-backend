#!/usr/bin/env python3
"""
One-shot verification of the Sarvam API contract.

Run this once when a SARVAM_API_KEY is first configured, and again whenever
Sarvam announce a model or API change. It calls every endpoint the pipeline
uses, prints the *actual* response keys, and reports whether the field names
app/services/sarvam/ reads are present.

This exists because Sarvam's model identifiers and response schemas version
independently of this codebase. A rename would otherwise surface as a silently
degraded pipeline — translation quietly returning the input unchanged — which is
exactly the failure mode that corrupts a research result without failing a test.

Usage:
    export SARVAM_API_KEY=...
    python scripts/verify_sarvam.py
    python scripts/verify_sarvam.py --skip-tts     # TTS returns large payloads
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import (  # noqa: E402
    SARVAM_CHAT_MODEL,
    SARVAM_ENABLED,
    SARVAM_TRANSLATE_MODEL,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_SPEAKER,
)
from app.services.sarvam.client import SarvamError, client  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
)

HINDI = "मैं अपने काम में बार-बार असफल हो रहा हूँ और हार मानने का मन करता है"


def report(name: str, ok: bool, detail: str = "") -> bool:
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"         {DIM}{line}{RESET}")
    return ok


def keys_of(payload) -> str:
    if isinstance(payload, dict):
        return "keys: " + ", ".join(sorted(payload))
    return f"type: {type(payload).__name__}"


async def check(name: str, path: str, payload: dict, expected: tuple[str, ...]):
    print(f"\n{YELLOW}{path}{RESET}  ({name})")
    try:
        response = await client.post(path, payload)
    except SarvamError as e:
        return report(name, False, str(e))

    detail = keys_of(response)
    found = [k for k in expected if isinstance(response, dict) and k in response]
    if found:
        sample = json.dumps(response.get(found[0]), ensure_ascii=False)[:160]
        return report(name, True, f"{detail}\nread via '{found[0]}': {sample}")
    return report(
        name,
        False,
        f"{detail}\nNONE of the field names this codebase reads were present: "
        f"{expected}\n-> update the pick() candidates in app/services/sarvam/",
    )


async def main(skip_tts: bool) -> int:
    if not SARVAM_ENABLED:
        print(f"{RED}SARVAM_API_KEY is not set.{RESET}")
        print("Add it to .env, then re-run. The pipeline works without it — every "
              "Sarvam call degrades to a defined fallback — but Indic support and "
              "conditions L1-L3 of the evaluation grid are unavailable.")
        return 2

    print("Verifying the Sarvam API contract against the live service.")
    print(f"{DIM}translate={SARVAM_TRANSLATE_MODEL}  tts={SARVAM_TTS_MODEL}  "
          f"chat={SARVAM_CHAT_MODEL}{RESET}")

    results = []

    results.append(await check(
        "language identification", "/text-lid",
        {"input": HINDI},
        ("language_code", "lang_code", "language"),
    ))

    results.append(await check(
        "translation (Mayura)", "/translate",
        {
            "input": HINDI,
            "source_language_code": "hi-IN",
            "target_language_code": "en-IN",
            "model": SARVAM_TRANSLATE_MODEL,
            "mode": "formal",
        },
        ("translated_text", "output", "text"),
    ))

    results.append(await check(
        "transliteration", "/transliterate",
        {
            "input": "karmanye vadhikaraste ma phaleshu kadachana",
            "source_language_code": "en-IN",
            "target_language_code": "hi-IN",
        },
        ("transliterated_text", "output", "text"),
    ))

    results.append(await check(
        "chat completions (emotion backend)", "/chat/completions",
        {
            "model": SARVAM_CHAT_MODEL,
            "messages": [{"role": "user", "content": "Reply with the word: ok"}],
            "max_tokens": 10,
            "temperature": 0.0,
        },
        ("choices",),
    ))

    if not skip_tts:
        results.append(await check(
            "text-to-speech (Bulbul)", "/text-to-speech",
            {
                "inputs": ["dharma"],
                "target_language_code": "hi-IN",
                "speaker": SARVAM_TTS_SPEAKER,
                "model": SARVAM_TTS_MODEL,
            },
            ("audios", "audio", "output"),
        ))

    await client.aclose()

    passed, total = sum(results), len(results)
    print(f"\n{'=' * 60}")
    if passed == total:
        print(f"{GREEN}All {total} endpoints match the expected contract.{RESET}")
        return 0
    print(f"{RED}{total - passed} of {total} endpoints did not match.{RESET}")
    print("Update the pick() candidate lists in app/services/sarvam/ to the key "
          "names printed above, then re-run.")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tts", action="store_true",
                        help="skip text-to-speech (returns large base64 payloads)")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.skip_tts)))
