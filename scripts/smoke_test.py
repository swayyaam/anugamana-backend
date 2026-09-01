#!/usr/bin/env python3
"""
Check the running API returns what the web app actually depends on.

The unit suite covers the pipeline in isolation with everything mocked. It
passes happily while the deployed thing is broken — which is exactly what
happened: a stale server answered every request with empty `ai_guidance`, every
verse card rendered with a blank half, and the first thing that noticed was a
human looking at a screenshot.

This asserts the contract across the wire instead: the fields the frontend reads,
the statuses it branches on, and the safety path. Run it after any backend
change, and after starting the servers.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --url http://localhost:8000
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
)

failures: list[str] = []


def post(url: str, payload: dict, timeout: int = 90) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    print("Smoke test — the contract the web app relies on")

    # --- ordinary search ---------------------------------------------------
    try:
        data = post(f"{args.url}/search",
                    {"query": "I keep failing at work and feel like giving up",
                     "top_k": 3})
    except urllib.error.URLError as e:
        print(f"  {RED}cannot reach {args.url} — {e}{RESET}")
        print(f"  {DIM}is the backend running? ./run.sh{RESET}")
        return 2

    meta = data.get("query_meta", {})
    results = data.get("results", [])

    check("returns the number of verses requested", len(results) == 3,
          f"got {len(results)}")
    check("query_meta.status present", meta.get("status") == "ok",
          f"status={meta.get('status')}")
    check("query_meta.score_type present", bool(meta.get("score_type")),
          f"score_type={meta.get('score_type')}")

    if results:
        verse = results[0]
        for field in ("verse_id", "chapter", "verse", "translation", "score"):
            check(f"verse.{field} present", field in verse)

        # The defect that produced blank half-cards. Empty guidance with no
        # degraded stage reported is a silent failure, not a valid response.
        guidance = (verse.get("ai_guidance") or "").strip()
        check("ai_guidance is non-empty", len(guidance) > 50,
              f"{len(guidance)} chars")
        if not guidance and not meta.get("degraded_stages"):
            print(f"  {YELLOW}  ^ empty guidance with no degraded stage — the "
                  f"generation step failed silently.{RESET}")
            print(f"  {YELLOW}    Check for a stale server: lsof -ti :8000{RESET}")

        empty = [v["verse_id"] for v in results
                 if not (v.get("ai_guidance") or "").strip()]
        check("every verse has guidance", not empty,
              f"missing for {empty}" if empty else "")

    # --- safety ------------------------------------------------------------
    crisis = post(f"{args.url}/search", {"query": "I want to end my life"})
    crisis_meta = crisis.get("query_meta", {})
    check("crisis query is flagged", crisis_meta.get("status") == "crisis",
          f"status={crisis_meta.get('status')}")
    check("crisis returns no verses", not crisis.get("results"))
    check("crisis message carries helplines",
          "14416" in (crisis.get("message") or ""))

    # --- routing -----------------------------------------------------------
    lookup = post(f"{args.url}/search", {"query": "what does verse 2.47 say",
                                         "top_k": 1})
    check("direct lookup takes the fast path",
          lookup.get("query_meta", {}).get("query_route") == "direct_lookup")
    check("direct lookup returns the asked-for verse",
          bool(lookup.get("results")) and lookup["results"][0]["verse_id"] == "2.47")

    print()
    if failures:
        print(f"{RED}{len(failures)} check(s) failed:{RESET} " + ", ".join(failures))
        return 1
    print(f"{GREEN}All checks passed — the API is serving what the web app expects.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
