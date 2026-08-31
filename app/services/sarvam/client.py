"""
Thin async client for the Sarvam AI REST API.

Design notes
------------
Sarvam's model identifiers and response field names version independently of
this codebase, and their published reference did not pin every field at the time
this was written. Two consequences, both deliberate:

* **Schema tolerance.** Responses are read through `pick()`, which accepts any of
  several plausible key names and raises a clear error naming what it actually
  received. A field rename produces a legible failure instead of a silent None
  that propagates into a research result.
* **Absent key is a supported state.** With no SARVAM_API_KEY the client reports
  `available == False` and every caller degrades to its non-Sarvam path. Indic
  support is an enhancement, never a hard dependency of the search pipeline.

`scripts/verify_sarvam.py` exercises every endpoint against the live API once and
reports the real shapes, so drift is caught deliberately rather than in
production.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from app.config import (
    SARVAM_API_KEY,
    SARVAM_AUTH_HEADER,
    SARVAM_BASE_URL,
    SARVAM_TIMEOUT_S,
)

logger = structlog.get_logger(__name__)

MAX_RETRIES = 3
RETRY_STATUS = {429, 500, 502, 503, 504}


class SarvamError(RuntimeError):
    """Any Sarvam call that failed in a way the caller should notice."""


class SarvamUnavailable(SarvamError):
    """No API key configured. Callers degrade rather than fail."""


def pick(payload: dict, *candidates: str, required: bool = True) -> Any:
    """
    Read the first present key from `candidates`.

    Tolerates the API returning `translated_text` where we expected `output`,
    and fails loudly — naming the keys actually returned — rather than silently
    yielding None.
    """
    for key in candidates:
        if key in payload and payload[key] is not None:
            return payload[key]
    if required:
        raise SarvamError(
            f"none of {candidates} present in Sarvam response; got keys "
            f"{sorted(payload)}"
        )
    return None


class SarvamClient:
    def __init__(
        self,
        api_key: str | None = SARVAM_API_KEY,
        base_url: str = SARVAM_BASE_URL,
        timeout: float = SARVAM_TIMEOUT_S,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={
                    SARVAM_AUTH_HEADER: self._api_key or "",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def post(self, path: str, payload: dict) -> dict:
        if not self.available:
            raise SarvamUnavailable("SARVAM_API_KEY is not set")

        client = await self._http()
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(path, json=payload)
                if response.status_code in RETRY_STATUS:
                    last_error = SarvamError(
                        f"{path} -> HTTP {response.status_code}: {response.text[:200]}"
                    )
                    await asyncio.sleep(2**attempt)
                    continue
                if response.status_code >= 400:
                    raise SarvamError(
                        f"{path} -> HTTP {response.status_code}: {response.text[:300]}"
                    )
                return response.json()
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_error = e
                await asyncio.sleep(2**attempt)

        raise SarvamError(f"{path} failed after {MAX_RETRIES} attempts: {last_error}")


#: Module-level default. Tests substitute their own instance.
client = SarvamClient()
