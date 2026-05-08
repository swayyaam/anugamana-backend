#!/usr/bin/env python3
"""
Bhagavad Gita scraper — vedabase.io

Improvements over v1:
  - Handles multi-verse range pages (e.g. /en/library/bg/2/41-43/)
  - Retry with exponential backoff via tenacity
  - Checkpoint/resume: saves after every page so a crash costs nothing
  - Sorted output (chapter, verse)
"""

import copy
import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ── Config ────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
BASE_URL = "https://vedabase.io"
BG_INDEX = f"{BASE_URL}/en/library/bg/"
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "gita_full.json"
CHECKPOINT_FILE = DATA_DIR / "scrape_checkpoint.json"
REQUEST_DELAY = 1.2  # seconds between requests — be polite

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── HTTP ──────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def fetch(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp


# ── Text helpers ──────────────────────────────────────────────────────────────

def clean_label(text: str, label: str) -> str:
    """Strip a leading section label like 'Translation:' from extracted text."""
    if not text:
        return ""
    return re.sub(rf"^\s*{label}[:\s]*", "", text, flags=re.IGNORECASE).strip()


def element_text(el) -> str:
    """Extract clean text from a BS4 element, preserving paragraph breaks."""
    if el is None:
        return ""
    el = copy.copy(el)
    for br in el.find_all("br"):
        br.replace_with("\n")
    paras = el.find_all("p")
    if paras:
        return "\n\n".join(p.get_text(" ", strip=True) for p in paras)
    return el.get_text(" ", strip=True)


def extract_sanskrit(verse_text_div) -> str:
    if verse_text_div is None:
        return ""
    el = copy.copy(verse_text_div)
    for br in el.find_all("br"):
        br.replace_with("\n")
    skip = {"Verse text", "Verse Text"}
    lines = [
        ln.strip()
        for ln in el.get_text("\n", strip=True).splitlines()
        if ln.strip() and ln.strip() not in skip
    ]
    return "\n".join(lines)


def extract_devanagari(container) -> str:
    """Extract Devanagari script text from a div.av-devanagari element."""
    if container is None:
        return ""
    el = copy.copy(container)
    # Remove the hidden section heading ("Devanagari")
    for h2 in el.find_all("h2"):
        h2.decompose()
    for br in el.find_all("br"):
        br.replace_with("\n")
    lines = [ln.strip() for ln in el.get_text("\n", strip=True).splitlines() if ln.strip()]
    return "\n".join(lines)


def extract_purport(soup) -> str:
    for sel in ["div.av-purport", "div.wrapper-purport", "div.purport", "div.r-r-p"]:
        text = element_text(soup.select_one(sel))
        if text and len(text) > 10:
            return clean_label(text, "Purport")
    # Fallback: locate "Purport" text marker
    marker = soup.find(string=lambda t: t and "Purport" in t.strip())
    if marker:
        parent = marker.find_parent("div")
        if parent:
            return clean_label(element_text(parent), "Purport")
    return ""


# ── Link / verse-range parsing ────────────────────────────────────────────────

_SINGLE_RE = re.compile(r"^/en/library/bg/(\d+)/(\d+)/$")
_RANGE_RE = re.compile(r"^/en/library/bg/(\d+)/(\d+)-(\d+)/$")


def parse_href(href: str) -> tuple[int, list[int]] | None:
    """
    Return (chapter, [verse_nums]) for single (/2/47/) and range (/2/41-43/) hrefs.
    """
    m = _SINGLE_RE.fullmatch(href)
    if m:
        return int(m.group(1)), [int(m.group(2))]
    m = _RANGE_RE.fullmatch(href)
    if m:
        ch, start, end = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return ch, list(range(start, end + 1))
    return None


def get_verse_links(chapter: int) -> list[tuple[str, int, list[int]]]:
    """Discover all verse page URLs for a chapter, including range pages."""
    resp = fetch(f"{BG_INDEX}{chapter}/")
    soup = BeautifulSoup(resp.text, "html.parser")

    seen_hrefs: set[str] = set()
    results: list[tuple[str, int, list[int]]] = []

    for a in soup.select("a[href]"):
        href = a["href"]
        if href in seen_hrefs:
            continue
        parsed = parse_href(href)
        if parsed and parsed[0] == chapter:
            seen_hrefs.add(href)
            results.append((BASE_URL + href, *parsed))

    return results


# ── Page scrapers ─────────────────────────────────────────────────────────────

def scrape_single(soup, chapter: int, verse: int) -> dict:
    return {
        "verse_id": f"{chapter}.{verse}",
        "chapter": chapter,
        "verse": verse,
        "devanagari": extract_devanagari(soup.select_one("div.av-devanagari")),
        "sanskrit": extract_sanskrit(soup.select_one("div.av-verse_text")),
        "synonyms": clean_label(
            element_text(soup.select_one("div.av-synonyms")), "Synonyms"
        ),
        "translation": clean_label(
            element_text(soup.select_one("div.av-translation")), "Translation"
        ),
        "purport": extract_purport(soup),
    }


def scrape_range(soup, chapter: int, verses: list[int]) -> list[dict]:
    """
    Range pages share synonyms and purport across all grouped verses.
    Each verse has its own Sanskrit block and translation section.
    """
    purport = extract_purport(soup)
    synonyms = clean_label(
        element_text(soup.select_one("div.av-synonyms")), "Synonyms"
    )

    devanagari_divs = soup.select("div.av-devanagari")
    verse_text_divs = soup.select("div.av-verse_text")
    translation_divs = soup.select("div.av-translation")

    records = []
    for i, v_num in enumerate(verses):
        devanagari = extract_devanagari(devanagari_divs[i] if i < len(devanagari_divs) else None)
        sanskrit = extract_sanskrit(verse_text_divs[i] if i < len(verse_text_divs) else None)
        translation = clean_label(
            element_text(translation_divs[i] if i < len(translation_divs) else None),
            "Translation",
        )
        records.append(
            {
                "verse_id": f"{chapter}.{v_num}",
                "chapter": chapter,
                "verse": v_num,
                "devanagari": devanagari,
                "sanskrit": sanskrit,
                "synonyms": synonyms,
                "translation": translation,
                "purport": purport,
            }
        )
    return records


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> tuple[set, list]:
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        return set(data["seen"]), data["verses"]
    return set(), []


def save_checkpoint(seen: set, verses: list) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps({"seen": list(seen), "verses": verses}, ensure_ascii=False)
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    seen, all_verses = load_checkpoint()
    if seen:
        log.info(f"Resuming from checkpoint — {len(all_verses)} verses already scraped")

    for chapter in range(1, 19):
        log.info(f"── Chapter {chapter}")

        try:
            links = get_verse_links(chapter)
        except Exception as exc:
            log.error(f"  Failed to load chapter index: {exc}")
            continue

        for url, ch, verse_nums in links:
            if all(f"{ch}.{v}" in seen for v in verse_nums):
                continue  # already done

            try:
                soup = BeautifulSoup(fetch(url).text, "html.parser")

                records = (
                    [scrape_single(soup, ch, verse_nums[0])]
                    if len(verse_nums) == 1
                    else scrape_range(soup, ch, verse_nums)
                )

                for r in records:
                    if r["verse_id"] not in seen:
                        all_verses.append(r)
                        seen.add(r["verse_id"])
                        flag = "" if r["purport"] else "  ⚠ no purport"
                        log.info(f"  ✓ {r['verse_id']}{flag}")

                save_checkpoint(seen, all_verses)
                time.sleep(REQUEST_DELAY)

            except Exception as exc:
                log.error(f"  Error scraping {url}: {exc}")

    all_verses.sort(key=lambda v: (v["chapter"], v["verse"]))

    OUTPUT_FILE.write_text(
        json.dumps(all_verses, ensure_ascii=False, indent=2)
    )

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    log.info(f"Done — {len(all_verses)} verses saved to {OUTPUT_FILE}")

    missing = [v["verse_id"] for v in all_verses if not v["purport"]]
    if missing:
        log.warning(f"Verses still missing purport: {missing}")


if __name__ == "__main__":
    main()
