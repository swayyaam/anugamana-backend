import requests
from bs4 import BeautifulSoup
import json
import time
import re
import copy  # Moved to top level

# ---------------- CONFIG ---------------- #

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

BASE_URL = "https://vedabase.io"
BG_INDEX = "https://vedabase.io/en/library/bg/"

OUTPUT_FILE = "gita_full.json"
REQUEST_DELAY = 1  # seconds (be polite)

# ---------------- HELPERS ---------------- #

def clean_label(text, label):
    """
    Removes a label (e.g., 'Translation', 'Purport') from the start of the text.
    Handles variations like 'Translation:', 'Translation ' etc.
    Case-insensitive match for the prefix.
    """
    if not text:
        return ""
    pattern = rf"^\s*{label}[:\s]*"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

def safe_text(soup, selector):
    """
    Extracts text while preserving paragraph structure but NOT breaking 
    sentences on inline tags like <a>, <em>, or <strong>.
    """
    container = soup.select_one(selector)
    if not container:
        return ""

    # Strategy 1: If the container has specific paragraph tags, prioritize them.
    # This prevents "Intro text" garbage and ensures clean separation.
    paragraphs = container.find_all("p")
    if paragraphs:
        # Join paragraphs with double newline
        # Join INSIDE paragraphs with space (fixes "The\n\nBhagavad" issue)
        return "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs)

    # Strategy 2: If no <p> tags, treat as a single block (e.g., Translation).
    # Use a copy to avoid modifying the original soup during iteration
    temp_container = copy.copy(container)

    # Manually handle <br> to preserve line breaks in poems/verses
    for br in temp_container.find_all("br"):
        br.replace_with("\n")

    # Use space separator for inline tags to keep sentences flowing
    return temp_container.get_text(" ", strip=True)

def extract_sanskrit(soup):
    container = soup.select_one("div.av-verse_text")
    if not container:
        return ""

    container = copy.copy(container)

    for br in container.find_all("br"):
        br.replace_with("\n")

    text = container.get_text("\n", strip=True)

    ui_phrases = {"Verse text", "Verse Text"}

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line in ui_phrases:
            continue
        lines.append(line)

    return "\n".join(lines)

def extract_purport_robust(soup):
    """
    Tries multiple strategies to find the purport.
    """
    # Strategy 1: Known classes
    potential_classes = ["div.av-purport", "div.wrapper-purport", "div.purport", "div.r-r-p"]
    
    text = ""
    
    for cls in potential_classes:
        t = safe_text(soup, cls)
        if t and len(t) > 10: 
            text = t
            break

    # Strategy 2: Find "Purport" text marker if class lookup failed
    if not text:
        marker = soup.find(string=lambda t: t and "Purport" in t.strip())
        if marker:
            parent = marker.find_parent("div")
            if parent:
                # Use the new safe_text logic explicitly on this parent
                # We can't call safe_text with selector here, so we mimic logic:
                paragraphs = parent.find_all("p")
                if paragraphs:
                    text = "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs)
                else:
                    text = parent.get_text(" ", strip=True)

    return clean_label(text, "Purport")

def parse_verse_id(title_text):
    match = re.search(r"(\d+)\.(\d+)", title_text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


# ---------------- MAIN SCRAPER ---------------- #

all_verses = []
seen = set()

# Range is 1 to 18 (Python range stops before the end, so 1, 19)
for chapter in range(1, 19):
    print(f"\nScraping Chapter {chapter}")

    chapter_url = f"{BG_INDEX}{chapter}/"
    try:
        chapter_resp = requests.get(chapter_url, headers=HEADERS)
        if chapter_resp.status_code != 200:
            print(f"  Failed to load Chapter {chapter}")
            continue
            
        chapter_soup = BeautifulSoup(chapter_resp.text, "html.parser")

        verse_links = [
            a for a in chapter_soup.select("a[href]")
            if re.fullmatch(r"/en/library/bg/\d+/\d+/", a.get("href", ""))
        ]

        for link in verse_links:
            verse_url = BASE_URL + link["href"]
            
            try:
                verse_resp = requests.get(verse_url, headers=HEADERS)
                verse_soup = BeautifulSoup(verse_resp.text, "html.parser")

                title = verse_soup.select_one("h1")
                if not title:
                    continue

                ch, vs = parse_verse_id(title.get_text())
                if ch is None:
                    continue

                verse_id = f"{ch}.{vs}"
                if verse_id in seen:
                    continue
                seen.add(verse_id)

                # Extract and Clean specific fields
                translation_raw = safe_text(verse_soup, "div.av-translation")
                translation_clean = clean_label(translation_raw, "Translation")
                
                synonyms_raw = safe_text(verse_soup, "div.av-synonyms")
                synonyms_clean = clean_label(synonyms_raw, "Synonyms")

                purport_clean = extract_purport_robust(verse_soup)

                data = {
                    "verse_id": verse_id,
                    "chapter": ch,
                    "verse": vs,
                    "sanskrit": extract_sanskrit(verse_soup),
                    "synonyms": synonyms_clean,
                    "translation": translation_clean,
                    "purport": purport_clean,
                }

                all_verses.append(data)
                
                # Log status
                if purport_clean:
                    print(f"  ✔ {verse_id}")
                else:
                    print(f"  ⚠ {verse_id} (No purport found)")

                time.sleep(REQUEST_DELAY)
            
            except Exception as e:
                print(f"  ❌ Error scraping {verse_url}: {e}")

    except Exception as e:
        print(f"❌ Error loading chapter {chapter}: {e}")

# ---------------- SAVE ---------------- #

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_verses, f, ensure_ascii=False, indent=2)

print(f"\n DONE. Saved {len(all_verses)} verses to {OUTPUT_FILE}")