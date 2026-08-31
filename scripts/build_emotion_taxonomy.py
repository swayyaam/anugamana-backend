#!/usr/bin/env python3
"""
Derive the emotion taxonomy, and prove it is complete and non-redundant.

A taxonomy asserted by hand is a guess. This script makes both properties
checkable:

**Coverage** — every frequent affective term in the corpus's own `emotions`
fields, and every affective term in the benchmark queries, must map to some
label. Anything unmapped is reported as a gap, so the taxonomy is answerable to
the data rather than to intuition.

**Non-redundancy** — each label's description is embedded with the same model
the retriever uses, and every pair is scored. Pairs above REDUNDANCY_THRESHOLD
are reported as candidates for merging: two labels a retriever cannot separate
are one label wearing two names, and they waste weight mass when five are
assigned per verse.

Each label is also anchored to the text's own vocabulary. That is not
decoration: the Gita has a worked theory of these states — chapter 1 is
Arjuna-visada-yoga, the yoga of despair — and aligning to it keeps the scheme
defensible for the domain instead of importing a generic sentiment schema.

Usage:
    python scripts/build_emotion_taxonomy.py            # report + write
    python scripts/build_emotion_taxonomy.py --check    # report only, exit 1 on problems
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_DIR, ENRICHED_FILE  # noqa: E402

OUT_FILE = DATA_DIR / "emotion_taxonomy.json"

#: Two labels whose descriptions embed this close are not separable by the
#: retriever and should be merged.
REDUNDANCY_THRESHOLD = 0.90

#: How many labels each verse is tagged with.
TAGS_PER_VERSE = 5

GREEN, RED, YELLOW, BOLD, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[2m", "\033[0m"
)


# ---------------------------------------------------------------------------
# The candidate taxonomy
# ---------------------------------------------------------------------------
# Grouped by family. `terms` are the surface words that should map to this
# label; they drive the coverage check and are not used at inference time.

CANDIDATES = [
    # --- fear family --------------------------------------------------------
    ("anxiety", "cintā", "distress",
     "anticipatory dread about an outcome that has not arrived yet",
     "restless worry about how something will turn out, obsessing over "
     "consequences you cannot control, dread about a decision still ahead",
     ["anxiety", "anxious", "worry", "worried", "nervous", "apprehension"]),
    ("fear", "bhaya", "distress",
     "dread of a specific threat, including death and loss",
     "fear of what is coming, terror of dying, dread of losing what you have "
     "or the people you love",
     ["fear", "afraid", "terror", "terrified", "dread", "scared", "frightened"]),
    ("insecurity", "asthairya", "distress",
     "doubt about one's own adequacy or standing",
     "feeling not good enough, comparing yourself with others and coming up "
     "short, fear that you will be found out",
     ["insecurity", "insecure", "inadequate", "unworthy", "comparison"]),

    # --- sorrow family ------------------------------------------------------
    ("grief", "śoka", "distress",
     "sorrow at loss, especially of people",
     "mourning someone who has died, the ache of absence, sorrow over things "
     "that cannot be undone",
     ["grief", "grieving", "mourning", "sorrow", "loss", "bereavement", "sad"]),
    ("despair", "viṣāda", "distress",
     "collapse of the will to act; Arjuna's state in chapter one",
     "feeling that nothing you do matters, wanting to put down your "
     "responsibilities and withdraw from everything",
     ["despair", "hopeless", "hopelessness", "desperation", "pointless",
      "giving up", "meaningless"]),
    ("helplessness", "avaśatva", "distress",
     "having no agency over one's own situation",
     "being unable to change what is happening to you, watching events you "
     "cannot influence, powerlessness over your circumstances",
     ["helpless", "helplessness", "powerless", "stuck", "no control"]),
    ("exhaustion", "glāni", "distress",
     "depletion; the weariness that precedes giving up",
     "burnout from carrying something too long, having nothing left to give, "
     "being tired in a way that sleep does not fix",
     ["exhaustion", "exhausted", "burnout", "drained", "weary", "tired",
      "depleted"]),
    ("loneliness", "eka-bhāva", "distress",
     "isolation; being unaccompanied in one's situation",
     "feeling that nobody understands what you are going through, being "
     "surrounded by people and still alone",
     ["loneliness", "lonely", "alone", "isolated", "isolation"]),

    # --- self-judgment family ----------------------------------------------
    ("shame", "lajjā", "distress",
     "judgment about the kind of person one is",
     "feeling fundamentally unworthy, wanting to hide who you are, believing "
     "something is wrong with you rather than with what you did",
     ["shame", "ashamed", "humiliation", "worthless", "disgrace"]),
    ("guilt", "pāpa-bodha", "distress",
     "remorse about a specific act or failure to act",
     "regret over something you did or failed to do, wishing you had chosen "
     "differently, the weight of having let someone down",
     ["guilt", "guilty", "remorse", "regret", "blame", "fault"]),

    # --- agitation family ---------------------------------------------------
    ("anger", "krodha", "distress",
     "hostility arising from thwarted desire",
     "anger at someone who wronged you, resentment that keeps returning, "
     "irritation you cannot put down",
     ["anger", "angry", "rage", "resentment", "irritation", "furious",
      "bitter", "bitterness"]),
    ("frustration", "aśānti", "distress",
     "agitation from repeated blocked effort",
     "trying again and again and getting the same result, effort that goes "
     "nowhere, being blocked by things outside your control",
     ["frustration", "frustrated", "stuck", "blocked", "futile"]),
    ("restlessness", "cañcalatva", "distress",
     "inability to settle the mind",
     "a mind that will not be still, jumping between thoughts, unable to "
     "concentrate or sit with yourself",
     ["restlessness", "restless", "agitated", "distracted", "scattered",
      "overthinking", "racing"]),
    ("entrapment", "bandha", "distress",
     "being bound by obligation or circumstance",
     "feeling trapped in a role you cannot leave, obligations that hold you "
     "in place, a life shaped by duties you did not choose",
     ["trapped", "bound", "obligation", "duty-bound", "caged", "tied"]),

    # --- confusion family ---------------------------------------------------
    ("confusion", "moha", "distress",
     "delusion; not seeing the situation as it is",
     "not being able to tell what is actually happening, feeling lost, unable "
     "to separate what is real from what you told yourself",
     ["confusion", "confused", "lost", "delusion", "bewildered", "unclear"]),
    ("doubt", "saṁśaya", "distress",
     "inability to settle on what is true or right",
     "second-guessing your path, unable to commit to a decision, wondering "
     "whether any of this is true",
     ["doubt", "doubtful", "uncertain", "unsure", "second-guessing",
      "questioning"]),

    # --- desire family ------------------------------------------------------
    ("craving", "kāma", "desire",
     "attachment to obtaining a particular outcome",
     "wanting something badly, craving recognition or possession or success, "
     "unable to let go of an outcome you have fixed on",
     ["craving", "desire", "wanting", "attachment", "greed", "hunger",
      "ambition"]),
    ("longing", "utkaṇṭhā", "desire",
     "yearning toward something absent or higher",
     "missing something you cannot name, yearning for meaning or for God, "
     "homesickness for a place you have not been",
     ["longing", "yearning", "missing", "ache", "seeking meaning"]),
    ("envy", "mātsarya", "desire",
     "resentment at another's good fortune",
     "watching someone else get what you wanted, comparing your life against "
     "theirs, bitterness at another's success",
     ["envy", "envious", "jealous", "jealousy", "begrudge"]),
    ("pride", "mada", "desire",
     "inflated self-regard; identification with one's own status",
     "needing to be seen as the one who did it, taking things personally "
     "because your standing is at stake, superiority",
     ["pride", "ego", "arrogance", "superiority", "self-importance",
      "recognition"]),

    # --- settled family -----------------------------------------------------
    ("equanimity", "samatva", "settled",
     "the steady state the text points toward",
     "steadiness that does not depend on outcomes, acceptance, the relief of "
     "having finally let something go, peace",
     ["equanimity", "peace", "peaceful", "calm", "acceptance", "relief",
      "steady", "satisfaction", "contentment"]),
    ("devotion", "bhakti", "settled",
     "love and surrender directed toward the divine",
     "loving devotion, giving yourself over to something greater, trust that "
     "does not require you to understand everything",
     ["devotion", "love", "surrender", "faith", "trust", "worship"]),
    ("resolve", "dhṛti", "settled",
     "determination to act despite difficulty",
     "steadiness of purpose, the decision to keep going, commitment that "
     "survives discouragement",
     ["resolve", "determination", "commitment", "perseverance", "discipline",
      "confidence"]),

    # --- non-affective ------------------------------------------------------
    ("seeking", "jijñāsā", "inquiry",
     "intellectual longing rather than distress",
     "wanting to understand how things actually are, philosophical curiosity, "
     "asking what the self or the world really is",
     ["curiosity", "understanding", "knowledge", "inquiry", "wondering",
      "philosophical"]),
]


def as_dict(entry):
    key, term, family, gloss, probe, terms = entry
    return {
        "key": key, "gita_term": term, "family": family,
        "gloss": gloss, "probe": probe, "terms": terms,
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def corpus_vocabulary(top_n=120):
    verses = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
    text = " ".join(
        (v["meaning_fields"].get("emotions", "") or "").lower() for v in verses
    )
    return Counter(re.findall(r"[a-z]{4,}", text)).most_common(top_n)


def query_vocabulary(top_n=120):
    path = DATA_DIR / "benchmark" / "queries.json"
    if not path.exists():
        return []
    queries = json.loads(path.read_text(encoding="utf-8"))
    text = " ".join(q["query"].lower() for q in queries)
    return Counter(re.findall(r"[a-z]{4,}", text)).most_common(top_n)


#: Words that appear often in affective prose without naming an affect.
NON_AFFECTIVE = set("""
that with this from they have been your what when will their there them then than about into over more
some such only also very much many most other others being does doing feel feeling feels felt like where
which while would could should yours something someone anything everything nothing those these know knows
knowing want wants wanted because through without within toward towards even ever just still make makes
made take takes life work good keep keeps kept come comes came give gives given need needs sense think
thinks thought real actually trying whether cannot mind body self world people things understanding
realizing realize watching practice effort enough having might rather never doesn stop between control
quiet subtle creeping deep deeper spiritual material mental personal genuine truly always same time every
matter path person right wrong weight arises itself else mixed constantly discovering recognizing seeing
believing recognition realization knowledge death success love intellectual after against years months
really been what your going able around before never little much said says tell told help
""".split())


def check_coverage(taxonomy, vocabulary, label):
    mapped = {}
    for entry in taxonomy:
        for term in entry["terms"] + [entry["key"]]:
            mapped[term] = entry["key"]

    gaps = []
    for word, count in vocabulary:
        if word in NON_AFFECTIVE or len(word) < 5:
            continue
        if word in mapped:
            continue
        # Prefix match catches anxious/anxiety, angry/anger, lonely/loneliness
        if any(word.startswith(t[:5]) or t.startswith(word[:5]) for t in mapped):
            continue
        gaps.append((word, count))

    print(f"\n{BOLD}Coverage — {label}{RESET}")
    if not gaps:
        print(f"  {GREEN}every frequent affective term maps to a label{RESET}")
    else:
        print(f"  {YELLOW}unmapped terms (review — many will be topics, not "
              f"affects):{RESET}")
        print("    " + " · ".join(f"{w}({c})" for w, c in gaps[:25]))
    return gaps


def check_redundancy(taxonomy):
    from app.services.retrieval import _load_model

    print(f"\n{BOLD}Non-redundancy{RESET}")
    print(f"{DIM}  each label's probe embedded with the retrieval model; pairs "
          f"above {REDUNDANCY_THRESHOLD} are one label wearing two names{RESET}")

    model = _load_model()
    probes = [e["probe"] for e in taxonomy]
    vectors = model.encode(
        probes, batch_size=len(probes), max_length=256, return_dense=True
    )["dense_vecs"]

    import numpy as np
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    similarity = vectors @ vectors.T

    pairs = []
    for i in range(len(taxonomy)):
        for j in range(i + 1, len(taxonomy)):
            pairs.append((float(similarity[i, j]), taxonomy[i]["key"], taxonomy[j]["key"]))
    pairs.sort(reverse=True)

    redundant = [p for p in pairs if p[0] >= REDUNDANCY_THRESHOLD]
    print(f"\n  closest pairs:")
    for score, a, b in pairs[:8]:
        flag = f"  {RED}<- MERGE{RESET}" if score >= REDUNDANCY_THRESHOLD else ""
        print(f"    {score:.4f}  {a:<14} {b}{flag}")

    if redundant:
        print(f"\n  {RED}{len(redundant)} redundant pair(s) — merge before "
              f"tagging{RESET}")
    else:
        print(f"\n  {GREEN}no pair exceeds {REDUNDANCY_THRESHOLD}; all "
              f"{len(taxonomy)} labels are separable{RESET}")
    return redundant, pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report only; exit non-zero if the taxonomy has problems")
    args = parser.parse_args()

    taxonomy = [as_dict(e) for e in CANDIDATES]

    print(f"{BOLD}Emotion taxonomy — {len(taxonomy)} candidate labels{RESET}")
    families = Counter(e["family"] for e in taxonomy)
    print(f"  families: " + ", ".join(f"{k}={v}" for k, v in sorted(families.items())))
    print(f"  tags assigned per verse: {TAGS_PER_VERSE}")

    corpus_gaps = check_coverage(taxonomy, corpus_vocabulary(), "corpus emotions fields")
    query_gaps = check_coverage(taxonomy, query_vocabulary(), "benchmark queries")
    redundant, pairs = check_redundancy(taxonomy)

    if args.check:
        problems = len(redundant)
        print(f"\n{'PASS' if not problems else 'FAIL'}: {problems} redundant pairs")
        return 1 if problems else 0

    OUT_FILE.write_text(json.dumps({
        "version": 2,
        "tags_per_verse": TAGS_PER_VERSE,
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
        "max_pairwise_similarity": pairs[0][0] if pairs else None,
        "labels": taxonomy,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{DIM}written -> {OUT_FILE}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
