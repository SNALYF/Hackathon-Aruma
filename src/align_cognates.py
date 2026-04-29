"""
align_cognates.py — Stage 2: Cognate identification across daughter languages.

Reads the processed outputs of Stage 1 and produces two new CSV files:

    cognate_candidates.csv   every (lang1_form, langN_token) pair that clears
                             the similarity threshold, with scores and provenance
    cognate_sets.csv         the single best candidate per (english, language),
                             laid out as a wide table for Stage 3 consumption

Two complementary methods are used:

    parallel   (primary)  — sentences that share an identical English translation
                            across languages are grouped.  Within each group the
                            lang1 interlinear gloss is parsed to obtain a
                            per-token meaning; tokens from lang2-7 are then
                            aligned to lang1 tokens by LCS similarity.

    search     (fallback) — for every dictionary entry whose meaning is not
                            covered by the parallel method in a given language,
                            sentences whose translation contains the English
                            keyword are searched and the most similar token is
                            kept as a candidate.

Run directly:
    python src/align_cognates.py
    python src/align_cognates.py --processed-dir path/to/processed
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = _ROOT / "data" / "processed"
LANGUAGES = [f"lang{i}" for i in range(1, 8)]

# Minimum LCS similarity (diacritics stripped) to accept a candidate.
# Lang2-7 tokens are inflected surface forms; the shared root plus affixes
# typically pushes the similarity of genuine cognates above 0.5.
PARALLEL_THRESHOLD = 0.50
SEARCH_THRESHOLD = 0.45

# Sentences longer than this are excluded from the search fallback.
# Very long sentences contain many unrelated tokens, increasing false positives.
MAX_SEARCH_SENTENCE_TOKENS = 10

# Grammatical morpheme glosses that should not be treated as content meanings.
_GRAM_TAGS = frozenset(
    "ERG ABS NOM ACC DAT GEN ELAT ILL INESS DEF NEAR FAR INAN ANIM "
    "PL SG WIT EVID FUT INFER DIR NEG POSS SUBJ OBJ INCL EXCL BEN "
    "INSTR CONJ MIR ADJ ADV 1SG 2SG 3SG 1PL 2PL 3PL CL".split()
)

_STOPWORDS = frozenset(
    "a an the of in at to by for on is be or and not no ie ".split()
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CognateCandidate:
    """One candidate (lang1_form, langN_token) pair for a dictionary meaning."""

    english: str
    pos: str
    lang1_form: str
    language: str           # "lang2" through "lang7"
    candidate_form: str     # surface token from the target language
    similarity: float       # LCS similarity on diacritic-stripped strings
    method: str             # "parallel" or "search"
    sentence_id: int        # sentence_id in the target language's corpus
    sentence_length: int    # number of tokens in source sentence


@dataclass
class CognateSet:
    """Best cognate form per language for one dictionary entry."""

    english: str
    pos: str
    lang1: str
    lang2: str
    lang3: str
    lang4: str
    lang5: str
    lang6: str
    lang7: str


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def strip_diacritics(s: str) -> str:
    """Remove combining diacritical marks; keep all base characters unchanged.

    IPA-derived consonants (þ U+00FE, ħ U+0127, ð U+00F0, etc.) are preserved
    because they represent distinct phonemes in these languages.

    Returns the result in lowercase.

    Examples:
        "dóruma"       -> "doruma"
        "þàdórúmàgùùg" -> "þadorumaguug"
        "mṍzu"         -> "mozu"
    """
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()


def tokenise(surface: str) -> list[str]:
    """Split a surface sentence on whitespace and strip edge punctuation.

    Preserves internal hyphens, slashes, and apostrophes that are part of
    the word form.  Returns only non-empty tokens.
    """
    tokens = []
    for tok in surface.split():
        tok = re.sub(r"^[^\wÀ-žɐ-ʯḀ-ỿ]+", "", tok)
        tok = re.sub(r"[^\wÀ-žɐ-ʯḀ-ỿ]+$", "", tok)
        if tok:
            tokens.append(tok)
    return tokens


def lcs_length(a: str, b: str) -> int:
    """Return the longest common subsequence length of a and b.

    Uses the standard two-row DP algorithm in O(m*n) time and O(n) space.
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def lcs_similarity(a: str, b: str) -> float:
    """Compute 2 * LCS(a, b) / (len(a) + len(b)) on diacritic-stripped forms.

    This metric rewards shared character subsequences and is robust to the
    affixal variation found across the daughter languages.  Returns 0.0 when
    both strings are empty.

    Examples (approximate):
        "dóruma",  "þàdórúmàgùùg"  -> ~0.67   (genuine cognate)
        "dóruma",  "kávuru"          -> ~0.22   (unrelated words)
    """
    na, nb = strip_diacritics(a), strip_diacritics(b)
    total = len(na) + len(nb)
    if total == 0:
        return 0.0
    return 2.0 * lcs_length(na, nb) / total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(
    processed_dir: Path,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load dictionary.csv and corpus.csv from processed_dir.

    Returns:
        (dictionary_rows, corpus_by_language)
        dictionary_rows      — list of dicts with keys english, native_form, pos, …
        corpus_by_language   — {lang_id: [row, …]}  keyed by "lang1" … "lang7"
    """
    def read_csv(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    dictionary = read_csv(processed_dir / "dictionary.csv")

    corpus_by_lang: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(processed_dir / "corpus.csv"):
        corpus_by_lang[row["language"]].append(row)

    return dictionary, dict(corpus_by_lang)


# ---------------------------------------------------------------------------
# Search term extraction
# ---------------------------------------------------------------------------


def get_search_terms(english: str) -> list[str]:
    """Extract content-word search terms from an English dictionary meaning.

    Handles common patterns:
        "to create"      -> ["create"]
        "ring/circle"    -> ["ring", "circle"]
        "to flee/to run" -> ["flee", "run"]
        "point of view"  -> ["point", "view"]
    """
    parts = re.split(r"\s*/\s*", english.lower())
    terms: list[str] = []
    for part in parts:
        part = re.sub(r"^to\s+", "", part.strip())
        for word in part.split():
            word = word.strip(".,;:!?()")
            if word and len(word) > 2 and word not in _STOPWORDS:
                terms.append(word)
    return list(dict.fromkeys(terms))   # deduplicate while preserving order


# ---------------------------------------------------------------------------
# Lang1 interlinear gloss parsing
# ---------------------------------------------------------------------------


def _content_gloss(gls_token: str) -> Optional[str]:
    """Return the first lowercase morpheme gloss in gls_token, or None.

    Given "student-PL-DEF-ERG" returns "student".
    Given "INAN-welcome-WIT" returns "welcome".
    Given "ERG" or "1SG" returns None (purely grammatical).
    """
    for part in gls_token.rstrip(",").split("-"):
        clean = part.strip(".")
        if not clean:
            continue
        upper = clean.upper()
        if upper in _GRAM_TAGS:
            continue
        if clean[0].islower():
            return clean
    return None


def build_gloss_to_english(dictionary: list[dict]) -> dict[str, str]:
    """Build a mapping from content gloss words to dictionary English entries.

    Dictionary entries may be "to create", "to learn", etc.  Gloss lines
    use the bare verb form ("create", "learn").  This index handles that gap.

    Returns {gloss_word: full_english_entry}.
    """
    index: dict[str, str] = {}
    for entry in dictionary:
        eng = entry["english"]
        # Strip "to " for verb entries
        key = re.sub(r"^to\s+", "", eng.lower()).strip()
        # Also add just the first word for multi-word entries
        first_word = key.split()[0] if key.split() else key
        index[key] = eng
        index[first_word] = eng
    return index


def extract_lang1_word_inventory(
    lang1_rows: list[dict],
    gloss_to_english: dict[str, str],
) -> dict[int, list[tuple[str, str]]]:
    """Parse lang1 interlinear data into a per-sentence word inventory.

    For each lang1 corpus row, align surface tokens to segmented/gloss tokens
    positionally and extract (surface_token, english_meaning) pairs where the
    gloss identifies a content word present in the dictionary.

    Returns {sentence_id: [(surface_token, english_meaning), …]}.
    """
    inventory: dict[int, list[tuple[str, str]]] = {}

    for row in lang1_rows:
        segmented = row.get("segmented", "")
        gloss_line = row.get("gloss", "")
        if not segmented or not gloss_line:
            continue

        surf_tokens = tokenise(row.get("surface", ""))
        seg_tokens = [t.rstrip(",") for t in segmented.split()]
        gls_tokens = [t.rstrip(",") for t in gloss_line.split()]

        if not (len(surf_tokens) == len(seg_tokens) == len(gls_tokens)):
            continue

        pairs: list[tuple[str, str]] = []
        for surf, _seg, gls in zip(surf_tokens, seg_tokens, gls_tokens):
            gloss_word = _content_gloss(gls)
            if gloss_word is None:
                continue
            english = gloss_to_english.get(gloss_word)
            if english:
                pairs.append((surf, english))

        if pairs:
            sid = int(row["sentence_id"])
            inventory[sid] = pairs

    return inventory


# ---------------------------------------------------------------------------
# Parallel sentence alignment (primary method)
# ---------------------------------------------------------------------------


def find_parallel_groups(
    corpus_by_lang: dict[str, list[dict]],
) -> list[dict[str, dict]]:
    """Group sentences with identical translations across languages.

    Returns a list of groups.  Each group is a dict {language: corpus_row}
    containing at least two languages.
    """
    by_translation: dict[str, dict[str, dict]] = defaultdict(dict)
    for lang, rows in corpus_by_lang.items():
        for row in rows:
            t = row["translation"].strip()
            by_translation[t][lang] = row

    return [g for g in by_translation.values() if len(g) >= 2]


def extract_parallel_candidates(
    groups: list[dict[str, dict]],
    lang1_inventory: dict[int, list[tuple[str, str]]],
    dictionary_by_english: dict[str, dict],
    threshold: float = PARALLEL_THRESHOLD,
) -> list[CognateCandidate]:
    """Align lang2-7 tokens to lang1 tokens within each parallel sentence group.

    For each group that contains a lang1 sentence with known word meanings,
    finds the most LCS-similar token in each other language's sentence and
    records it as a CognateCandidate if it clears the threshold.
    """
    candidates: list[CognateCandidate] = []

    for group in groups:
        lang1_row = group.get("lang1")
        if lang1_row is None:
            continue

        lang1_sid = int(lang1_row["sentence_id"])
        word_pairs = lang1_inventory.get(lang1_sid)
        if not word_pairs:
            continue

        for lang, row in group.items():
            if lang == "lang1":
                continue

            langN_tokens = tokenise(row["surface"])
            langN_sid = int(row["sentence_id"])
            sentence_len = len(langN_tokens)

            for lang1_token, english in word_pairs:
                entry = dictionary_by_english.get(english)
                if entry is None:
                    continue

                # Find the best-matching token in the langN sentence
                best_sim = -1.0
                best_tok = ""
                for tok in langN_tokens:
                    sim = lcs_similarity(lang1_token, tok)
                    if sim > best_sim:
                        best_sim = sim
                        best_tok = tok

                if best_sim >= threshold:
                    candidates.append(
                        CognateCandidate(
                            english=english,
                            pos=entry["pos"],
                            lang1_form=entry["native_form"],
                            language=lang,
                            candidate_form=best_tok,
                            similarity=round(best_sim, 4),
                            method="parallel",
                            sentence_id=langN_sid,
                            sentence_length=sentence_len,
                        )
                    )

    return candidates


# ---------------------------------------------------------------------------
# Translation-search fallback
# ---------------------------------------------------------------------------


def extract_search_candidates(
    dictionary: list[dict],
    corpus_by_lang: dict[str, list[dict]],
    covered: set[tuple[str, str]],
    threshold: float = SEARCH_THRESHOLD,
    max_tokens: int = MAX_SEARCH_SENTENCE_TOKENS,
) -> list[CognateCandidate]:
    """Search-based candidate extraction for (english, language) pairs not yet covered.

    For each dictionary entry that has no parallel-method candidate in a given
    language, searches that language's corpus for sentences whose translation
    contains a keyword from the English meaning.  The most similar token in
    each matching sentence is kept if it clears the threshold.

    covered — set of (english, language) pairs already handled by the parallel
              method; skipped here to avoid duplication.
    """
    candidates: list[CognateCandidate] = []

    for entry in dictionary:
        english = entry["english"]
        lang1_form = entry["native_form"]
        if not lang1_form:
            continue

        search_terms = get_search_terms(english)
        if not search_terms:
            continue

        for lang in LANGUAGES[1:]:
            if (english, lang) in covered:
                continue

            lang_rows = corpus_by_lang.get(lang, [])
            for row in lang_rows:
                translation_lower = row["translation"].lower()
                if not any(term in translation_lower for term in search_terms):
                    continue

                tokens = tokenise(row["surface"])
                if not tokens or len(tokens) > max_tokens:
                    continue

                for tok in tokens:
                    sim = lcs_similarity(lang1_form, tok)
                    if sim >= threshold:
                        candidates.append(
                            CognateCandidate(
                                english=english,
                                pos=entry["pos"],
                                lang1_form=lang1_form,
                                language=lang,
                                candidate_form=tok,
                                similarity=round(sim, 4),
                                method="search",
                                sentence_id=int(row["sentence_id"]),
                                sentence_length=len(tokens),
                            )
                        )

    return candidates


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def best_candidate(
    candidates: list[CognateCandidate],
) -> Optional[CognateCandidate]:
    """Select the best candidate from a list for the same (english, language) pair.

    Ranking priority:
    1. Method: "parallel" outranks "search" (positional context is more reliable).
    2. Similarity: higher is better.
    3. Sentence length: shorter sentences have less token ambiguity.
    """
    if not candidates:
        return None

    def rank(c: CognateCandidate) -> tuple:
        method_score = 1 if c.method == "parallel" else 0
        return (method_score, c.similarity, -c.sentence_length)

    return max(candidates, key=rank)


def build_cognate_sets(
    all_candidates: list[CognateCandidate],
    dictionary: list[dict],
) -> list[CognateSet]:
    """Produce one CognateSet row per dictionary entry.

    For each (english, language) pair, selects the best candidate and
    records its form.  Entries with no candidate receive an empty string.
    """
    grouped: dict[tuple[str, str], list[CognateCandidate]] = defaultdict(list)
    for c in all_candidates:
        grouped[(c.english, c.language)].append(c)

    sets: list[CognateSet] = []
    for entry in dictionary:
        eng = entry["english"]
        forms: dict[str, str] = {"lang1": entry["native_form"]}

        for lang in LANGUAGES[1:]:
            bc = best_candidate(grouped.get((eng, lang), []))
            forms[lang] = bc.candidate_form if bc else ""

        sets.append(
            CognateSet(
                english=eng,
                pos=entry["pos"],
                lang1=forms["lang1"],
                lang2=forms["lang2"],
                lang3=forms["lang3"],
                lang4=forms["lang4"],
                lang5=forms["lang5"],
                lang6=forms["lang6"],
                lang7=forms["lang7"],
            )
        )

    return sets


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(records: list, output_path: Path) -> None:
    """Write a list of dataclass instances to a UTF-8 CSV file."""
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [f.name for f in fields(records[0])]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for record in records:
            writer.writerow([getattr(record, f.name) for f in fields(record)])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, list]:
    """Run the full Stage 2 pipeline.

    Steps:
    1.  Load dictionary.csv and corpus.csv.
    2.  Build the lang1 word inventory from interlinear glosses.
    3.  Find parallel sentence groups (same translation across languages).
    4.  Extract parallel-method candidates within those groups.
    5.  Extract search-fallback candidates for uncovered (english, language) pairs.
    6.  Aggregate best candidate per (english, language) into CognateSet rows.
    7.  Export cognate_candidates.csv and cognate_sets.csv.

    Returns {"candidates": list[CognateCandidate], "cognate_sets": list[CognateSet]}.
    """
    dictionary, corpus_by_lang = load_data(processed_dir)

    dictionary_by_english: dict[str, dict] = {e["english"]: e for e in dictionary}
    gloss_to_english = build_gloss_to_english(dictionary)

    lang1_inventory = extract_lang1_word_inventory(
        corpus_by_lang.get("lang1", []),
        gloss_to_english,
    )

    parallel_groups = find_parallel_groups(corpus_by_lang)

    parallel_candidates = extract_parallel_candidates(
        parallel_groups,
        lang1_inventory,
        dictionary_by_english,
    )

    # Track which (english, language) pairs have parallel coverage
    covered = {(c.english, c.language) for c in parallel_candidates}

    search_candidates = extract_search_candidates(
        dictionary,
        corpus_by_lang,
        covered,
    )

    all_candidates = parallel_candidates + search_candidates
    cognate_sets = build_cognate_sets(all_candidates, dictionary)

    export_csv(all_candidates, processed_dir / "cognate_candidates.csv")
    export_csv(cognate_sets, processed_dir / "cognate_sets.csv")

    return {"candidates": all_candidates, "cognate_sets": cognate_sets}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory containing dictionary.csv and corpus.csv (default: data/processed/)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    results = run(processed_dir=args.processed_dir)

    cands = results["candidates"]
    sets = results["cognate_sets"]
    n_parallel = sum(1 for c in cands if c.method == "parallel")
    n_search = sum(1 for c in cands if c.method == "search")
    n_full = sum(
        1 for s in sets
        if all(getattr(s, f"lang{i}") for i in range(1, 8))
    )
    covered_per_lang = {
        lang: sum(1 for s in sets if getattr(s, lang))
        for lang in [f"lang{i}" for i in range(1, 8)]
    }

    print(f"candidates      : {len(cands):>5}  "
          f"(parallel: {n_parallel}, search: {n_search})")
    print(f"cognate sets    : {len(sets):>5}")
    print(f"fully covered   : {n_full:>5}  (form found in all 7 languages)")
    print("coverage per language:")
    for lang, count in covered_per_lang.items():
        pct = 100 * count / len(sets) if sets else 0
        print(f"  {lang}  {count:>4} / {len(sets)}  ({pct:.0f}%)")
