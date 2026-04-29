"""
sound_correspondences.py — Stage 3: Sound correspondence analysis.

Reads cognate_sets.csv (Stage 2 output) and extracts systematic character-
level correspondences between lang1 and each of lang2-7.

For each cognate pair the lang1 citation form and the langN surface token are
split into grapheme clusters (base character + any combining diacritical marks).
The two grapheme sequences are then aligned by their longest common subsequence
(LCS) on stripped base characters.  Every aligned grapheme pair is a
correspondence data point.

Outputs:
    correspondences.csv        every (language, lang1_grapheme, langN_grapheme)
                               observed across cognate pairs, with count,
                               regularity, and example English meanings
    correspondence_table.csv   wide-format view: one row per lang1 grapheme,
                               one column per daughter language, showing the
                               most common mappings with occurrence counts

Run directly:
    python src/sound_correspondences.py
    python src/sound_correspondences.py --processed-dir path/to/processed
    python src/sound_correspondences.py --min-cognates 3
"""

from __future__ import annotations

import argparse
import csv
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = _ROOT / "data" / "processed"
LANGUAGES = [f"lang{i}" for i in range(1, 8)]

# A correspondence must appear in at least this many distinct cognate pairs
# to be reported as a regular sound-change candidate.
MIN_COGNATES: int = 2

# Forms shorter than this (in graphemes) are skipped.  Very short forms are
# often inflectional morphemes or function words that add noise.
MIN_FORM_GRAPHEMES: int = 3

# Maximum number of example English meanings recorded per correspondence.
MAX_EXAMPLES: int = 5

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Correspondence:
    """One attested (lang1_grapheme, langN_grapheme) mapping in one language."""

    language: str
    lang1_grapheme: str     # original grapheme as it appears in the lang1 form
    langN_grapheme: str     # corresponding grapheme in the langN form
    is_identity: bool       # True when lang1_grapheme == langN_grapheme exactly
    count: int              # number of cognate pairs showing this mapping
    regularity: float       # count / total_occurrences_of_lang1_grapheme in this language
    examples: str           # up to MAX_EXAMPLES English meanings, comma-separated


@dataclass
class TableRow:
    """One row of the wide-format correspondence table (one lang1 grapheme)."""

    lang1_grapheme: str
    lang2: str
    lang3: str
    lang4: str
    lang5: str
    lang6: str
    lang7: str


# ---------------------------------------------------------------------------
# Grapheme utilities
# ---------------------------------------------------------------------------


def grapheme_split(s: str) -> list[str]:
    """Split s into grapheme clusters, each normalised to NFC.

    A grapheme cluster is one base character plus any immediately following
    combining diacritical marks (Unicode category Mn).  The result preserves
    the original diacritic information — nothing is stripped at this stage.

    Examples:
        "dóruma"       -> ["d", "ó", "r", "u", "m", "a"]
        "þàdórúmàgù"   -> ["þ", "à", "d", "ó", "r", "ú", "m", "à", "g", "ù"]
        "ã̀"            -> ["ã̀"]   (a + tilde + grave = one grapheme)
    """
    nfd = unicodedata.normalize("NFD", s)
    graphemes: list[str] = []
    j = 0
    while j < len(nfd):
        if unicodedata.category(nfd[j]) == "Mn":
            j += 1
            continue  # skip any leading stray combining marks
        start = j
        j += 1
        while j < len(nfd) and unicodedata.category(nfd[j]) == "Mn":
            j += 1
        graphemes.append(unicodedata.normalize("NFC", nfd[start:j]))
    return graphemes


def grapheme_base(g: str) -> str:
    """Return the lowercase base character of grapheme g (no combining marks).

    This is used only for LCS matching, not for correspondence reporting.
    The original grapheme is always preserved in the output.

    Examples:
        "ó" -> "o"
        "ã̀" -> "a"
        "þ"  -> "þ"   (thorn has no combining marks)
    """
    nfd = unicodedata.normalize("NFD", g)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()


# ---------------------------------------------------------------------------
# LCS grapheme alignment
# ---------------------------------------------------------------------------


def lcs_align(
    a: list[str],
    b: list[str],
) -> list[tuple[str, str]]:
    """Align two grapheme sequences by their LCS on base characters.

    Returns a list of (a_grapheme, b_grapheme) pairs for matched positions.
    Unmatched graphemes (insertions or deletions relative to the LCS) are
    silently dropped — they represent potential sound changes not yet analysed
    (see docs/sound_correspondences.md §Limitations).

    Uses the standard two-row DP to compute the LCS, then backtracks once
    to recover the aligned pairs.
    """
    a_bases = [grapheme_base(g) for g in a]
    b_bases = [grapheme_base(g) for g in b]

    m, n = len(a_bases), len(b_bases)
    if m == 0 or n == 0:
        return []

    # Forward pass — build the full DP table (needed for traceback)
    dp: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a_bases[i - 1] == b_bases[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Traceback — collect matched pairs in reverse, then reverse
    pairs: list[tuple[str, str]] = []
    i, j = m, n
    while i > 0 and j > 0:
        if a_bases[i - 1] == b_bases[j - 1]:
            pairs.append((a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# Correspondence extraction
# ---------------------------------------------------------------------------


def extract_correspondences(
    cognate_sets: list[dict],
) -> tuple[
    dict[str, Counter],                          # counts[lang][(a_g, b_g)] = n
    dict[str, dict[tuple[str, str], list[str]]], # examples[lang][(a_g, b_g)] = [english, ...]
]:
    """Extract all (lang1_grapheme, langN_grapheme) pairs from cognate_sets.

    For each row in cognate_sets, the lang1 citation form and each langN form
    are split into grapheme clusters and aligned with lcs_align.  Every
    matched pair increments the counter for that language.

    Rows are skipped when either form is empty or shorter than
    MIN_FORM_GRAPHEMES graphemes.

    Returns:
        counts   — {language: Counter({(lang1_g, langN_g): count})}
        examples — {language: {(lang1_g, langN_g): [english, ...]}}
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, dict[tuple[str, str], list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in cognate_sets:
        lang1_form = row.get("lang1", "").strip()
        if not lang1_form:
            continue

        lang1_graphemes = grapheme_split(lang1_form)
        if len(lang1_graphemes) < MIN_FORM_GRAPHEMES:
            continue

        english = row.get("english", "")

        for lang in LANGUAGES[1:]:
            langN_form = row.get(lang, "").strip()
            if not langN_form:
                continue

            langN_graphemes = grapheme_split(langN_form)
            if len(langN_graphemes) < MIN_FORM_GRAPHEMES:
                continue

            pairs = lcs_align(lang1_graphemes, langN_graphemes)
            for a_g, b_g in pairs:
                key = (a_g, b_g)
                counts[lang][key] += 1
                if len(examples[lang][key]) < MAX_EXAMPLES:
                    examples[lang][key].append(english)

    return dict(counts), dict(examples)


# ---------------------------------------------------------------------------
# Regularity calculation
# ---------------------------------------------------------------------------


def compute_regularity(
    counts: dict[str, Counter],
) -> dict[str, dict[tuple[str, str], float]]:
    """Compute P(langN_grapheme | lang1_grapheme, language) for every mapping.

    Regularity is the fraction of occurrences of a lang1 grapheme (in a given
    language's cognate pairs) that correspond to a specific langN grapheme.
    A regularity of 1.0 means the change is exceptionless; 0.5 means it
    applies in half the observed cognate pairs.

    Returns {language: {(lang1_g, langN_g): regularity}}.
    """
    regularity: dict[str, dict[tuple[str, str], float]] = {}
    for lang, counter in counts.items():
        # Total occurrences of each lang1 grapheme in this language's cognate data
        lang1_totals: Counter = Counter()
        for (a_g, _b_g), cnt in counter.items():
            lang1_totals[a_g] += cnt

        lang_reg: dict[tuple[str, str], float] = {}
        for (a_g, b_g), cnt in counter.items():
            total = lang1_totals[a_g]
            lang_reg[(a_g, b_g)] = cnt / total if total > 0 else 0.0

        regularity[lang] = lang_reg

    return regularity


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------


def build_correspondence_records(
    counts: dict[str, Counter],
    regularity: dict[str, dict[tuple[str, str], float]],
    examples: dict[str, dict[tuple[str, str], list[str]]],
    min_count: int = MIN_COGNATES,
) -> list[Correspondence]:
    """Build Correspondence dataclass records, sorted by language then count desc.

    Only correspondences with count >= min_count are included.
    """
    records: list[Correspondence] = []
    for lang in LANGUAGES[1:]:
        counter = counts.get(lang, Counter())
        lang_reg = regularity.get(lang, {})
        lang_ex = examples.get(lang, {})

        for (a_g, b_g), cnt in counter.most_common():
            if cnt < min_count:
                continue
            key = (a_g, b_g)
            records.append(
                Correspondence(
                    language=lang,
                    lang1_grapheme=a_g,
                    langN_grapheme=b_g,
                    is_identity=(a_g == b_g),
                    count=cnt,
                    regularity=round(lang_reg.get(key, 0.0), 3),
                    examples=", ".join(lang_ex.get(key, [])),
                )
            )
    return records


def build_table_rows(
    counts: dict[str, Counter],
    min_count: int = MIN_COGNATES,
) -> list[TableRow]:
    """Build wide-format correspondence table, one row per lang1 grapheme.

    Each language cell contains the top 3 observed mappings in descending
    count order, formatted as "grapheme:count" and separated by spaces.
    Cells are empty when no mapping meets min_count for that language.

    Rows are sorted by lang1 grapheme for readability.
    """
    # Collect all lang1 graphemes that have at least one qualifying mapping
    all_lang1_graphemes: set[str] = set()
    for lang, counter in counts.items():
        for (a_g, _b_g), cnt in counter.items():
            if cnt >= min_count:
                all_lang1_graphemes.add(a_g)

    rows: list[TableRow] = []
    for a_g in sorted(all_lang1_graphemes):
        lang_cells: dict[str, str] = {}
        for lang in LANGUAGES[1:]:
            counter = counts.get(lang, Counter())
            mappings = [
                (b_g, cnt)
                for (ag, b_g), cnt in counter.items()
                if ag == a_g and cnt >= min_count
            ]
            mappings.sort(key=lambda x: -x[1])
            lang_cells[lang] = "  ".join(
                f"{b_g}:{cnt}" for b_g, cnt in mappings[:3]
            )

        rows.append(
            TableRow(
                lang1_grapheme=a_g,
                lang2=lang_cells.get("lang2", ""),
                lang3=lang_cells.get("lang3", ""),
                lang4=lang_cells.get("lang4", ""),
                lang5=lang_cells.get("lang5", ""),
                lang6=lang_cells.get("lang6", ""),
                lang7=lang_cells.get("lang7", ""),
            )
        )
    return rows


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
# Summary statistics
# ---------------------------------------------------------------------------


def compute_summary(
    counts: dict[str, Counter],
    min_count: int = MIN_COGNATES,
) -> dict:
    """Compute summary statistics for the CLI report.

    Returns a dict with:
        total_correspondences   total (lang, a_g, b_g) triples above min_count
        identity_rate           {language: fraction of pairs where a_g == b_g (base)}
        top_changes             {language: [(a_g, b_g, count), ...] top-5 non-identity}
    """
    total = 0
    identity_rate: dict[str, float] = {}
    top_changes: dict[str, list[tuple[str, str, int]]] = {}

    for lang in LANGUAGES[1:]:
        counter = counts.get(lang, Counter())
        n_identity = 0
        n_total = 0
        changes: list[tuple[str, str, int]] = []

        for (a_g, b_g), cnt in counter.items():
            if cnt < min_count:
                continue
            total += 1
            n_total += cnt
            if a_g == b_g:
                n_identity += cnt
            else:
                changes.append((a_g, b_g, cnt))

        identity_rate[lang] = n_identity / n_total if n_total > 0 else 0.0
        changes.sort(key=lambda x: -x[2])
        top_changes[lang] = changes[:5]

    return {
        "total_correspondences": total,
        "identity_rate": identity_rate,
        "top_changes": top_changes,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    processed_dir: Path = PROCESSED_DIR,
    min_count: int = MIN_COGNATES,
) -> dict[str, list]:
    """Run the full Stage 3 pipeline.

    Steps:
    1.  Load cognate_sets.csv from processed_dir.
    2.  Split each form into grapheme clusters.
    3.  Align lang1 and langN grapheme sequences by LCS on base characters.
    4.  Count every (lang1_grapheme, langN_grapheme) pair per language.
    5.  Compute regularity (conditional probability) for each mapping.
    6.  Build Correspondence records (filtered by min_count).
    7.  Build wide-format TableRow records.
    8.  Export both outputs to processed_dir.

    Returns:
        {
          "correspondences": list[Correspondence],
          "table_rows":       list[TableRow],
          "summary":          dict,
        }
    """
    path = processed_dir / "cognate_sets.csv"
    if not path.exists():
        return {"correspondences": [], "table_rows": [], "summary": {}}

    with path.open(encoding="utf-8") as fh:
        cognate_sets = list(csv.DictReader(fh))

    counts, examples = extract_correspondences(cognate_sets)
    regularity = compute_regularity(counts)

    corr_records = build_correspondence_records(counts, regularity, examples, min_count)
    table_rows = build_table_rows(counts, min_count)
    summary = compute_summary(counts, min_count)

    export_csv(corr_records, processed_dir / "correspondences.csv")
    export_csv(table_rows, processed_dir / "correspondence_table.csv")

    return {
        "correspondences": corr_records,
        "table_rows": table_rows,
        "summary": summary,
    }


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
        help="Directory containing cognate_sets.csv (default: data/processed/)",
    )
    p.add_argument(
        "--min-cognates",
        type=int,
        default=MIN_COGNATES,
        help=f"Minimum cognate pairs required to report a correspondence (default: {MIN_COGNATES})",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    results = run(processed_dir=args.processed_dir, min_count=args.min_cognates)

    summary = results["summary"]
    print(f"correspondences reported : {summary['total_correspondences']:>5}  "
          f"(min_count={args.min_cognates})")
    print(f"unique lang1 graphemes   : {len(results['table_rows']):>5}")
    print()

    id_rates = summary.get("identity_rate", {})
    print("Identity rate per language (higher = more conservative / less changed):")
    for lang in LANGUAGES[1:]:
        rate = id_rates.get(lang, 0.0)
        bar = "█" * int(rate * 20)
        print(f"  {lang}  {rate:.2%}  {bar}")

    print()
    print("Top 5 non-identity correspondences per language:")
    for lang in LANGUAGES[1:]:
        changes = summary.get("top_changes", {}).get(lang, [])
        if changes:
            parts = "  ".join(f"{a}→{b}({n})" for a, b, n in changes)
            print(f"  {lang}: {parts}")
        else:
            print(f"  {lang}: (no changes above threshold)")
