"""
reconstruct_proto.py — Stage 4: Proto-language reconstruction.

Builds on the outputs of Stages 1–3 to produce:

    sound_laws.csv              one row per regular correspondence that meets the
                                regularity and count thresholds; these are the
                                proposed sound-change rules from proto to each
                                daughter language

    proto_lexicon.csv           one row per dictionary entry; contains the
                                reconstructed *proto_form, a confidence tier
                                (high / medium / low), and all seven language
                                forms side-by-side for comparison

    reconstruction_summary.md   human-readable account of the proto-language
                                phonological system, the sound laws per daughter
                                language, representative lexicon samples, and
                                coverage statistics

Reconstruction strategy
-----------------------
Lang1 has the richest documentation (dictionary + interlinear glosses) and
shares ~87 % of its graphemes unchanged with lang5 (the most conservative
daughter).  Lang1 citation forms are therefore used as the direct basis for
proto-form reconstruction:

    *proto_form = "*" + lang1_citation_form

Sound laws are then read as changes FROM the proto-language TO each daughter.
Where lang1 itself may have innovated (e.g. lang1 *á is changed to *à by all
six daughters), this is noted in the summary as a candidate proto-form
correction, but the main outputs keep the reconstruction grounded in lang1.

Run directly:
    python src/reconstruct_proto.py
    python src/reconstruct_proto.py --processed-dir path/to/processed
    python src/reconstruct_proto.py --regularity 0.8 --min-count 10
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = _ROOT / "data" / "processed"
LANGUAGES = [f"lang{i}" for i in range(1, 8)]

# A correspondence must meet both thresholds to become a sound law.
REGULARITY_THRESHOLD: float = 0.70
COUNT_THRESHOLD: int = 5

# Confidence tiers for proto-lexicon entries.
HIGH_COGNATE_MIN = 4
MEDIUM_COGNATE_MIN = 2

# Number of sample lexicon entries to include in the markdown summary.
SUMMARY_SAMPLE_SIZE = 20

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SoundLaw:
    """One proposed sound change from proto to a daughter language."""

    language: str
    proto_grapheme: str     # the proto-language grapheme (= lang1 grapheme)
    reflex: str             # the daughter-language grapheme
    is_change: bool         # False means the sound was preserved unchanged
    regularity: float       # fraction of proto_grapheme occurrences → this reflex
    count: int              # number of cognate pairs supporting this law
    examples: str           # up to 5 English meanings, comma-separated


@dataclass
class ProtoEntry:
    """One reconstructed proto-language lexicon entry."""

    english: str
    pos: str
    proto_form: str         # "*" + lang1 citation form
    confidence: str         # "high" / "medium" / "low"
    n_cognates: int         # number of daughter languages with a cognate form
    lang1: str
    lang2: str
    lang3: str
    lang4: str
    lang5: str
    lang6: str
    lang7: str


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_data(
    processed_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load correspondences.csv, cognate_sets.csv, and dictionary.csv.

    Returns (correspondences, cognate_sets, dictionary).
    """
    return (
        _read_csv(processed_dir / "correspondences.csv"),
        _read_csv(processed_dir / "cognate_sets.csv"),
        _read_csv(processed_dir / "dictionary.csv"),
    )


# ---------------------------------------------------------------------------
# Sound law extraction
# ---------------------------------------------------------------------------


def extract_sound_laws(
    correspondences: list[dict],
    regularity_threshold: float = REGULARITY_THRESHOLD,
    count_threshold: int = COUNT_THRESHOLD,
) -> list[SoundLaw]:
    """Filter correspondences to those meeting both thresholds.

    Only correspondences with regularity >= regularity_threshold AND
    count >= count_threshold are retained.  Identity correspondences
    (proto_grapheme == reflex) are included as "conservation" laws because
    they confirm which sounds were stable in that daughter language.

    Results are sorted by language, then by count descending.
    """
    laws: list[SoundLaw] = []
    for row in correspondences:
        reg = float(row["regularity"])
        cnt = int(row["count"])
        if reg < regularity_threshold or cnt < count_threshold:
            continue
        laws.append(
            SoundLaw(
                language=row["language"],
                proto_grapheme=row["lang1_grapheme"],
                reflex=row["langN_grapheme"],
                is_change=(row["is_identity"] == "False"),
                regularity=round(reg, 3),
                count=cnt,
                examples=row["examples"],
            )
        )

    laws.sort(key=lambda x: (x.language, -x.count))
    return laws


# ---------------------------------------------------------------------------
# Proto-lexicon construction
# ---------------------------------------------------------------------------


def _confidence(n_cognates: int) -> str:
    """Assign a confidence tier based on number of daughter cognates found."""
    if n_cognates >= HIGH_COGNATE_MIN:
        return "high"
    if n_cognates >= MEDIUM_COGNATE_MIN:
        return "medium"
    return "low"


def build_proto_lexicon(
    cognate_sets: list[dict],
    dictionary: list[dict],
) -> list[ProtoEntry]:
    """Build one ProtoEntry per dictionary entry.

    The proto_form is constructed by prepending "*" to the lang1 citation
    form.  Confidence is determined by how many of lang2–7 have a cognate
    form in the cognate set.

    Dictionary entries are the authoritative source for english and pos.
    Cognate sets are joined on the english field.
    """
    # Index cognate sets by english meaning
    cset_by_english: dict[str, dict] = {
        row["english"]: row for row in cognate_sets
    }

    entries: list[ProtoEntry] = []
    for d_row in dictionary:
        english = d_row["english"]
        lang1_form = d_row["native_form"]
        pos = d_row["pos"]

        cset = cset_by_english.get(english, {})
        lang_forms = {
            lang: cset.get(lang, "") for lang in LANGUAGES
        }
        # lang1 in the cognate set may differ from dictionary (inflected vs citation)
        # Always use the dictionary citation form for lang1
        lang_forms["lang1"] = lang1_form

        n_cognates = sum(1 for lang in LANGUAGES[1:] if lang_forms[lang])

        entries.append(
            ProtoEntry(
                english=english,
                pos=pos,
                proto_form=f"*{lang1_form}",
                confidence=_confidence(n_cognates),
                n_cognates=n_cognates,
                lang1=lang_forms["lang1"],
                lang2=lang_forms["lang2"],
                lang3=lang_forms["lang3"],
                lang4=lang_forms["lang4"],
                lang5=lang_forms["lang5"],
                lang6=lang_forms["lang6"],
                lang7=lang_forms["lang7"],
            )
        )

    return entries


# ---------------------------------------------------------------------------
# Summary statistics helpers
# ---------------------------------------------------------------------------


def _law_table(laws: list[SoundLaw], language: str) -> list[SoundLaw]:
    return [l for l in laws if l.language == language]


def _coverage_stats(entries: list[ProtoEntry]) -> dict:
    total = len(entries)
    by_conf = defaultdict(int)
    lang_coverage: dict[str, int] = defaultdict(int)
    for e in entries:
        by_conf[e.confidence] += 1
        for lang in LANGUAGES[1:]:
            if getattr(e, lang):
                lang_coverage[lang] += 1
    return {
        "total": total,
        "high": by_conf["high"],
        "medium": by_conf["medium"],
        "low": by_conf["low"],
        "lang_coverage": dict(lang_coverage),
    }


def _vowel_laws(laws: list[SoundLaw], language: str) -> list[SoundLaw]:
    vowel_bases = set("aeiouáéíóúàèìòùãẽĩõũ")
    return [
        l for l in laws
        if l.language == language and l.proto_grapheme in vowel_bases
    ]


def _change_laws(laws: list[SoundLaw], language: str) -> list[SoundLaw]:
    return [l for l in laws if l.language == language and l.is_change]


# ---------------------------------------------------------------------------
# Markdown summary generator
# ---------------------------------------------------------------------------


def generate_summary(
    laws: list[SoundLaw],
    entries: list[ProtoEntry],
    correspondences: list[dict],
) -> str:
    """Generate a human-readable reconstruction summary in Markdown.

    Sections:
      1. Overview
      2. Proto-phoneme inventory (vowels and consonants)
      3. Sound laws per daughter language
      4. Proto-lexicon sample
      5. Coverage statistics
      6. Candidate proto-form corrections
      7. Limitations
    """
    stats = _coverage_stats(entries)
    lang_cov = stats["lang_coverage"]

    # ── 1. Header ────────────────────────────────────────────────────────────
    lines: list[str] = [
        "# Proto-Language Reconstruction Summary",
        "",
        "Generated by `src/reconstruct_proto.py` (Stage 4 of the proto-language",
        "reconstruction pipeline).",
        "",
        "## Overview",
        "",
        "The proto-language documented here is the common ancestor of seven",
        "daughter languages (lang1–lang7).  Lang1 serves as the primary anchor",
        "for reconstruction because it has the most complete documentation",
        "(dictionary of 578 entries + full interlinear corpus).  Lang5 serves as",
        "a secondary anchor because it shows the highest phonological identity",
        "with lang1 (~87.7 % of graphemes unchanged).",
        "",
        f"- **Proto-lexicon entries**: {stats['total']}",
        f"- **High confidence** (cognates in ≥ 4 languages): {stats['high']}",
        f"- **Medium confidence** (2–3 languages): {stats['medium']}",
        f"- **Low confidence** (0–1 languages): {stats['low']}",
        "",
    ]

    # ── 2. Proto-phoneme inventory ────────────────────────────────────────────
    lines += [
        "## Proto-Phoneme Inventory",
        "",
        "The inventory is derived from lang1 citation forms.  Stability is",
        "assessed from Stage 3: a grapheme is *stable* if it is preserved",
        "unchanged in lang5 with regularity ≥ 0.90.",
        "",
        "### Vowels",
        "",
        "| Proto | lang1 | lang5 reflex | lang2 reflex | lang6 reflex | Status |",
        "|---|---|---|---|---|---|",
    ]

    # Build per-grapheme rows from correspondences
    def top_reflex(lang: str, g: str, corrs: list[dict]) -> str:
        candidates = [
            (int(r["count"]), r["langN_grapheme"])
            for r in corrs
            if r["language"] == lang and r["lang1_grapheme"] == g
        ]
        if not candidates:
            return "—"
        return max(candidates)[1]

    vowels_of_interest = [
        ("a", "unaccented low vowel"),
        ("á", "accented low vowel (acute)"),
        ("e", "unaccented mid-front vowel"),
        ("é", "accented mid-front vowel"),
        ("i", "unaccented high-front vowel"),
        ("í", "accented high-front vowel"),
        ("o", "unaccented mid-back vowel"),
        ("ó", "accented mid-back vowel"),
        ("u", "unaccented high-back vowel"),
        ("ú", "accented high-back vowel"),
    ]

    for g, label in vowels_of_interest:
        r5 = top_reflex("lang5", g, correspondences)
        r2 = top_reflex("lang2", g, correspondences)
        r6 = top_reflex("lang6", g, correspondences)
        if r5 == "—" and r2 == "—":
            continue
        stable = "stable" if r5 == g else "changed in lang5"
        # Check if all daughters show a different reflex (suggests lang1 innovated)
        daughter_reflexes = [
            top_reflex(f"lang{i}", g, correspondences) for i in range(2, 8)
        ]
        non_g = [r for r in daughter_reflexes if r != "—" and r != g]
        if non_g and all(r == non_g[0] for r in non_g):
            stable = f"⚠ lang1 may have raised *{non_g[0]} → {g}"
        lines.append(f"| *{g} | {g} | {r5} | {r2} | {r6} | {stable} |")

    lines += [
        "",
        "### Consonants",
        "",
        "All consonants (r, m, l, n, d, b, g, t, v, p, k, h, f, s, x, y, z, þ, ħ)",
        "are fully stable across all daughter languages (regularity = 1.00 in lang2,",
        "verified across lang3–lang7).  The proto-language consonant inventory is",
        "therefore identical to the lang1 consonant inventory.",
        "",
    ]

    # ── 3. Sound laws ─────────────────────────────────────────────────────────
    lines += [
        "## Sound Laws",
        "",
        "Only changes (non-identity correspondences) with regularity ≥",
        f"{REGULARITY_THRESHOLD:.0%} and count ≥ {COUNT_THRESHOLD} are listed.",
        "Conservation (identity) laws are omitted here for brevity.",
        "",
    ]

    for lang in LANGUAGES[1:]:
        change_laws = _change_laws(laws, lang)
        lines.append(f"### {lang}")
        lines.append("")
        if not change_laws:
            lines.append("No non-identity correspondences above threshold.")
        else:
            lines.append("| Proto | → | Reflex | Regularity | Count | Examples |")
            lines.append("|---|---|---|---|---|---|")
            for law in sorted(change_laws, key=lambda x: -x.count):
                lines.append(
                    f"| *{law.proto_grapheme} | → | {law.reflex} "
                    f"| {law.regularity:.0%} | {law.count} "
                    f"| {law.examples} |"
                )
        lines.append("")

    # ── 4. Proto-lexicon sample ───────────────────────────────────────────────
    # Show high-confidence entries first, prioritise short citation forms
    sample = sorted(
        [e for e in entries if e.confidence == "high"],
        key=lambda e: len(e.proto_form),
    )[:SUMMARY_SAMPLE_SIZE]

    lines += [
        "## Proto-Lexicon Sample",
        "",
        f"Top {len(sample)} high-confidence reconstructions (shortest forms first).",
        "",
        "| Proto form | English | POS | lang1 | lang2 | lang5 | lang6 |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in sample:
        lines.append(
            f"| {e.proto_form} | {e.english} | {e.pos} "
            f"| {e.lang1} | {e.lang2} | {e.lang5} | {e.lang6} |"
        )

    lines += [""]

    # ── 5. Coverage statistics ────────────────────────────────────────────────
    lines += [
        "## Coverage Statistics",
        "",
        "| Language | Entries with cognate | % of 578 |",
        "|---|---|---|",
    ]
    for lang in LANGUAGES:
        if lang == "lang1":
            lines.append(f"| lang1 | 578 | 100 % |")
        else:
            n = lang_cov.get(lang, 0)
            pct = 100 * n / stats["total"] if stats["total"] else 0
            lines.append(f"| {lang} | {n} | {pct:.0f} % |")

    lines += [""]

    # ── 6. Candidate proto-form corrections ───────────────────────────────────
    lines += [
        "## Candidate Proto-Form Corrections",
        "",
        "The following lang1 graphemes show evidence that lang1 itself may have",
        "innovated away from the proto-language, based on all daughters converging",
        "on a different reflex:",
        "",
        "| Lang1 grapheme | Likely proto-phoneme | Evidence |",
        "|---|---|---|",
        "| á | *à | All 5 daughters with data (lang2–6) map á → à (regularity 1.00) |",
        "| a | *à | Majority of daughters (lang2: 100 %, lang5: 79 %) map a → à |",
        "",
        "If these corrections are applied, all instances of lang1 `á` and `a` in",
        "proto-forms should be written as `*à`.  This is left optional in the",
        "machine-readable outputs so downstream stages can apply or ignore it.",
        "",
    ]

    # ── 7. Limitations ────────────────────────────────────────────────────────
    lines += [
        "## Limitations",
        "",
        "- **Inflected forms**: cognate_sets.csv contains surface tokens from",
        "  sentences, not citation forms.  Proto-forms are anchored on lang1",
        "  dictionary citation forms, which are the cleanest available data.",
        "",
        "- **Conditioned changes not analysed**: in lang3/4/6/7, many vowels show",
        "  splits (e.g. *u → ú ~60 % vs ṹ ~40 % in lang3).  The conditioning",
        "  environment is not yet identified — this requires morpheme-segmented",
        "  data for lang2–7 (Stage 5 work).",
        "",
        "- **Low-confidence entries**: 578 − "
        f"{stats['high'] + stats['medium']} = "
        f"{stats['low']} entries have 0–1 cognate forms.  Their proto-forms",
        "  rest entirely on lang1 evidence and should be treated as provisional.",
        "",
        "- **Gap correspondences**: phoneme deletions (e.g. *ó → ∅ in lang3 for",
        "  some words) are not tracked in Stage 3 and are therefore absent from",
        "  the sound laws.  They are visible in the correspondence table as",
        "  reduced LCS coverage.",
    ]

    return "\n".join(lines) + "\n"


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
    regularity_threshold: float = REGULARITY_THRESHOLD,
    count_threshold: int = COUNT_THRESHOLD,
) -> dict[str, object]:
    """Run the full Stage 4 pipeline.

    Steps:
    1.  Load correspondences.csv, cognate_sets.csv, dictionary.csv.
    2.  Extract sound laws (correspondences meeting both thresholds).
    3.  Build proto-lexicon (one entry per dictionary word).
    4.  Generate human-readable reconstruction summary.
    5.  Export sound_laws.csv, proto_lexicon.csv, reconstruction_summary.md.

    Returns:
        {
          "sound_laws":   list[SoundLaw],
          "proto_lexicon": list[ProtoEntry],
          "summary":      str,
        }
    """
    correspondences, cognate_sets, dictionary = load_data(processed_dir)

    laws = extract_sound_laws(correspondences, regularity_threshold, count_threshold)
    proto_lexicon = build_proto_lexicon(cognate_sets, dictionary)
    summary = generate_summary(laws, proto_lexicon, correspondences)

    export_csv(laws, processed_dir / "sound_laws.csv")
    export_csv(proto_lexicon, processed_dir / "proto_lexicon.csv")

    summary_path = processed_dir / "reconstruction_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    return {
        "sound_laws": laws,
        "proto_lexicon": proto_lexicon,
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
        help="Directory with Stage 1–3 outputs (default: data/processed/)",
    )
    p.add_argument(
        "--regularity",
        type=float,
        default=REGULARITY_THRESHOLD,
        help=f"Min regularity to include a correspondence as a sound law "
             f"(default: {REGULARITY_THRESHOLD})",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=COUNT_THRESHOLD,
        help=f"Min cognate count to include a correspondence as a sound law "
             f"(default: {COUNT_THRESHOLD})",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    results = run(
        processed_dir=args.processed_dir,
        regularity_threshold=args.regularity,
        count_threshold=args.min_count,
    )

    laws = results["sound_laws"]
    lexicon = results["proto_lexicon"]
    stats = _coverage_stats(lexicon)

    n_change = sum(1 for l in laws if l.is_change)
    n_stable = sum(1 for l in laws if not l.is_change)

    print(f"sound laws      : {len(laws):>4}  "
          f"(changes: {n_change}, conservation: {n_stable})")
    print(f"proto-lexicon   : {len(lexicon):>4}  "
          f"(high: {stats['high']}, medium: {stats['medium']}, low: {stats['low']})")
    print()
    print("Sound changes per language:")
    for lang in LANGUAGES[1:]:
        changes = [l for l in laws if l.language == lang and l.is_change]
        stable = [l for l in laws if l.language == lang and not l.is_change]
        print(f"  {lang}  {len(changes):>2} changes  {len(stable):>2} stable")
    print()
    print("Top sound changes across all languages:")
    all_changes = sorted(
        [l for l in laws if l.is_change], key=lambda x: -x.count
    )
    for law in all_changes[:10]:
        print(
            f"  {law.language}: *{law.proto_grapheme} → {law.reflex}"
            f"  reg={law.regularity:.0%}  n={law.count}"
        )
