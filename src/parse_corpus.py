"""
parse_corpus.py — Stage 1: Parse raw language data into structured CSVs.

Reads the seven daughter-language files from data/raw/Hackaton 2026/ and
writes three normalised CSV files to data/processed/:

    dictionary.csv  — lang1 vocabulary with POS and compound annotations
    corpus.csv      — all inter-linear glossed sentences across all languages
    docs.csv        — all prose document lines across all languages

Run directly:
    python src/parse_corpus.py
    python src/parse_corpus.py --raw-dir path/to/raw --out-dir path/to/out

See docs/parse_corpus.md for a full description of the input formats and
output schemas.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path defaults
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).parent
_ROOT_DIR = _SRC_DIR.parent
RAW_DIR = _ROOT_DIR / "data" / "raw" / "Hackaton 2026"
PROCESSED_DIR = _ROOT_DIR / "data" / "processed"

LANGUAGES = [f"lang{i}" for i in range(1, 8)]

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class DictionaryEntry:
    """One entry from lang1-dictionary.txt."""

    english: str
    native_form: str
    compound_explanation: str   # empty string when no (ie, ...) annotation
    pos: str


@dataclass
class CorpusEntry:
    """One sentence from a langN-corpus.txt file."""

    language: str
    sentence_id: int
    surface: str
    segmented: str      # morpheme-segmented form; empty for lang2-7
    gloss: str          # morpheme-by-morpheme gloss; empty for lang2-7
    translation: str    # English free translation


@dataclass
class DocEntry:
    """One non-empty line from a langN-docM.txt file."""

    language: str
    doc_id: int
    line_no: int
    text: str


# ---------------------------------------------------------------------------
# Dictionary parser
# ---------------------------------------------------------------------------


def parse_dictionary(path: Path) -> list[DictionaryEntry]:
    """Parse lang1-dictionary.txt into a list of DictionaryEntry objects.

    File format — each line is tab-separated (or 2+ spaces) with three
    fields:

        english_meaning    native_form    POS

    The native_form may carry a compound annotation in parentheses:

        fálurunadóruma (ie, star-scholar)

    which is split into native_form='fálurunadóruma' and
    compound_explanation='star-scholar'.

    Lines that do not contain at least three parseable fields are skipped.
    """
    entries: list[DictionaryEntry] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = _split_dict_line(line)
        if len(parts) < 3:
            continue

        english = parts[0].strip()
        raw_form = parts[1].strip()
        pos = parts[2].strip()

        if not english or not raw_form or not pos:
            continue

        native_form, compound_explanation = _extract_compound(raw_form)
        entries.append(
            DictionaryEntry(
                english=english,
                native_form=native_form,
                compound_explanation=compound_explanation,
                pos=pos,
            )
        )
    return entries


_POS_PATTERN = re.compile(
    r"\s+(NN|NNP|VB|JJ|RB|DT|CONJ|WH|IN|CARD|MD|SCONJ|ADP|UH|INSTR)\s*$"
)


def _split_dict_line(line: str) -> list[str]:
    """Split a dictionary line into [english, native_form, pos].

    Strategy (in order):
    1. Tab-separated — split on tabs.
    2. 2+-space-separated — all three fields are present when split gives ≥3 parts.
    3. POS-tag fallback — when inconsistent spacing produces only 2 parts, the
       known POS tag at line-end is extracted first; the remainder is split into
       english and native_form on the last whitespace boundary (accommodating
       compound annotations like 'form (ie, explanation)').

    Lines with no recognisable POS tag in the fallback path are returned as-is
    so the caller can skip them.
    """
    if "\t" in line:
        return line.split("\t")

    parts = re.split(r"\s{2,}", line)
    if len(parts) >= 3:
        return parts

    # Fallback: anchor on the known POS tag at end of line
    m = _POS_PATTERN.search(line)
    if not m:
        return parts  # no recognisable POS; caller will skip

    pos = m.group(1)
    rest = line[: m.start()].strip()

    # If native_form carries a compound annotation '(ie, ...)', keep it intact
    ann_m = re.search(r"\s+(\S+\s*\(ie,\s*.+?\))\s*$", rest)
    if ann_m:
        native = ann_m.group(1).strip()
        english = rest[: ann_m.start()].strip()
    else:
        # Native form is the last whitespace-delimited token
        idx = rest.rfind(" ")
        if idx == -1:
            return parts  # cannot split further
        native = rest[idx:].strip()
        english = rest[:idx].strip()

    return [english, native, pos]


def _extract_compound(raw_form: str) -> tuple[str, str]:
    """Separate 'form (ie, explanation)' into (form, explanation).

    Returns (raw_form, '') when no annotation is present.
    """
    m = re.match(r"^(.+?)\s*\(ie,\s*(.+?)\)\s*$", raw_form)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return raw_form.strip(), ""


# ---------------------------------------------------------------------------
# Corpus parser
# ---------------------------------------------------------------------------


def parse_corpus(path: Path, language: str) -> list[CorpusEntry]:
    """Parse a langN-corpus.txt file into a list of CorpusEntry objects.

    lang1 uses 4-line interlinear blocks:
        <surface form>
        <morpheme-segmented form>
        <morpheme-by-morpheme gloss>
        "<English translation>"

    lang2-7 use 2-line blocks:
        <surface form>
        "<English translation>"

    Blocks are delimited by one or more blank lines.  Entries that lack a
    non-empty translation are silently skipped (they are partial/incomplete
    corpus lines present in some files).

    sentence_id is a 1-based counter over the accepted entries.
    """
    text = path.read_text(encoding="utf-8")
    blocks = _split_blocks(text)

    is_lang1 = language == "lang1"
    entries: list[CorpusEntry] = []
    sentence_id = 1

    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        if is_lang1:
            entry = _parse_lang1_block(lines, language, sentence_id)
        else:
            entry = _parse_lang2_7_block(lines, language, sentence_id)

        if entry is not None:
            entries.append(entry)
            sentence_id += 1

    return entries


def _split_blocks(text: str) -> list[str]:
    """Split text into non-empty paragraph blocks on blank lines."""
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def _parse_lang1_block(
    lines: list[str], language: str, sid: int
) -> Optional[CorpusEntry]:
    """Parse a 4-line lang1 interlinear block.

    Returns None when fewer than 4 non-empty lines are present or when the
    translation is empty after stripping quotes.
    """
    if len(lines) < 4:
        return None

    surface = lines[0]
    segmented = lines[1]
    gloss = lines[2]
    translation = _clean_translation(lines[3])

    if not translation:
        return None

    return CorpusEntry(
        language=language,
        sentence_id=sid,
        surface=surface,
        segmented=segmented,
        gloss=gloss,
        translation=translation,
    )


def _parse_lang2_7_block(
    lines: list[str], language: str, sid: int
) -> Optional[CorpusEntry]:
    """Parse a 2-line corpus block for lang2-7.

    Returns None when fewer than 2 non-empty lines are present or when the
    translation is empty after stripping quotes.
    """
    if len(lines) < 2:
        return None

    surface = lines[0]
    translation = _clean_translation(lines[1])

    if not translation:
        return None

    return CorpusEntry(
        language=language,
        sentence_id=sid,
        surface=surface,
        segmented="",
        gloss="",
        translation=translation,
    )


def _clean_translation(raw: str) -> str:
    """Strip surrounding whitespace and double-quote characters."""
    return raw.strip().strip('"').strip()


# ---------------------------------------------------------------------------
# Doc parser
# ---------------------------------------------------------------------------


def parse_doc(path: Path, language: str, doc_id: int) -> list[DocEntry]:
    """Parse a langN-docM.txt file into a list of DocEntry objects.

    Doc files are continuous prose with no interlinear glosses or
    translations.  Each non-empty line is recorded as a separate entry.
    line_no is 1-based and refers to the original file position.
    """
    entries: list[DocEntry] = []
    for line_no, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = raw_line.strip()
        if text:
            entries.append(
                DocEntry(language=language, doc_id=doc_id, line_no=line_no, text=text)
            )
    return entries


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(records: list, output_path: Path) -> None:
    """Write a list of dataclass instances to a UTF-8 CSV file.

    The header row is derived from the dataclass field names.  Does nothing
    when records is empty.
    """
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
# Top-level orchestrator
# ---------------------------------------------------------------------------


def parse_all(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, list]:
    """Parse every raw language file and write results to processed_dir.

    Returns a dict with three keys:
        'dictionary'  — list[DictionaryEntry]
        'corpus'      — list[CorpusEntry]  (all languages combined)
        'docs'        — list[DocEntry]     (all languages combined)
    """
    results: dict[str, list] = {"dictionary": [], "corpus": [], "docs": []}

    dict_path = raw_dir / "lang1-dictionary.txt"
    if dict_path.exists():
        results["dictionary"] = parse_dictionary(dict_path)
        export_csv(results["dictionary"], processed_dir / "dictionary.csv")

    for lang in LANGUAGES:
        corpus_path = raw_dir / f"{lang}-corpus.txt"
        if corpus_path.exists():
            results["corpus"].extend(parse_corpus(corpus_path, lang))

        for doc_id in (1, 2):
            doc_path = raw_dir / f"{lang}-doc{doc_id}.txt"
            if doc_path.exists():
                results["docs"].extend(parse_doc(doc_path, lang, doc_id))

    export_csv(results["corpus"], processed_dir / "corpus.csv")
    export_csv(results["docs"], processed_dir / "docs.csv")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Path to raw data directory")
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DIR, help="Path to output directory")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    results = parse_all(raw_dir=args.raw_dir, processed_dir=args.out_dir)
    print(f"dictionary : {len(results['dictionary']):>4} entries")
    print(f"corpus     : {len(results['corpus']):>4} sentences")
    print(f"docs       : {len(results['docs']):>4} lines")
