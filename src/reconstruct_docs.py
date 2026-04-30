"""
reconstruct_docs.py — Produce doc1 and doc2 in the proto-language.

Reconstructs the two prose documents (doc1, doc2) from all seven daughter
languages into the proto-language, using the proto-lexicon and proto-morpheme
inventory built in Stages 4 and 6.

Reconstruction strategy (three tiers, applied in order):

    Tier 1 — Citation match
        The cleaned token matches a lang1 citation form in proto_lexicon.csv.
        The stored *proto_form is used directly.
        Example: vúlunudonu → *vúlunu  (citation form of 'heaven')

    Tier 2 — Morphological segmentation
        The token was seen in the lang1 interlinear corpus.  Its segmented
        form is retrieved, each morpheme is replaced with its proto-citation
        form (for lexical morphemes) or proto-morpheme (for grammatical ones),
        and the pieces are joined with '-'.
        Example: izamimi → i-sami-mi → *i-sami-mi  (INAN-be-WIT)

    Tier 3 — Phonological fallback
        No prior match.  The lang1 surface token is used as-is with '*' prefix,
        applying the Stage 4 candidate correction á → à.
        Example: Tára → *Tàra

Outputs:
    output/proto_doc1.txt   proto-language document 1
    output/proto_doc2.txt   proto-language document 2

Each output file contains:
    - The full proto-language text (main body)
    - A line-by-line comparison section showing proto vs lang1 vs all other
      daughter languages side by side
    - A reconstruction statistics section

Run:
    python -m src.reconstruct_docs
    python -m src.reconstruct_docs --processed-dir path/to/processed
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = _ROOT / "data" / "processed"
OUTPUT_DIR = _ROOT / "output"
LANGUAGES = [f"lang{i}" for i in range(1, 8)]

# Grammatical morpheme → proto-morpheme form
PROTO_GRAM: dict[str, str] = {
    # Agreement prefixes (bound morphemes only)
    "i": "i", "a": "a", "u": "a",
    # TAM / evidentiality suffixes
    "mi": "mi", "mu": "mi",
    "me": "me",
    "ke": "ke", "ge": "ke", "go": "ke",
    # Negation — canonical form is *ne; do NOT include "ni" here
    # because "ni" is also the 1SG pronoun (free morpheme)
    "ne": "ne", "no": "ne",
    # Case suffixes
    "ko": "ko", "o": "o", "e": "e",
    "li": "li", "lu": "lu",
    "nu": "nu",
    "ta": "ta", "da": "ta",
    # Valence
    "su": "su",
    # Number / agreement
    "te": "te", "po": "po",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    """Lowercase + strip combining diacritics (for matching only)."""
    nfd = unicodedata.normalize("NFD", s.strip())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()


def _clean(tok: str) -> str:
    """Strip leading/trailing non-word characters."""
    return re.sub(r"[^\wÀ-žɐ-ʯḀ-ỿ]+$", "", re.sub(r"^[^\wÀ-žɐ-ʯḀ-ỿ]+", "", tok))


def _trailing_punct(tok: str) -> str:
    m = re.search(r"[^\wÀ-žɐ-ʯḀ-ỿ]+$", tok)
    return m.group() if m else ""


def _apply_proto_corrections(form: str) -> str:
    """Apply Stage 4 candidate proto-form corrections (á → à, á → à)."""
    return form.replace("á", "à").replace("Á", "À")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_resources(processed_dir: Path) -> dict:
    """Load all lookup tables needed for reconstruction."""

    def read(name: str) -> list[dict]:
        p = processed_dir / name
        if not p.exists():
            return []
        with p.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    corpus = read("corpus.csv")
    lexicon = read("proto_lexicon.csv")
    docs = read("docs.csv")

    # citation_to_proto: norm(lang1_citation) → (proto_form, english)
    citation_to_proto: dict[str, tuple[str, str]] = {}
    for row in lexicon:
        l1 = row["lang1"].strip()
        if l1:
            citation_to_proto[_norm(l1)] = (row["proto_form"], row["english"])

    # surf_to_seg: norm(surface_token) → segmented_form  (from lang1 interlinear)
    surf_to_seg: dict[str, str] = {}
    for row in corpus:
        if row["language"] != "lang1" or not row.get("segmented"):
            continue
        surf_toks = [_clean(t) for t in row["surface"].lstrip("﻿").split()]
        seg_toks = [t.rstrip(",") for t in row["segmented"].lstrip("﻿").split()]
        for s, g in zip(surf_toks, seg_toks):
            if s:
                surf_to_seg[_norm(s)] = g

    # docs_by_lang_doc: (language, doc_id) → [text_line, ...]
    docs_by: dict[tuple[str, str], list[str]] = {}
    for row in docs:
        key = (row["language"], row["doc_id"])
        docs_by.setdefault(key, []).append(row["text"].lstrip("﻿"))

    return {
        "citation_to_proto": citation_to_proto,
        "surf_to_seg": surf_to_seg,
        "docs_by": docs_by,
    }


# ---------------------------------------------------------------------------
# Token reconstruction
# ---------------------------------------------------------------------------


@dataclass
class TokenResult:
    original: str
    proto: str
    tier: int        # 1=citation, 2=morphological, 3=fallback
    english: str


def reconstruct_token(
    tok: str,
    citation_to_proto: dict[str, tuple[str, str]],
    surf_to_seg: dict[str, str],
) -> TokenResult:
    """Reconstruct one surface token into its proto-language form."""
    clean = _clean(tok)
    n = _norm(clean)

    if not n:
        return TokenResult(tok, tok, 3, "")

    # Tier 1: citation match
    r = citation_to_proto.get(n)
    if r:
        return TokenResult(tok, r[0], 1, r[1])

    # Tier 2: morphological segmentation from interlinear corpus
    seg = surf_to_seg.get(n)
    if seg:
        morphs = seg.split("-")
        proto_parts: list[str] = []
        for m in morphs:
            mn = _norm(m)
            rc = citation_to_proto.get(mn)
            if rc:
                proto_parts.append(rc[0].lstrip("*"))
            else:
                proto_parts.append(PROTO_GRAM.get(mn, m))
        proto_form = "*" + "-".join(proto_parts)
        return TokenResult(tok, proto_form, 2, "")

    # Tier 3: phonological fallback — apply corrections, prefix *
    corrected = _apply_proto_corrections(clean)
    return TokenResult(tok, "*" + corrected, 3, "")


def reconstruct_line(
    line: str,
    citation_to_proto: dict[str, tuple[str, str]],
    surf_to_seg: dict[str, str],
) -> tuple[str, list[TokenResult]]:
    """Reconstruct one text line. Returns (proto_line, token_results)."""
    tokens = line.split()
    results: list[TokenResult] = []

    for tok in tokens:
        result = reconstruct_token(tok, citation_to_proto, surf_to_seg)
        results.append(result)

    # Rebuild the line, preserving trailing punctuation on each token
    proto_tokens: list[str] = []
    for tok, res in zip(tokens, results):
        punct = _trailing_punct(tok)
        proto_tokens.append(res.proto + punct)

    return " ".join(proto_tokens), results


# ---------------------------------------------------------------------------
# Document generation
# ---------------------------------------------------------------------------


def _tier_label(t: int) -> str:
    return {1: "citation", 2: "morphological", 3: "fallback"}[t]


def generate_doc_output(
    doc_id: str,
    docs_by: dict[tuple[str, str], list[str]],
    citation_to_proto: dict[str, tuple[str, str]],
    surf_to_seg: dict[str, str],
) -> str:
    """Generate the full output text for one document."""

    lang1_lines = docs_by.get(("lang1", doc_id), [])
    if not lang1_lines:
        return f"No data for doc{doc_id}.\n"

    # Reconstruct each line
    proto_lines: list[str] = []
    all_results: list[list[TokenResult]] = []
    for line in lang1_lines:
        proto_line, results = reconstruct_line(line, citation_to_proto, surf_to_seg)
        proto_lines.append(proto_line)
        all_results.append(results)

    # Aggregate stats
    tier_counts = Counter(r.tier for results in all_results for r in results)
    total_tokens = sum(tier_counts.values())

    out: list[str] = []

    # ── Header ───────────────────────────────────────────────────────────────
    out += [
        "=" * 70,
        f"  PROTO-LANGUAGE DOCUMENT {doc_id}",
        "=" * 70,
        "",
        "Reconstruction tiers:",
        "  [1] citation     — token matched a dictionary citation form",
        "  [2] morphological — token segmented via interlinear corpus",
        "  [3] fallback     — lang1 form used with *-prefix + á→à correction",
        "",
        f"Token statistics: {tier_counts[1]} citation ({100*tier_counts[1]//total_tokens}%),",
        f"                  {tier_counts[2]} morphological ({100*tier_counts[2]//total_tokens}%),",
        f"                  {tier_counts[3]} fallback ({100*tier_counts[3]//total_tokens}%)",
        "",
        "-" * 70,
        "  PROTO-LANGUAGE TEXT",
        "-" * 70,
        "",
    ]

    for proto_line in proto_lines:
        out.append(proto_line)
    out.append("")

    # ── Line-by-line comparison ───────────────────────────────────────────────
    out += [
        "-" * 70,
        "  LINE-BY-LINE COMPARISON",
        "-" * 70,
        "",
    ]

    for i, (proto_line, lang1_line) in enumerate(zip(proto_lines, lang1_lines), 1):
        out.append(f"Line {i}:")
        out.append(f"  Proto : {proto_line}")
        out.append(f"  Lang1 : {lang1_line}")

        # Show all other languages if available
        for lang in LANGUAGES[1:]:
            other_lines = docs_by.get((lang, doc_id), [])
            if i <= len(other_lines):
                out.append(f"  {lang}  : {other_lines[i-1]}")
        out.append("")

    # ── Token-level annotation ────────────────────────────────────────────────
    out += [
        "-" * 70,
        "  TOKEN ANNOTATIONS",
        "-" * 70,
        "",
        f"  {'Original':<25} {'Proto-form':<28} {'Tier':<15} {'English'}",
        f"  {'-'*24} {'-'*27} {'-'*14} {'-'*20}",
    ]

    seen: set[str] = set()
    for results in all_results:
        for r in results:
            key = _clean(r.original).lower()
            if key in seen:
                continue
            seen.add(key)
            label = _tier_label(r.tier)
            english = r.english if r.english else ""
            out.append(
                f"  {r.original:<25} {r.proto:<28} {label:<15} {english}"
            )

    out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    processed_dir: Path = PROCESSED_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, str]:
    """Reconstruct both documents and write output files.

    Returns {"doc1": text, "doc2": text}.
    """
    resources = load_resources(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    for doc_id in ("1", "2"):
        text = generate_doc_output(
            doc_id,
            resources["docs_by"],
            resources["citation_to_proto"],
            resources["surf_to_seg"],
        )
        outputs[f"doc{doc_id}"] = text
        out_path = output_dir / f"proto_doc{doc_id}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"Written: {out_path}")

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    results = run(processed_dir=args.processed_dir, output_dir=args.output_dir)

    for doc_id in ("1", "2"):
        text = results[f"doc{doc_id}"]
        # Print the proto text section only
        lines = text.splitlines()
        in_proto = False
        print(f"\n{'='*70}")
        print(f"  PROTO DOC{doc_id} — RECONSTRUCTED TEXT")
        print(f"{'='*70}")
        for line in lines:
            if "PROTO-LANGUAGE TEXT" in line:
                in_proto = True
                continue
            if in_proto and "LINE-BY-LINE" in line:
                break
            if in_proto:
                print(line)
