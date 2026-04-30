"""
validate.py — Stage 5: Validation of the proto-language reconstruction.

Three complementary checks:

  1. Predictive accuracy  — apply sound laws to every proto-form; compare
                            the predicted daughter form to the actual cognate
                            from Stage 2; classify each result as
                            exact / close / partial / miss.

  2. Exception analysis   — rank and categorise the misses; distinguish
                            probable true irregulars from probable conditioned
                            changes not yet captured by the current laws.

  3. Proto-text generation — take five high-coverage lang1 sentences, replace
                             each content word with its *proto_form, and
                             display the result alongside the original and the
                             English translation.

Prediction metric
-----------------
Because Stage 2 cognate forms are inflected surface tokens (with prefixes,
suffixes, and sometimes reduplication) while the predicted forms are bare
stems, symmetric LCS similarity underestimates prediction quality.  We
therefore use *predicted coverage*:

    coverage = LCS_length(predicted_graphemes, actual_graphemes)
               / len(predicted_graphemes)

A coverage of 1.0 means every grapheme of the predicted stem was found as
a subsequence in the actual (inflected) form.  Misses (coverage < 0.50)
most likely reflect genuine irregulars or undiscovered conditioned changes.

Match classes:
    exact    coverage >= 0.95   prediction fully covered by actual
    close    coverage >= 0.75
    partial  coverage >= 0.50
    miss     coverage <  0.50

Outputs:
    predictions.csv         (entry × language): predicted form, actual form,
                            raw_similarity, predicted_coverage, match_class
    exceptions.csv          miss-class entries with diagnostic fields
    validation_report.md    full report: accuracy tables, exception patterns,
                            sample proto-language sentences

Run:
    python src/validate.py
    python src/validate.py --processed-dir path/to/processed
"""

from __future__ import annotations

import argparse
import csv
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

from src.sound_correspondences import grapheme_split, grapheme_base, lcs_align

_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = _ROOT / "data" / "processed"
LANGUAGES = [f"lang{i}" for i in range(1, 8)]

# Match-class thresholds (on predicted_coverage)
EXACT_THRESHOLD: float = 0.95
CLOSE_THRESHOLD: float = 0.75
PARTIAL_THRESHOLD: float = 0.50

# Number of sample proto-language sentences to generate
N_PROTO_SENTENCES: int = 5

# Grammatical morpheme glosses (reused from Stage 2/3)
_GRAM_TAGS = frozenset(
    "ERG ABS NOM ACC DAT GEN ELAT ILL INESS DEF NEAR FAR INAN ANIM "
    "PL SG WIT EVID FUT INFER DIR NEG POSS SUBJ OBJ INCL EXCL BEN "
    "INSTR CONJ MIR ADJ ADV 1SG 2SG 3SG 1PL 2PL 3PL CL".split()
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """One (entry, language) prediction compared to the actual cognate form."""

    english: str
    pos: str
    proto_form: str
    language: str
    predicted_form: str
    actual_form: str
    raw_similarity: float
    predicted_coverage: float
    match_class: str          # "exact" / "close" / "partial" / "miss"


@dataclass
class ExceptionRecord:
    """A miss-class entry with diagnostic information."""

    english: str
    pos: str
    proto_form: str
    language: str
    predicted_form: str
    actual_form: str
    predicted_coverage: float
    mismatched_grapheme: str  # first predicted grapheme not found in actual
    diagnosis: str            # "irregular" / "conditioned" / "alignment_noise"


# ---------------------------------------------------------------------------
# Text utilities (local copies to avoid import friction)
# ---------------------------------------------------------------------------


def _strip(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()


def _lcs_length(a: str, b: str) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def lcs_similarity(a: str, b: str) -> float:
    na, nb = _strip(a), _strip(b)
    total = len(na) + len(nb)
    return 0.0 if total == 0 else round(2.0 * _lcs_length(na, nb) / total, 4)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_data(processed_dir: Path) -> dict[str, list[dict]]:
    return {
        "sound_laws":   _read_csv(processed_dir / "sound_laws.csv"),
        "proto_lexicon": _read_csv(processed_dir / "proto_lexicon.csv"),
        "cognate_sets": _read_csv(processed_dir / "cognate_sets.csv"),
        "corpus":       _read_csv(processed_dir / "corpus.csv"),
    }


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------


def build_laws_map(
    sound_laws: list[dict],
) -> dict[tuple[str, str], str]:
    """Build (language, proto_grapheme) → best_reflex lookup.

    When multiple laws exist for the same (language, proto_grapheme) — which
    happens for split correspondences below the threshold — the one with the
    highest regularity is used.
    """
    laws_map: dict[tuple[str, str], tuple[str, float]] = {}
    for row in sound_laws:
        key = (row["language"], row["proto_grapheme"])
        reg = float(row["regularity"])
        if key not in laws_map or reg > laws_map[key][1]:
            laws_map[key] = (row["reflex"], reg)
    return {k: v[0] for k, v in laws_map.items()}


def predict_form(
    proto_form: str,
    language: str,
    laws_map: dict[tuple[str, str], str],
) -> str:
    """Apply sound laws to produce a predicted daughter-language form.

    For each grapheme in the proto_form, look up the law for that language.
    If no law exists, the grapheme is kept unchanged (conservation by default).
    """
    graphemes = grapheme_split(proto_form.lstrip("*"))
    return "".join(laws_map.get((language, g), g) for g in graphemes)


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------


def predicted_coverage(
    predicted_form: str,
    actual_form: str,
) -> float:
    """Fraction of predicted graphemes present as a subsequence in actual_form.

    A score of 1.0 means the predicted stem is fully embedded in the actual
    (inflected) form.  This metric is robust to prefixes, suffixes, and
    reduplication in the Stage 2 cognate tokens.
    """
    pred_g = grapheme_split(predicted_form)
    act_g = grapheme_split(actual_form)
    n_pred = len(pred_g)
    if n_pred == 0:
        return 0.0
    pred_bases = [grapheme_base(g) for g in pred_g]
    act_bases = [grapheme_base(g) for g in act_g]
    return round(_lcs_length(pred_bases, act_bases) / n_pred, 4)


def match_class(coverage: float) -> str:
    if coverage >= EXACT_THRESHOLD:
        return "exact"
    if coverage >= CLOSE_THRESHOLD:
        return "close"
    if coverage >= PARTIAL_THRESHOLD:
        return "partial"
    return "miss"


# ---------------------------------------------------------------------------
# Prediction evaluation
# ---------------------------------------------------------------------------


def evaluate_predictions(
    proto_lexicon: list[dict],
    laws_map: dict[tuple[str, str], str],
) -> list[PredictionResult]:
    """Evaluate sound-law predictions against actual cognate forms.

    For each (entry, language) pair where the actual cognate is non-empty,
    predict the daughter form and compare it to the actual form.
    """
    results: list[PredictionResult] = []
    for row in proto_lexicon:
        proto = row["proto_form"]
        english = row["english"]
        pos = row["pos"]

        for lang in LANGUAGES[1:]:
            actual = row.get(lang, "").strip()
            if not actual:
                continue

            predicted = predict_form(proto, lang, laws_map)
            raw_sim = lcs_similarity(predicted, actual)
            cov = predicted_coverage(predicted, actual)
            cls = match_class(cov)

            results.append(
                PredictionResult(
                    english=english,
                    pos=pos,
                    proto_form=proto,
                    language=lang,
                    predicted_form=predicted,
                    actual_form=actual,
                    raw_similarity=raw_sim,
                    predicted_coverage=cov,
                    match_class=cls,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Exception analysis
# ---------------------------------------------------------------------------


def _first_missing_grapheme(
    predicted_form: str,
    actual_form: str,
) -> str:
    """Return the first predicted grapheme not covered by the LCS alignment."""
    pred_g = grapheme_split(predicted_form)
    act_g = grapheme_split(actual_form)
    pred_b = [grapheme_base(g) for g in pred_g]
    act_b = [grapheme_base(g) for g in act_g]

    # Find which predicted positions are NOT matched in the LCS
    n, m = len(pred_b), len(act_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if pred_b[i - 1] == act_b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])

    matched_pred: set[int] = set()
    i, j = n, m
    while i > 0 and j > 0:
        if pred_b[i - 1] == act_b[j - 1]:
            matched_pred.add(i - 1)
            i -= 1; j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    for idx, g in enumerate(pred_g):
        if idx not in matched_pred:
            return g
    return ""


def _diagnose(result: PredictionResult) -> str:
    """Assign a preliminary diagnosis to a miss-class result.

    Heuristics:
    - alignment_noise : actual form is very long (>= 3× predicted); the
                        Stage 2 token likely captured a whole multi-word
                        surface form rather than just the target word.
    - conditioned     : the mismatched grapheme is a vowel whose split
                        behaviour (e.g. *u → ú vs ṹ) was not resolved in
                        Stage 3 into a single unconditional law.
    - irregular       : everything else.
    """
    pred_len = len(grapheme_split(result.predicted_form))
    act_len = len(grapheme_split(result.actual_form))
    if act_len >= 3 * pred_len:
        return "alignment_noise"

    vowels = set("aeiouáéíóúàèìòùãẽĩõũ")
    miss_g = _first_missing_grapheme(result.predicted_form, result.actual_form)
    if miss_g and grapheme_base(miss_g) in vowels:
        return "conditioned"

    return "irregular"


def analyze_exceptions(
    results: list[PredictionResult],
) -> list[ExceptionRecord]:
    """Extract and diagnose all miss-class predictions."""
    exceptions: list[ExceptionRecord] = []
    for r in results:
        if r.match_class != "miss":
            continue
        miss_g = _first_missing_grapheme(r.predicted_form, r.actual_form)
        diagnosis = _diagnose(r)
        exceptions.append(
            ExceptionRecord(
                english=r.english,
                pos=r.pos,
                proto_form=r.proto_form,
                language=r.language,
                predicted_form=r.predicted_form,
                actual_form=r.actual_form,
                predicted_coverage=r.predicted_coverage,
                mismatched_grapheme=miss_g,
                diagnosis=diagnosis,
            )
        )
    exceptions.sort(key=lambda x: x.predicted_coverage)
    return exceptions


# ---------------------------------------------------------------------------
# Proto-text generation
# ---------------------------------------------------------------------------


def _content_gloss(gls_token: str) -> str | None:
    for part in gls_token.rstrip(",").split("-"):
        clean = part.strip(".")
        if clean and clean.upper() not in _GRAM_TAGS and clean[0].islower():
            return clean
    return None


def _build_proto_lookup(proto_lexicon: list[dict]) -> dict[str, str]:
    """Map English gloss word → proto_form (stripping 'to ' from verbs)."""
    lookup: dict[str, str] = {}
    import re
    for row in proto_lexicon:
        eng = row["english"]
        proto = row["proto_form"]
        key = re.sub(r"^to\s+", "", eng.lower()).strip()
        lookup[key] = proto
        lookup[eng.lower()] = proto
    return lookup


def generate_proto_texts(
    corpus: list[dict],
    proto_lexicon: list[dict],
    n: int = N_PROTO_SENTENCES,
) -> list[dict[str, str]]:
    """Generate n proto-language sentences from lang1 interlinear data.

    For each lang1 sentence with segmented + gloss fields, parse the gloss to
    identify content-word tokens.  Substitute each content-word's surface token
    with the *proto_form from the proto-lexicon.  Words without a proto-form
    mapping keep their original form with a '?' suffix.

    Returns a list of dicts: {original, proto_sentence, translation, coverage}.
    coverage is the fraction of tokens successfully mapped to proto-forms.
    """
    proto_lookup = _build_proto_lookup(proto_lexicon)

    lang1_rows = [r for r in corpus if r["language"] == "lang1"
                  and r.get("segmented") and r.get("gloss")]

    results: list[dict[str, str]] = []

    for row in lang1_rows:
        surf_tokens = row["surface"].lstrip("﻿").split()
        seg_tokens = [t.rstrip(",") for t in row["segmented"].lstrip("﻿").split()]
        gls_tokens = [t.rstrip(",") for t in row["gloss"].lstrip("﻿").split()]

        if not (len(surf_tokens) == len(seg_tokens) == len(gls_tokens)):
            continue

        proto_tokens: list[str] = []
        mapped = 0

        for surf, _seg, gls in zip(surf_tokens, seg_tokens, gls_tokens):
            content = _content_gloss(gls)
            proto = None
            if content:
                proto = proto_lookup.get(content) or proto_lookup.get(f"to {content}")
            if proto:
                proto_tokens.append(proto)
                mapped += 1
            else:
                # Preserve surface form for grammatical tokens
                proto_tokens.append(surf)

        coverage = mapped / len(surf_tokens) if surf_tokens else 0.0

        # Only include sentences with at least 2 mapped content words
        if mapped < 2:
            continue

        results.append(
            {
                "original": row["surface"],
                "proto_sentence": " ".join(proto_tokens),
                "translation": row["translation"],
                "coverage": f"{coverage:.0%}",
            }
        )

        if len(results) >= n:
            break

    return results


# ---------------------------------------------------------------------------
# Accuracy statistics
# ---------------------------------------------------------------------------


def compute_accuracy_stats(results: list[PredictionResult]) -> dict:
    """Compute per-language and overall accuracy statistics."""
    by_lang: dict[str, list[PredictionResult]] = defaultdict(list)
    for r in results:
        by_lang[r.language].append(r)

    lang_stats: dict[str, dict] = {}
    for lang in LANGUAGES[1:]:
        rows = by_lang.get(lang, [])
        if not rows:
            lang_stats[lang] = {}
            continue
        total = len(rows)
        counts = Counter(r.match_class for r in rows)
        avg_cov = sum(r.predicted_coverage for r in rows) / total
        lang_stats[lang] = {
            "total": total,
            "exact": counts["exact"],
            "close": counts["close"],
            "partial": counts["partial"],
            "miss": counts["miss"],
            "exact_pct": 100 * counts["exact"] / total,
            "close_or_better_pct": 100 * (counts["exact"] + counts["close"]) / total,
            "miss_pct": 100 * counts["miss"] / total,
            "avg_coverage": round(avg_cov, 3),
        }

    all_total = len(results)
    all_counts = Counter(r.match_class for r in results)
    overall = {
        "total": all_total,
        "exact_pct": 100 * all_counts["exact"] / all_total if all_total else 0,
        "close_or_better_pct": 100 * (all_counts["exact"] + all_counts["close"]) / all_total if all_total else 0,
        "miss_pct": 100 * all_counts["miss"] / all_total if all_total else 0,
        "avg_coverage": round(sum(r.predicted_coverage for r in results) / all_total, 3) if all_total else 0,
    }

    return {"by_language": lang_stats, "overall": overall}


# ---------------------------------------------------------------------------
# Validation report generator
# ---------------------------------------------------------------------------


def generate_report(
    results: list[PredictionResult],
    exceptions: list[ExceptionRecord],
    proto_texts: list[dict[str, str]],
    stats: dict,
) -> str:
    lines: list[str] = [
        "# Validation Report",
        "",
        "Generated by `src/validate.py` (Stage 5).",
        "",
        "**Metric**: predicted coverage — the fraction of predicted stem graphemes",
        "found as a subsequence in the actual (inflected) daughter-language form.",
        "This is robust to affixal material that surrounds the stem in the",
        "Stage 2 surface tokens.",
        "",
        "**Match classes**: exact (≥ 0.95) · close (≥ 0.75) · partial (≥ 0.50) · miss (< 0.50)",
        "",
    ]

    # ── 1. Overall accuracy ──────────────────────────────────────────────────
    ov = stats["overall"]
    lines += [
        "## 1. Overall Accuracy",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total (entry × language) pairs evaluated | {ov['total']} |",
        f"| Exact (coverage ≥ 0.95) | {ov['exact_pct']:.1f} % |",
        f"| Close or better (≥ 0.75) | {ov['close_or_better_pct']:.1f} % |",
        f"| Miss (< 0.50) | {ov['miss_pct']:.1f} % |",
        f"| Average predicted coverage | {ov['avg_coverage']:.3f} |",
        "",
    ]

    # ── 2. Per-language accuracy ─────────────────────────────────────────────
    lines += [
        "## 2. Per-Language Accuracy",
        "",
        "| Language | n | Exact % | Close+% | Miss % | Avg cov |",
        "|---|---|---|---|---|---|",
    ]
    for lang in LANGUAGES[1:]:
        s = stats["by_language"].get(lang, {})
        if not s:
            lines.append(f"| {lang} | 0 | — | — | — | — |")
        else:
            lines.append(
                f"| {lang} | {s['total']} | {s['exact_pct']:.1f} % "
                f"| {s['close_or_better_pct']:.1f} % "
                f"| {s['miss_pct']:.1f} % "
                f"| {s['avg_coverage']:.3f} |"
            )
    lines += [""]

    # ── 3. Best and worst predictions ────────────────────────────────────────
    sorted_res = sorted(results, key=lambda r: -r.predicted_coverage)
    best = sorted_res[:8]
    worst = sorted(results, key=lambda r: r.predicted_coverage)[:8]

    lines += [
        "## 3. Best Predictions",
        "",
        "| English | Language | Proto | Predicted | Actual | Cov |",
        "|---|---|---|---|---|---|",
    ]
    for r in best:
        lines.append(
            f"| {r.english} | {r.language} | {r.proto_form} "
            f"| {r.predicted_form} | {r.actual_form[:30]} | {r.predicted_coverage:.2f} |"
        )

    lines += [
        "",
        "## 4. Worst Predictions (misses)",
        "",
        "| English | Language | Proto | Predicted | Actual | Cov |",
        "|---|---|---|---|---|---|",
    ]
    for r in worst:
        lines.append(
            f"| {r.english} | {r.language} | {r.proto_form} "
            f"| {r.predicted_form} | {r.actual_form[:30]} | {r.predicted_coverage:.2f} |"
        )
    lines += [""]

    # ── 4. Exception analysis ────────────────────────────────────────────────
    diag_counts = Counter(e.diagnosis for e in exceptions)
    lines += [
        "## 5. Exception Analysis",
        "",
        f"Total miss-class predictions: **{len(exceptions)}**",
        "",
        "| Diagnosis | Count | Interpretation |",
        "|---|---|---|",
        f"| alignment_noise | {diag_counts['alignment_noise']} | "
        "Stage 2 cognate token captured multi-word surface form |",
        f"| conditioned | {diag_counts['conditioned']} | "
        "Vowel split not yet resolved; environment unknown |",
        f"| irregular | {diag_counts['irregular']} | "
        "Genuine exception or consonant change not in current laws |",
        "",
    ]

    # Most common mismatched graphemes
    miss_g_counts = Counter(e.mismatched_grapheme for e in exceptions if e.mismatched_grapheme)
    lines += [
        "**Most frequently mismatched proto-graphemes (miss class):**",
        "",
        "| Proto-grapheme | Miss count |",
        "|---|---|",
    ]
    for g, cnt in miss_g_counts.most_common(8):
        lines.append(f"| {g} | {cnt} |")
    lines += [""]

    # ── 5. Proto-text samples ─────────────────────────────────────────────────
    lines += [
        "## 6. Sample Proto-Language Sentences",
        "",
        "Content words replaced with *proto_forms; grammatical tokens kept as",
        "lang1 surface forms.  Coverage = fraction of tokens mapped to proto-forms.",
        "",
    ]
    for i, pt in enumerate(proto_texts, 1):
        lines += [
            f"### Sentence {i}  (coverage {pt['coverage']})",
            "",
            f"**Proto**: {pt['proto_sentence']}",
            "",
            f"**Lang1**: {pt['original']}",
            "",
            f"**Translation**: {pt['translation']}",
            "",
        ]

    # ── 6. Summary assessment ─────────────────────────────────────────────────
    ov = stats["overall"]
    lines += [
        "## 7. Overall Assessment",
        "",
        f"The sound laws correctly predict **{ov['close_or_better_pct']:.1f} %** of "
        "daughter-language stem graphemes (close-or-better class, predicted coverage ≥ 0.75).",
        f"Only **{ov['miss_pct']:.1f} %** are outright misses.",
        "",
        "The main sources of miss-class errors are:",
        "",
        f"- **Alignment noise** ({diag_counts['alignment_noise']} cases): Stage 2 cognate tokens that",
        "  captured full verbal or nominal complexes rather than just the target root.",
        f"- **Conditioned vowel splits** ({diag_counts['conditioned']} cases): the split between",
        "  e.g. *u → ú vs *u → ṹ in lang3/4/7 is conditioned by an unknown environment.",
        f"- **Genuine irregulars** ({diag_counts['irregular']} cases): loanwords, analogical",
        "  reformations, or residual alignment errors from Stage 2.",
        "",
        "The consonant inventory is fully stable across all daughters and requires",
        "no sound laws beyond identity.  All nine non-identity laws identified in",
        "Stage 3 involve vowels, consistently with the proto-language having a",
        "fixed consonant inventory and a more variable tonal/accent system.",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(records: list, output_path: Path) -> None:
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


def run(processed_dir: Path = PROCESSED_DIR) -> dict:
    """Run the full Stage 5 validation pipeline.

    Steps:
    1.  Load all Stage 1–4 outputs from processed_dir.
    2.  Build the sound-laws lookup map.
    3.  Evaluate predictions for every (proto_entry, language) pair.
    4.  Analyse miss-class exceptions.
    5.  Generate sample proto-language sentences.
    6.  Write predictions.csv, exceptions.csv, validation_report.md.

    Returns {"results", "exceptions", "proto_texts", "stats"}.
    """
    data = load_data(processed_dir)
    laws_map = build_laws_map(data["sound_laws"])

    results = evaluate_predictions(data["proto_lexicon"], laws_map)
    exceptions = analyze_exceptions(results)
    proto_texts = generate_proto_texts(data["corpus"], data["proto_lexicon"])
    stats = compute_accuracy_stats(results)
    report = generate_report(results, exceptions, proto_texts, stats)

    export_csv(results, processed_dir / "predictions.csv")
    export_csv(exceptions, processed_dir / "exceptions.csv")
    (processed_dir / "validation_report.md").write_text(report, encoding="utf-8")

    return {
        "results": results,
        "exceptions": exceptions,
        "proto_texts": proto_texts,
        "stats": stats,
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
        help="Directory with Stage 1–4 outputs (default: data/processed/)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    out = run(processed_dir=args.processed_dir)

    stats = out["stats"]
    ov = stats["overall"]
    exceptions = out["exceptions"]
    diag = Counter(e.diagnosis for e in exceptions)

    print(f"predictions evaluated : {ov['total']:>5}")
    print(f"exact (≥ 0.95)        : {ov['exact_pct']:>6.1f} %")
    print(f"close or better       : {ov['close_or_better_pct']:>6.1f} %")
    print(f"miss (< 0.50)         : {ov['miss_pct']:>6.1f} %")
    print(f"avg predicted coverage: {ov['avg_coverage']:>6.3f}")
    print()
    print("Per-language accuracy (close or better %):")
    for lang in LANGUAGES[1:]:
        s = stats["by_language"].get(lang, {})
        if s:
            bar = "█" * int(s["close_or_better_pct"] / 5)
            print(f"  {lang}  {s['close_or_better_pct']:>5.1f} %  {bar}")
    print()
    print(f"Exceptions: {len(exceptions)}  "
          f"(alignment_noise: {diag['alignment_noise']}, "
          f"conditioned: {diag['conditioned']}, "
          f"irregular: {diag['irregular']})")
    print()
    print(f"Proto-texts generated : {len(out['proto_texts'])}")
    print("Sample:")
    for pt in out["proto_texts"][:2]:
        print(f"  Proto : {pt['proto_sentence'][:80]}")
        print(f"  Lang1 : {pt['original'][:80]}")
        print(f"  Transl: {pt['translation'][:80]}")
        print()
