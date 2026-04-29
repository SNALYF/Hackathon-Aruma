"""
Tests for src/sound_correspondences.py.

Unit tests use inline fixtures and tmp_path so they do not depend on real data.
Integration tests are marked @pytest.mark.integration and skipped when the
processed data directory is absent.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from src.sound_correspondences import (
    LANGUAGES,
    MAX_EXAMPLES,
    MIN_COGNATES,
    MIN_FORM_GRAPHEMES,
    PROCESSED_DIR,
    Correspondence,
    TableRow,
    build_correspondence_records,
    build_table_rows,
    compute_regularity,
    compute_summary,
    export_csv,
    extract_correspondences,
    grapheme_base,
    grapheme_split,
    lcs_align,
    run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROCESSED_DATA_AVAILABLE = (PROCESSED_DIR / "cognate_sets.csv").exists()
integration = pytest.mark.skipif(
    not PROCESSED_DATA_AVAILABLE, reason="processed data directory not found"
)


def write_cognate_sets(path: Path, rows: list[dict]) -> None:
    """Write cognate_sets.csv to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["english", "pos", "lang1", "lang2", "lang3", "lang4", "lang5", "lang6", "lang7"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full = {f: row.get(f, "") for f in fieldnames}
            writer.writerow(full)


def make_counts(lang: str, pairs: dict[tuple[str, str], int]) -> dict[str, Counter]:
    """Build a counts dict for one language."""
    return {lang: Counter(pairs)}


# ---------------------------------------------------------------------------
# grapheme_split
# ---------------------------------------------------------------------------


class TestGraphemeSplit:
    def test_plain_ascii(self):
        assert grapheme_split("abc") == ["a", "b", "c"]

    def test_empty_string(self):
        assert grapheme_split("") == []

    def test_precomposed_accented(self):
        # "dóruma" — ó is a single precomposed grapheme
        result = grapheme_split("dóruma")
        assert result == ["d", "ó", "r", "u", "m", "a"]

    def test_ipa_thorn_preserved(self):
        result = grapheme_split("þ")
        assert result == ["þ"]

    def test_combining_mark_attached_to_base(self):
        # NFD: a + grave → one grapheme cluster
        nfd_a_grave = "a\u0300"  # a + combining grave
        result = grapheme_split(nfd_a_grave)
        assert len(result) == 1
        # result should be NFC
        assert result[0] == unicodedata.normalize("NFC", nfd_a_grave)

    def test_multiple_combining_marks_one_grapheme(self):
        # a + tilde + grave → single grapheme
        s = "a\u0303\u0300"
        result = grapheme_split(s)
        assert len(result) == 1

    def test_result_is_nfc(self):
        # NFD input → NFC graphemes
        nfd = unicodedata.normalize("NFD", "dóruma")
        result = grapheme_split(nfd)
        for g in result:
            assert g == unicodedata.normalize("NFC", g)

    def test_mixed_plain_and_accented(self):
        result = grapheme_split("þàdórúmà")
        assert len(result) == 8

    def test_leading_stray_combining_skipped(self):
        # stray combining mark at start, then base
        s = "\u0300a"  # grave + a
        result = grapheme_split(s)
        # stray grave skipped, "a" is the only grapheme
        assert result == ["a"]

    def test_length_equals_visual_character_count(self):
        # "dóruma" has 6 visual characters
        assert len(grapheme_split("dóruma")) == 6


# ---------------------------------------------------------------------------
# grapheme_base
# ---------------------------------------------------------------------------


class TestGraphemeBase:
    def test_plain_letter(self):
        assert grapheme_base("a") == "a"

    def test_accented_vowel(self):
        assert grapheme_base("ó") == "o"

    def test_grave_accent(self):
        assert grapheme_base("à") == "a"

    def test_thorn_preserved(self):
        # þ has no combining mark — returned as-is
        assert grapheme_base("þ") == "þ"

    def test_returns_lowercase(self):
        assert grapheme_base("Á") == "a"

    def test_multiple_combining_marks(self):
        # ã̀: a + tilde + grave → base is "a"
        g = unicodedata.normalize("NFC", "a\u0303\u0300")
        assert grapheme_base(g) == "a"

    def test_plain_uppercase(self):
        assert grapheme_base("D") == "d"


# ---------------------------------------------------------------------------
# lcs_align
# ---------------------------------------------------------------------------


class TestLcsAlign:
    def test_empty_a(self):
        assert lcs_align([], ["a", "b"]) == []

    def test_empty_b(self):
        assert lcs_align(["a", "b"], []) == []

    def test_both_empty(self):
        assert lcs_align([], []) == []

    def test_identical_sequences(self):
        a = ["d", "o", "r", "u", "m", "a"]
        b = ["d", "o", "r", "u", "m", "a"]
        pairs = lcs_align(a, b)
        assert pairs == list(zip(a, b))

    def test_completely_different_bases(self):
        # No common bases → empty alignment
        a = grapheme_split("bbb")
        b = grapheme_split("zzz")
        assert lcs_align(a, b) == []

    def test_diacritics_ignored_in_matching(self):
        # "ó" in a, "o" in b — both have base "o" → should align
        a = grapheme_split("dóruma")
        b = grapheme_split("doruma")
        pairs = lcs_align(a, b)
        assert len(pairs) == 6
        # "ó" from a aligned with "o" from b
        assert ("ó", "o") in pairs

    def test_subsequence_alignment(self):
        # a = ["a", "b", "c", "d"], b = ["a", "c", "d"]
        # LCS = ["a","c","d"]
        a = ["a", "b", "c", "d"]
        b = ["a", "c", "d"]
        pairs = lcs_align(a, b)
        assert ("a", "a") in pairs
        assert ("c", "c") in pairs
        assert ("d", "d") in pairs
        assert len(pairs) == 3

    def test_original_graphemes_preserved_in_output(self):
        # Even though bases are used for matching, the output contains original graphemes
        a = grapheme_split("dóruma")
        b = grapheme_split("þàdórúmàgùùg")
        pairs = lcs_align(a, b)
        a_in_pairs = {p[0] for p in pairs}
        b_in_pairs = {p[1] for p in pairs}
        # All returned graphemes should be from the originals
        assert a_in_pairs.issubset(set(a))
        assert b_in_pairs.issubset(set(b))

    def test_insertion_and_deletion_dropped(self):
        # Unmatched graphemes are silently dropped
        a = ["a", "x", "b"]
        b = ["a", "b"]
        pairs = lcs_align(a, b)
        # LCS = ["a","b"], "x" in a is dropped
        assert ("a", "a") in pairs
        assert ("b", "b") in pairs
        assert len(pairs) == 2

    def test_pairs_in_order(self):
        a = grapheme_split("doruma")
        b = grapheme_split("doruma")
        pairs = lcs_align(a, b)
        a_indices = [a.index(p[0]) for p in pairs]
        assert a_indices == sorted(a_indices)


# ---------------------------------------------------------------------------
# extract_correspondences
# ---------------------------------------------------------------------------


class TestExtractCorrespondences:
    def _row(self, english="student", lang1="dóruma", **kwargs):
        base = {"english": english, "pos": "NN", "lang1": lang1,
                "lang2": "", "lang3": "", "lang4": "", "lang5": "", "lang6": "", "lang7": ""}
        base.update(kwargs)
        return base

    def test_empty_input(self):
        counts, examples = extract_correspondences([])
        assert counts == {}
        assert examples == {}

    def test_empty_lang1_skipped(self):
        row = self._row(lang1="", lang2="doruma")
        counts, _ = extract_correspondences([row])
        assert counts == {}

    def test_short_lang1_skipped(self):
        # fewer than MIN_FORM_GRAPHEMES graphemes
        row = self._row(lang1="ab", lang2="ab")
        counts, _ = extract_correspondences([row])
        assert counts == {}

    def test_empty_langN_skipped(self):
        row = self._row(lang1="dóruma", lang2="")
        counts, _ = extract_correspondences([row])
        assert "lang2" not in counts

    def test_short_langN_skipped(self):
        row = self._row(lang1="dóruma", lang2="ab")
        counts, _ = extract_correspondences([row])
        assert "lang2" not in counts

    def test_basic_correspondence_counted(self):
        row = self._row(lang1="doruma", lang2="doruma")
        counts, _ = extract_correspondences([row])
        assert "lang2" in counts
        # identity pairs: d-d, o-o, r-r, u-u, m-m, a-a
        assert counts["lang2"][("d", "d")] == 1

    def test_diacritic_preserved_in_correspondence(self):
        row = self._row(lang1="dóruma", lang2="doruma")
        counts, _ = extract_correspondences([row])
        # "ó" in lang1 aligns with "o" in lang2
        assert counts["lang2"][("ó", "o")] == 1

    def test_counts_accumulate_across_rows(self):
        rows = [
            self._row(english="student", lang1="doruma", lang2="doruma"),
            self._row(english="teacher", lang1="talimi", lang2="talimi"),
        ]
        counts, _ = extract_correspondences(rows)
        # "a" appears in both; expect count >= 2
        assert counts["lang2"][("a", "a")] >= 2

    def test_examples_collected(self):
        row = self._row(english="student", lang1="doruma", lang2="doruma")
        _, examples = extract_correspondences([row])
        assert "student" in examples["lang2"][("d", "d")]

    def test_examples_capped_at_max(self):
        rows = [
            self._row(english=f"word{i}", lang1="doruma", lang2="doruma")
            for i in range(MAX_EXAMPLES + 5)
        ]
        _, examples = extract_correspondences(rows)
        for key, ex_list in examples["lang2"].items():
            assert len(ex_list) <= MAX_EXAMPLES

    def test_multiple_languages_populated(self):
        row = self._row(lang1="doruma", lang2="doruma", lang3="doruma")
        counts, _ = extract_correspondences([row])
        assert "lang2" in counts
        assert "lang3" in counts

    def test_lang1_not_in_output_languages(self):
        row = self._row(lang1="doruma", lang2="doruma")
        counts, _ = extract_correspondences([row])
        assert "lang1" not in counts


# ---------------------------------------------------------------------------
# compute_regularity
# ---------------------------------------------------------------------------


class TestComputeRegularity:
    def test_single_mapping_regularity_one(self):
        counts = {"lang2": Counter({("a", "à"): 5})}
        reg = compute_regularity(counts)
        assert reg["lang2"][("a", "à")] == 1.0

    def test_two_competing_mappings_sum_to_one(self):
        counts = {"lang2": Counter({("a", "à"): 3, ("a", "a"): 1})}
        reg = compute_regularity(counts)
        total = reg["lang2"][("a", "à")] + reg["lang2"][("a", "a")]
        assert abs(total - 1.0) < 1e-9

    def test_regularity_proportion(self):
        counts = {"lang2": Counter({("o", "ó"): 6, ("o", "o"): 2})}
        reg = compute_regularity(counts)
        assert abs(reg["lang2"][("o", "ó")] - 6 / 8) < 1e-9
        assert abs(reg["lang2"][("o", "o")] - 2 / 8) < 1e-9

    def test_empty_counts(self):
        assert compute_regularity({}) == {}

    def test_multiple_lang1_graphemes_independent(self):
        counts = {"lang2": Counter({("a", "à"): 4, ("o", "ó"): 2})}
        reg = compute_regularity(counts)
        # "a" has only one mapping → regularity 1.0
        assert reg["lang2"][("a", "à")] == 1.0
        # "o" has only one mapping → regularity 1.0
        assert reg["lang2"][("o", "ó")] == 1.0

    def test_multiple_languages_independent(self):
        counts = {
            "lang2": Counter({("a", "à"): 3}),
            "lang3": Counter({("a", "ā"): 2}),
        }
        reg = compute_regularity(counts)
        assert "lang2" in reg
        assert "lang3" in reg
        assert reg["lang2"][("a", "à")] == 1.0
        assert reg["lang3"][("a", "ā")] == 1.0


# ---------------------------------------------------------------------------
# build_correspondence_records
# ---------------------------------------------------------------------------


class TestBuildCorrespondenceRecords:
    def _setup(self, lang="lang2", pairs=None, min_count=1):
        if pairs is None:
            pairs = {("a", "à"): 3, ("o", "ó"): 1}
        counts = {lang: Counter(pairs)}
        reg = compute_regularity(counts)
        examples = {lang: {k: ["student"] for k in pairs}}
        return build_correspondence_records(counts, reg, examples, min_count)

    def test_returns_correspondence_instances(self):
        records = self._setup()
        assert all(isinstance(r, Correspondence) for r in records)

    def test_count_below_min_excluded(self):
        records = self._setup(pairs={("a", "à"): 3, ("o", "ó"): 1}, min_count=2)
        counts = {r.count for r in records}
        assert 1 not in counts

    def test_count_at_min_included(self):
        records = self._setup(pairs={("a", "à"): 2}, min_count=2)
        assert len(records) == 1

    def test_is_identity_true_for_same_grapheme(self):
        records = self._setup(pairs={("a", "a"): 3})
        assert records[0].is_identity is True

    def test_is_identity_false_for_different_grapheme(self):
        records = self._setup(pairs={("a", "à"): 3})
        assert records[0].is_identity is False

    def test_regularity_rounded_to_3_decimals(self):
        pairs = {("a", "à"): 2, ("a", "a"): 1}
        records = self._setup(pairs=pairs)
        for r in records:
            assert len(str(r.regularity).split(".")[-1]) <= 3

    def test_examples_joined_with_comma_space(self):
        counts = {"lang2": Counter({("a", "à"): 3})}
        reg = compute_regularity(counts)
        examples = {"lang2": {("a", "à"): ["word1", "word2"]}}
        records = build_correspondence_records(counts, reg, examples, min_count=1)
        assert records[0].examples == "word1, word2"

    def test_sorted_by_count_descending(self):
        pairs = {("a", "à"): 10, ("o", "ó"): 5, ("u", "ú"): 2}
        records = self._setup(pairs=pairs, min_count=1)
        counts_in_order = [r.count for r in records]
        assert counts_in_order == sorted(counts_in_order, reverse=True)

    def test_language_field_correct(self):
        records = self._setup(lang="lang5", pairs={("a", "à"): 3})
        assert records[0].language == "lang5"

    def test_empty_counts_returns_empty_list(self):
        records = build_correspondence_records({}, {}, {})
        assert records == []

    def test_records_ordered_by_language_first(self):
        counts = {
            "lang3": Counter({("a", "à"): 5}),
            "lang2": Counter({("o", "ó"): 3}),
        }
        reg = compute_regularity(counts)
        examples = {lang: {k: [] for k in c} for lang, c in counts.items()}
        records = build_correspondence_records(counts, reg, examples, min_count=1)
        langs = [r.language for r in records]
        # lang2 should come before lang3
        assert langs.index("lang2") < langs.index("lang3")


# ---------------------------------------------------------------------------
# build_table_rows
# ---------------------------------------------------------------------------


class TestBuildTableRows:
    def test_returns_table_row_instances(self):
        counts = {"lang2": Counter({("a", "à"): 3})}
        rows = build_table_rows(counts, min_count=1)
        assert all(isinstance(r, TableRow) for r in rows)

    def test_empty_counts_returns_empty(self):
        assert build_table_rows({}) == []

    def test_below_min_count_excluded(self):
        counts = {"lang2": Counter({("a", "à"): 1})}
        rows = build_table_rows(counts, min_count=2)
        assert rows == []

    def test_cell_format_grapheme_colon_count(self):
        counts = {"lang2": Counter({("a", "à"): 5})}
        rows = build_table_rows(counts, min_count=1)
        assert rows[0].lang2 == "à:5"

    def test_top_3_mappings_per_cell(self):
        # 5 different mappings for "a" in lang2
        pairs = {("a", f"variant{i}"): 10 - i for i in range(5)}
        counts = {"lang2": Counter(pairs)}
        rows = build_table_rows(counts, min_count=1)
        # Only top 3 should appear
        assert rows[0].lang2.count(":") == 3

    def test_mappings_ordered_by_count_desc(self):
        pairs = {("a", "x"): 1, ("a", "y"): 3, ("a", "z"): 2}
        counts = {"lang2": Counter(pairs)}
        rows = build_table_rows(counts, min_count=1)
        cell = rows[0].lang2
        # "y:3" should appear before "z:2" before "x:1"
        assert cell.index("y:3") < cell.index("z:2") < cell.index("x:1")

    def test_empty_cell_when_no_qualifying_mapping(self):
        # lang3 has no mappings for the grapheme
        counts = {"lang2": Counter({("a", "à"): 3})}
        rows = build_table_rows(counts, min_count=1)
        assert rows[0].lang3 == ""

    def test_rows_sorted_by_lang1_grapheme(self):
        counts = {"lang2": Counter({("z", "z"): 3, ("a", "a"): 3})}
        rows = build_table_rows(counts, min_count=1)
        graphemes = [r.lang1_grapheme for r in rows]
        assert graphemes == sorted(graphemes)

    def test_multiple_graphemes_one_row_each(self):
        counts = {"lang2": Counter({("a", "à"): 3, ("o", "ó"): 3})}
        rows = build_table_rows(counts, min_count=1)
        graphemes = {r.lang1_grapheme for r in rows}
        assert graphemes == {"a", "o"}

    def test_all_language_fields_present(self):
        counts = {"lang2": Counter({("a", "à"): 3})}
        rows = build_table_rows(counts, min_count=1)
        for r in rows:
            for lang in ["lang2", "lang3", "lang4", "lang5", "lang6", "lang7"]:
                assert hasattr(r, lang)


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_writes_correspondence_records(self, tmp_path):
        counts = {"lang2": Counter({("a", "à"): 3})}
        reg = compute_regularity(counts)
        examples = {"lang2": {("a", "à"): ["student"]}}
        records = build_correspondence_records(counts, reg, examples, min_count=1)
        out = tmp_path / "correspondences.csv"
        export_csv(records, out)
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 1
        assert rows[0]["lang1_grapheme"] == "a"
        assert rows[0]["langN_grapheme"] == "à"

    def test_writes_table_rows(self, tmp_path):
        counts = {"lang2": Counter({("a", "à"): 3})}
        rows = build_table_rows(counts, min_count=1)
        out = tmp_path / "table.csv"
        export_csv(rows, out)
        csv_rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(csv_rows) == 1
        assert csv_rows[0]["lang1_grapheme"] == "a"

    def test_empty_list_no_file_created(self, tmp_path):
        out = tmp_path / "empty.csv"
        export_csv([], out)
        assert not out.exists()

    def test_creates_parent_directories(self, tmp_path):
        counts = {"lang2": Counter({("a", "à"): 3})}
        rows = build_table_rows(counts, min_count=1)
        out = tmp_path / "deep" / "nested" / "table.csv"
        export_csv(rows, out)
        assert out.exists()

    def test_unicode_graphemes_preserved(self, tmp_path):
        counts = {"lang2": Counter({("þ", "ħ"): 3})}
        reg = compute_regularity(counts)
        examples = {"lang2": {("þ", "ħ"): ["word"]}}
        records = build_correspondence_records(counts, reg, examples, min_count=1)
        out = tmp_path / "unicode.csv"
        export_csv(records, out)
        content = out.read_text(encoding="utf-8")
        assert "þ" in content
        assert "ħ" in content

    def test_header_matches_dataclass_fields(self, tmp_path):
        counts = {"lang2": Counter({("a", "à"): 3})}
        reg = compute_regularity(counts)
        examples = {"lang2": {("a", "à"): []}}
        records = build_correspondence_records(counts, reg, examples, min_count=1)
        out = tmp_path / "out.csv"
        export_csv(records, out)
        header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == [
            "language", "lang1_grapheme", "langN_grapheme",
            "is_identity", "count", "regularity", "examples"
        ]


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------


class TestComputeSummary:
    def test_total_correspondences_count(self):
        counts = {
            "lang2": Counter({("a", "à"): 3, ("o", "ó"): 2}),
            "lang3": Counter({("a", "ā"): 4}),
        }
        summary = compute_summary(counts, min_count=1)
        assert summary["total_correspondences"] == 3

    def test_below_min_count_not_counted(self):
        counts = {"lang2": Counter({("a", "à"): 3, ("o", "ó"): 1})}
        summary = compute_summary(counts, min_count=2)
        # Only ("a","à"):3 qualifies
        assert summary["total_correspondences"] == 1

    def test_identity_rate_all_identity(self):
        counts = {"lang2": Counter({("a", "a"): 4, ("o", "o"): 4})}
        summary = compute_summary(counts, min_count=1)
        assert summary["identity_rate"]["lang2"] == 1.0

    def test_identity_rate_no_identity(self):
        counts = {"lang2": Counter({("a", "à"): 4, ("o", "ó"): 4})}
        summary = compute_summary(counts, min_count=1)
        assert summary["identity_rate"]["lang2"] == 0.0

    def test_identity_rate_mixed(self):
        counts = {"lang2": Counter({("a", "a"): 3, ("a", "à"): 1})}
        summary = compute_summary(counts, min_count=1)
        assert abs(summary["identity_rate"]["lang2"] - 3 / 4) < 1e-9

    def test_top_changes_excludes_identity(self):
        counts = {"lang2": Counter({("a", "a"): 10, ("o", "ó"): 3})}
        summary = compute_summary(counts, min_count=1)
        changes = summary["top_changes"]["lang2"]
        change_pairs = [(a, b) for a, b, _ in changes]
        assert ("a", "a") not in change_pairs
        assert ("o", "ó") in change_pairs

    def test_top_changes_at_most_5(self):
        pairs = {(f"a{i}", f"b{i}"): 10 - i for i in range(10)}
        counts = {"lang2": Counter(pairs)}
        summary = compute_summary(counts, min_count=1)
        assert len(summary["top_changes"]["lang2"]) <= 5

    def test_top_changes_sorted_descending(self):
        counts = {"lang2": Counter({("a", "à"): 5, ("o", "ó"): 3, ("u", "ú"): 8})}
        summary = compute_summary(counts, min_count=1)
        changes = [cnt for _, _, cnt in summary["top_changes"]["lang2"]]
        assert changes == sorted(changes, reverse=True)

    def test_empty_counts(self):
        summary = compute_summary({}, min_count=1)
        assert summary["total_correspondences"] == 0
        for lang in LANGUAGES[1:]:
            assert summary["identity_rate"].get(lang, 0.0) == 0.0

    def test_language_with_no_qualifying_pairs(self):
        counts = {"lang2": Counter({("a", "à"): 1})}
        summary = compute_summary(counts, min_count=2)
        # No pairs qualify → identity rate 0.0
        assert summary["identity_rate"]["lang2"] == 0.0


# ---------------------------------------------------------------------------
# run (end-to-end with minimal fixture data)
# ---------------------------------------------------------------------------


class TestRun:
    def _write_cognate_sets(self, tmp_path: Path, rows: list[dict]) -> Path:
        path = tmp_path / "cognate_sets.csv"
        write_cognate_sets(path, rows)
        return tmp_path

    def test_missing_cognate_sets_returns_empty(self, tmp_path):
        result = run(processed_dir=tmp_path)
        assert result["correspondences"] == []
        assert result["table_rows"] == []
        assert result["summary"] == {}

    def test_returns_expected_keys(self, tmp_path):
        self._write_cognate_sets(tmp_path, [
            {"english": "student", "lang1": "dóruma", "lang2": "doruma"},
        ])
        result = run(processed_dir=tmp_path)
        assert set(result.keys()) == {"correspondences", "table_rows", "summary"}

    def test_correspondences_are_instances(self, tmp_path):
        self._write_cognate_sets(tmp_path, [
            {"english": "student", "lang1": "dóruma", "lang2": "doruma"},
            {"english": "teacher", "lang1": "talimi", "lang2": "talimi"},
        ])
        result = run(processed_dir=tmp_path, min_count=1)
        for c in result["correspondences"]:
            assert isinstance(c, Correspondence)

    def test_table_rows_are_instances(self, tmp_path):
        self._write_cognate_sets(tmp_path, [
            {"english": "student", "lang1": "dóruma", "lang2": "doruma"},
            {"english": "teacher", "lang1": "talimi", "lang2": "talimi"},
        ])
        result = run(processed_dir=tmp_path, min_count=1)
        for r in result["table_rows"]:
            assert isinstance(r, TableRow)

    def test_output_csvs_created(self, tmp_path):
        self._write_cognate_sets(tmp_path, [
            {"english": "student", "lang1": "dóruma", "lang2": "doruma"},
            {"english": "teacher", "lang1": "talimi", "lang2": "talimi"},
        ])
        run(processed_dir=tmp_path, min_count=1)
        assert (tmp_path / "correspondences.csv").exists()
        assert (tmp_path / "correspondence_table.csv").exists()

    def test_min_count_filters_results(self, tmp_path):
        # Only one row → all pairs have count=1
        self._write_cognate_sets(tmp_path, [
            {"english": "student", "lang1": "dóruma", "lang2": "doruma"},
        ])
        result_high = run(processed_dir=tmp_path, min_count=5)
        result_low = run(processed_dir=tmp_path, min_count=1)
        assert len(result_high["correspondences"]) < len(result_low["correspondences"])

    def test_short_forms_produce_no_correspondences(self, tmp_path):
        # lang1 forms with < MIN_FORM_GRAPHEMES graphemes
        self._write_cognate_sets(tmp_path, [
            {"english": "go", "lang1": "ab", "lang2": "ab"},
        ])
        result = run(processed_dir=tmp_path, min_count=1)
        assert result["correspondences"] == []

    def test_no_csv_written_when_empty_result(self, tmp_path):
        # All forms too short → no qualifying correspondences at min_count=1
        self._write_cognate_sets(tmp_path, [
            {"english": "x", "lang1": "ab", "lang2": "ab"},
        ])
        run(processed_dir=tmp_path, min_count=1)
        # export_csv skips empty lists, so files should not exist
        assert not (tmp_path / "correspondences.csv").exists()
        assert not (tmp_path / "correspondence_table.csv").exists()

    def test_identity_pairs_present_in_summary(self, tmp_path):
        self._write_cognate_sets(tmp_path, [
            {"english": "student", "lang1": "doruma", "lang2": "doruma"},
            {"english": "teacher", "lang1": "talimi", "lang2": "talimi"},
        ])
        result = run(processed_dir=tmp_path, min_count=1)
        # With identical forms, identity rate should be 1.0
        assert result["summary"]["identity_rate"].get("lang2", 0.0) == 1.0


# ---------------------------------------------------------------------------
# Integration tests (require actual processed data)
# ---------------------------------------------------------------------------


@integration
class TestIntegration:
    @pytest.fixture(scope="class")
    def results(self):
        return run(processed_dir=PROCESSED_DIR)

    def test_correspondences_nonempty(self, results):
        assert len(results["correspondences"]) > 0

    def test_table_rows_nonempty(self, results):
        assert len(results["table_rows"]) > 0

    def test_all_correspondences_above_min_count(self, results):
        for c in results["correspondences"]:
            assert c.count >= MIN_COGNATES

    def test_regularity_in_range(self, results):
        for c in results["correspondences"]:
            assert 0.0 <= c.regularity <= 1.0, f"bad regularity: {c}"

    def test_all_languages_in_range(self, results):
        for c in results["correspondences"]:
            assert c.language in LANGUAGES[1:], f"unexpected language: {c.language}"

    def test_is_identity_flag_correct(self, results):
        for c in results["correspondences"]:
            expected = (c.lang1_grapheme == c.langN_grapheme)
            assert c.is_identity == expected

    def test_table_rows_sorted(self, results):
        graphemes = [r.lang1_grapheme for r in results["table_rows"]]
        assert graphemes == sorted(graphemes)

    def test_summary_keys_present(self, results):
        assert "total_correspondences" in results["summary"]
        assert "identity_rate" in results["summary"]
        assert "top_changes" in results["summary"]

    def test_identity_rate_between_0_and_1(self, results):
        for lang, rate in results["summary"]["identity_rate"].items():
            assert 0.0 <= rate <= 1.0, f"{lang}: {rate}"

    def test_correspondences_csv_readable(self):
        path = PROCESSED_DIR / "correspondences.csv"
        if not path.exists():
            pytest.skip("correspondences.csv not yet generated")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) > 0
        assert "lang1_grapheme" in rows[0]
        assert "regularity" in rows[0]

    def test_correspondence_table_csv_readable(self):
        path = PROCESSED_DIR / "correspondence_table.csv"
        if not path.exists():
            pytest.skip("correspondence_table.csv not yet generated")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) > 0
        assert "lang1_grapheme" in rows[0]
