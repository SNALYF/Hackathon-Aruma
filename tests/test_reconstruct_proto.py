"""
Tests for src/reconstruct_proto.py.

Unit tests use inline fixtures and tmp_path so they do not depend on real data.
Integration tests are marked @pytest.mark.integration and skipped when the
processed data directory is absent.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.reconstruct_proto import (
    COUNT_THRESHOLD,
    HIGH_COGNATE_MIN,
    LANGUAGES,
    MEDIUM_COGNATE_MIN,
    PROCESSED_DIR,
    REGULARITY_THRESHOLD,
    ProtoEntry,
    SoundLaw,
    _change_laws,
    _confidence,
    _coverage_stats,
    _law_table,
    _read_csv,
    _vowel_laws,
    build_proto_lexicon,
    export_csv,
    extract_sound_laws,
    generate_summary,
    load_data,
    run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROCESSED_DATA_AVAILABLE = (PROCESSED_DIR / "correspondences.csv").exists()
integration = pytest.mark.skipif(
    not PROCESSED_DATA_AVAILABLE, reason="processed data directory not found"
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_corr_row(
    language="lang2",
    lang1_grapheme="a",
    langN_grapheme="à",
    is_identity="False",
    count="10",
    regularity="0.9",
    examples="student, teacher",
) -> dict:
    return {
        "language": language,
        "lang1_grapheme": lang1_grapheme,
        "langN_grapheme": langN_grapheme,
        "is_identity": is_identity,
        "count": count,
        "regularity": regularity,
        "examples": examples,
    }


def make_sound_law(
    language="lang2",
    proto_grapheme="a",
    reflex="à",
    is_change=True,
    regularity=0.9,
    count=10,
    examples="student",
) -> SoundLaw:
    return SoundLaw(
        language=language,
        proto_grapheme=proto_grapheme,
        reflex=reflex,
        is_change=is_change,
        regularity=regularity,
        count=count,
        examples=examples,
    )


def make_proto_entry(
    english="student",
    pos="NN",
    proto_form="*dóruma",
    confidence="high",
    n_cognates=5,
    lang1="dóruma",
    lang2="doruma",
    lang3="",
    lang4="",
    lang5="doruma",
    lang6="",
    lang7="",
) -> ProtoEntry:
    return ProtoEntry(
        english=english, pos=pos, proto_form=proto_form,
        confidence=confidence, n_cognates=n_cognates,
        lang1=lang1, lang2=lang2, lang3=lang3,
        lang4=lang4, lang5=lang5, lang6=lang6, lang7=lang7,
    )


# ---------------------------------------------------------------------------
# _read_csv
# ---------------------------------------------------------------------------


class TestReadCsv:
    def test_existing_file_returns_rows(self, tmp_path):
        p = tmp_path / "data.csv"
        write_csv(p, [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])
        rows = _read_csv(p)
        assert len(rows) == 2
        assert rows[0]["a"] == "1"

    def test_missing_file_returns_empty(self, tmp_path):
        assert _read_csv(tmp_path / "nonexistent.csv") == []

    def test_returns_list_of_dicts(self, tmp_path):
        p = tmp_path / "data.csv"
        write_csv(p, [{"x": "hello"}])
        rows = _read_csv(p)
        assert isinstance(rows[0], dict)

    def test_unicode_preserved(self, tmp_path):
        p = tmp_path / "data.csv"
        write_csv(p, [{"form": "dóruma"}])
        rows = _read_csv(p)
        assert rows[0]["form"] == "dóruma"


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def _setup(self, tmp_path):
        write_csv(tmp_path / "correspondences.csv", [make_corr_row()])
        write_csv(tmp_path / "cognate_sets.csv", [
            {"english": "student", "pos": "NN", "lang1": "dóruma",
             "lang2": "doruma", "lang3": "", "lang4": "", "lang5": "", "lang6": "", "lang7": ""}
        ])
        write_csv(tmp_path / "dictionary.csv", [
            {"english": "student", "native_form": "dóruma", "pos": "NN", "compound_explanation": ""}
        ])

    def test_returns_three_lists(self, tmp_path):
        self._setup(tmp_path)
        result = load_data(tmp_path)
        assert len(result) == 3
        corrs, csets, dicts = result
        assert isinstance(corrs, list)
        assert isinstance(csets, list)
        assert isinstance(dicts, list)

    def test_loads_correspondences(self, tmp_path):
        self._setup(tmp_path)
        corrs, _, _ = load_data(tmp_path)
        assert len(corrs) == 1
        assert corrs[0]["language"] == "lang2"

    def test_loads_cognate_sets(self, tmp_path):
        self._setup(tmp_path)
        _, csets, _ = load_data(tmp_path)
        assert len(csets) == 1
        assert csets[0]["english"] == "student"

    def test_loads_dictionary(self, tmp_path):
        self._setup(tmp_path)
        _, _, dicts = load_data(tmp_path)
        assert len(dicts) == 1
        assert dicts[0]["native_form"] == "dóruma"

    def test_missing_files_return_empty(self, tmp_path):
        corrs, csets, dicts = load_data(tmp_path / "nonexistent")
        assert corrs == []
        assert csets == []
        assert dicts == []


# ---------------------------------------------------------------------------
# extract_sound_laws
# ---------------------------------------------------------------------------


class TestExtractSoundLaws:
    def test_empty_input(self):
        assert extract_sound_laws([]) == []

    def test_row_below_regularity_excluded(self):
        row = make_corr_row(regularity="0.5", count="10")
        laws = extract_sound_laws([row], regularity_threshold=0.7, count_threshold=5)
        assert laws == []

    def test_row_below_count_excluded(self):
        row = make_corr_row(regularity="0.9", count="3")
        laws = extract_sound_laws([row], regularity_threshold=0.7, count_threshold=5)
        assert laws == []

    def test_row_meeting_both_thresholds_included(self):
        row = make_corr_row(regularity="0.9", count="10")
        laws = extract_sound_laws([row], regularity_threshold=0.7, count_threshold=5)
        assert len(laws) == 1

    def test_row_at_exact_thresholds_included(self):
        row = make_corr_row(regularity="0.7", count="5")
        laws = extract_sound_laws([row], regularity_threshold=0.7, count_threshold=5)
        assert len(laws) == 1

    def test_is_change_false_when_is_identity_is_false_string(self):
        # is_identity="False" in CSV → is_change=True
        row = make_corr_row(is_identity="False")
        laws = extract_sound_laws([row], regularity_threshold=0.0, count_threshold=0)
        assert laws[0].is_change is True

    def test_is_change_false_when_is_identity_true(self):
        row = make_corr_row(is_identity="True")
        laws = extract_sound_laws([row], regularity_threshold=0.0, count_threshold=0)
        assert laws[0].is_change is False

    def test_fields_populated_correctly(self):
        row = make_corr_row(
            language="lang3", lang1_grapheme="o", langN_grapheme="ó",
            is_identity="False", count="12", regularity="0.85",
            examples="student, teacher"
        )
        laws = extract_sound_laws([row], regularity_threshold=0.0, count_threshold=0)
        law = laws[0]
        assert law.language == "lang3"
        assert law.proto_grapheme == "o"
        assert law.reflex == "ó"
        assert law.count == 12
        assert abs(law.regularity - 0.85) < 1e-9
        assert law.examples == "student, teacher"

    def test_regularity_rounded_to_3_decimals(self):
        row = make_corr_row(regularity="0.8571428")
        laws = extract_sound_laws([row], regularity_threshold=0.0, count_threshold=0)
        assert len(str(laws[0].regularity).split(".")[-1]) <= 3

    def test_sorted_by_language_then_count_descending(self):
        rows = [
            make_corr_row(language="lang3", count="5"),
            make_corr_row(language="lang2", count="10"),
            make_corr_row(language="lang2", count="20"),
        ]
        laws = extract_sound_laws(rows, regularity_threshold=0.0, count_threshold=0)
        langs = [l.language for l in laws]
        assert langs.index("lang2") < langs.index("lang3")
        lang2_laws = [l for l in laws if l.language == "lang2"]
        assert lang2_laws[0].count > lang2_laws[1].count

    def test_identity_correspondences_included(self):
        row = make_corr_row(is_identity="True", regularity="1.0", count="20")
        laws = extract_sound_laws([row], regularity_threshold=0.7, count_threshold=5)
        assert len(laws) == 1
        assert laws[0].is_change is False


# ---------------------------------------------------------------------------
# _confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_zero_cognates_low(self):
        assert _confidence(0) == "low"

    def test_one_cognate_low(self):
        assert _confidence(1) == "low"

    def test_two_cognates_medium(self):
        assert _confidence(MEDIUM_COGNATE_MIN) == "medium"

    def test_three_cognates_medium(self):
        assert _confidence(HIGH_COGNATE_MIN - 1) == "medium"

    def test_four_cognates_high(self):
        assert _confidence(HIGH_COGNATE_MIN) == "high"

    def test_six_cognates_high(self):
        assert _confidence(6) == "high"

    def test_boundary_medium_min(self):
        # exactly MEDIUM_COGNATE_MIN → medium
        assert _confidence(MEDIUM_COGNATE_MIN) == "medium"

    def test_boundary_high_min(self):
        # exactly HIGH_COGNATE_MIN → high
        assert _confidence(HIGH_COGNATE_MIN) == "high"


# ---------------------------------------------------------------------------
# build_proto_lexicon
# ---------------------------------------------------------------------------


class TestBuildProtoLexicon:
    def _dict(self):
        return [
            {"english": "student", "native_form": "dóruma", "pos": "NN", "compound_explanation": ""},
            {"english": "to create", "native_form": "góruamu", "pos": "VB", "compound_explanation": ""},
        ]

    def _csets(self):
        return [
            {"english": "student", "lang1": "dóruma",
             "lang2": "doruma", "lang3": "doruma", "lang4": "doruma",
             "lang5": "doruma", "lang6": "", "lang7": ""},
        ]

    def test_one_entry_per_dictionary_row(self):
        entries = build_proto_lexicon(self._csets(), self._dict())
        assert len(entries) == 2

    def test_proto_form_prefixed_with_asterisk(self):
        entries = build_proto_lexicon(self._csets(), self._dict())
        for e in entries:
            assert e.proto_form.startswith("*")

    def test_proto_form_uses_dictionary_citation(self):
        entries = build_proto_lexicon(self._csets(), self._dict())
        student = next(e for e in entries if e.english == "student")
        assert student.proto_form == "*dóruma"

    def test_lang1_from_dictionary_not_cognate_set(self):
        # Cognate set may have a different (inflected) lang1 form
        csets = [{"english": "student", "lang1": "inflected_form",
                  "lang2": "x", "lang3": "", "lang4": "", "lang5": "", "lang6": "", "lang7": ""}]
        entries = build_proto_lexicon(csets, self._dict())
        student = next(e for e in entries if e.english == "student")
        assert student.lang1 == "dóruma"  # from dictionary, not cognate set

    def test_n_cognates_counts_non_empty_lang2_to_7(self):
        entries = build_proto_lexicon(self._csets(), self._dict())
        student = next(e for e in entries if e.english == "student")
        # lang2,3,4,5 are non-empty, lang6,7 are empty → 4
        assert student.n_cognates == 4

    def test_confidence_assigned_correctly(self):
        entries = build_proto_lexicon(self._csets(), self._dict())
        student = next(e for e in entries if e.english == "student")
        assert student.confidence == "high"  # 4 cognates → high

    def test_no_cognate_set_gives_empty_forms(self):
        entries = build_proto_lexicon([], self._dict())
        for e in entries:
            for lang in ["lang2", "lang3", "lang4", "lang5", "lang6", "lang7"]:
                assert getattr(e, lang) == ""

    def test_no_cognate_set_gives_low_confidence(self):
        entries = build_proto_lexicon([], self._dict())
        for e in entries:
            assert e.confidence == "low"

    def test_english_and_pos_from_dictionary(self):
        entries = build_proto_lexicon(self._csets(), self._dict())
        student = next(e for e in entries if e.english == "student")
        assert student.pos == "NN"

    def test_empty_dictionary_returns_empty(self):
        assert build_proto_lexicon(self._csets(), []) == []

    def test_lang_forms_populated_from_cognate_set(self):
        entries = build_proto_lexicon(self._csets(), self._dict())
        student = next(e for e in entries if e.english == "student")
        assert student.lang2 == "doruma"
        assert student.lang6 == ""


# ---------------------------------------------------------------------------
# _coverage_stats
# ---------------------------------------------------------------------------


class TestCoverageStats:
    def test_total_count(self):
        entries = [make_proto_entry(), make_proto_entry()]
        stats = _coverage_stats(entries)
        assert stats["total"] == 2

    def test_empty_entries(self):
        stats = _coverage_stats([])
        assert stats["total"] == 0
        assert stats["high"] == 0
        assert stats["medium"] == 0
        assert stats["low"] == 0

    def test_confidence_buckets(self):
        entries = [
            make_proto_entry(confidence="high"),
            make_proto_entry(confidence="high"),
            make_proto_entry(confidence="medium"),
            make_proto_entry(confidence="low"),
        ]
        stats = _coverage_stats(entries)
        assert stats["high"] == 2
        assert stats["medium"] == 1
        assert stats["low"] == 1

    def test_lang_coverage_counts_non_empty(self):
        entries = [
            make_proto_entry(lang2="doruma", lang3="", lang5="doruma"),
            make_proto_entry(lang2="talimi", lang3="talimi", lang5=""),
        ]
        stats = _coverage_stats(entries)
        assert stats["lang_coverage"]["lang2"] == 2
        assert stats["lang_coverage"]["lang3"] == 1
        assert stats["lang_coverage"]["lang5"] == 1

    def test_lang_coverage_excludes_lang1(self):
        entries = [make_proto_entry(lang1="dóruma")]
        stats = _coverage_stats(entries)
        assert "lang1" not in stats["lang_coverage"]


# ---------------------------------------------------------------------------
# _law_table, _vowel_laws, _change_laws
# ---------------------------------------------------------------------------


class TestLawFilters:
    def _laws(self):
        return [
            make_sound_law(language="lang2", proto_grapheme="a", is_change=True),
            make_sound_law(language="lang2", proto_grapheme="o", is_change=False),
            make_sound_law(language="lang3", proto_grapheme="á", is_change=True),
            make_sound_law(language="lang2", proto_grapheme="d", is_change=True),
        ]

    def test_law_table_filters_by_language(self):
        result = _law_table(self._laws(), "lang2")
        assert all(l.language == "lang2" for l in result)
        assert len(result) == 3

    def test_law_table_wrong_language_returns_empty(self):
        assert _law_table(self._laws(), "lang7") == []

    def test_change_laws_excludes_identity(self):
        result = _change_laws(self._laws(), "lang2")
        assert all(l.is_change for l in result)

    def test_change_laws_filters_by_language(self):
        result = _change_laws(self._laws(), "lang2")
        assert all(l.language == "lang2" for l in result)

    def test_vowel_laws_only_vowel_graphemes(self):
        result = _vowel_laws(self._laws(), "lang2")
        for l in result:
            assert l.proto_grapheme in "aeiouáéíóúàèìòùãẽĩõũ"
        # "d" is a consonant → must not be included
        assert not any(l.proto_grapheme == "d" for l in result)

    def test_vowel_laws_filters_by_language(self):
        result = _vowel_laws(self._laws(), "lang2")
        assert all(l.language == "lang2" for l in result)


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    def _inputs(self):
        laws = [
            make_sound_law(language="lang2", proto_grapheme="a", reflex="à", is_change=True),
            make_sound_law(language="lang2", proto_grapheme="o", reflex="o", is_change=False),
        ]
        entries = [
            make_proto_entry(english="student", confidence="high"),
            make_proto_entry(english="teacher", confidence="medium"),
            make_proto_entry(english="word", confidence="low"),
        ]
        correspondences = [
            make_corr_row(language="lang2", lang1_grapheme="a", langN_grapheme="à", count="10"),
        ]
        return laws, entries, correspondences

    def test_returns_string(self):
        laws, entries, corrs = self._inputs()
        result = generate_summary(laws, entries, corrs)
        assert isinstance(result, str)

    def test_ends_with_newline(self):
        laws, entries, corrs = self._inputs()
        result = generate_summary(laws, entries, corrs)
        assert result.endswith("\n")

    def test_contains_overview_header(self):
        laws, entries, corrs = self._inputs()
        result = generate_summary(laws, entries, corrs)
        assert "## Overview" in result

    def test_contains_sound_laws_header(self):
        laws, entries, corrs = self._inputs()
        result = generate_summary(laws, entries, corrs)
        assert "## Sound Laws" in result

    def test_contains_proto_phoneme_inventory(self):
        laws, entries, corrs = self._inputs()
        result = generate_summary(laws, entries, corrs)
        assert "## Proto-Phoneme Inventory" in result

    def test_contains_coverage_statistics(self):
        laws, entries, corrs = self._inputs()
        result = generate_summary(laws, entries, corrs)
        assert "## Coverage Statistics" in result

    def test_contains_proto_lexicon_sample(self):
        laws, entries, corrs = self._inputs()
        result = generate_summary(laws, entries, corrs)
        assert "## Proto-Lexicon Sample" in result

    def test_contains_limitations_section(self):
        laws, entries, corrs = self._inputs()
        result = generate_summary(laws, entries, corrs)
        assert "## Limitations" in result

    def test_sample_only_high_confidence_entries(self):
        laws, entries, corrs = self._inputs()
        result = generate_summary(laws, entries, corrs)
        # "teacher" is medium confidence and "word" is low — should not appear in sample table
        # "student" is high — its proto_form "*dóruma" should appear
        assert "*dóruma" in result

    def test_confidence_counts_in_overview(self):
        laws, entries, corrs = self._inputs()
        result = generate_summary(laws, entries, corrs)
        # 1 high, 1 medium, 1 low
        assert "**High confidence**" in result
        assert "**Medium confidence**" in result
        assert "**Low confidence**" in result

    def test_language_sections_present(self):
        laws, entries, corrs = self._inputs()
        result = generate_summary(laws, entries, corrs)
        for lang in LANGUAGES[1:]:
            assert f"### {lang}" in result

    def test_non_identity_change_in_sound_law_table(self):
        laws = [make_sound_law(language="lang2", proto_grapheme="a", reflex="à", is_change=True,
                               regularity=0.9, count=10, examples="student")]
        entries = [make_proto_entry()]
        result = generate_summary(laws, entries, [])
        assert "a" in result
        assert "à" in result

    def test_empty_laws_produces_no_threshold_line_for_lang(self):
        # With no change laws for a language, the fallback message should appear
        result = generate_summary([], [make_proto_entry()], [])
        assert "No non-identity correspondences above threshold." in result


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_writes_sound_law_records(self, tmp_path):
        law = make_sound_law()
        out = tmp_path / "sound_laws.csv"
        export_csv([law], out)
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 1
        assert rows[0]["language"] == "lang2"
        assert rows[0]["proto_grapheme"] == "a"

    def test_writes_proto_entry_records(self, tmp_path):
        entry = make_proto_entry()
        out = tmp_path / "proto_lexicon.csv"
        export_csv([entry], out)
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 1
        assert rows[0]["english"] == "student"
        assert rows[0]["proto_form"] == "*dóruma"

    def test_empty_list_no_file(self, tmp_path):
        out = tmp_path / "empty.csv"
        export_csv([], out)
        assert not out.exists()

    def test_creates_parent_directories(self, tmp_path):
        law = make_sound_law()
        out = tmp_path / "deep" / "nested" / "laws.csv"
        export_csv([law], out)
        assert out.exists()

    def test_unicode_preserved(self, tmp_path):
        law = make_sound_law(proto_grapheme="ó", reflex="ò")
        out = tmp_path / "unicode.csv"
        export_csv([law], out)
        content = out.read_text(encoding="utf-8")
        assert "ó" in content
        assert "ò" in content

    def test_sound_law_header(self, tmp_path):
        out = tmp_path / "laws.csv"
        export_csv([make_sound_law()], out)
        header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == [
            "language", "proto_grapheme", "reflex",
            "is_change", "regularity", "count", "examples"
        ]

    def test_proto_entry_header(self, tmp_path):
        out = tmp_path / "lexicon.csv"
        export_csv([make_proto_entry()], out)
        header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == [
            "english", "pos", "proto_form", "confidence", "n_cognates",
            "lang1", "lang2", "lang3", "lang4", "lang5", "lang6", "lang7"
        ]


# ---------------------------------------------------------------------------
# run (end-to-end with minimal fixture data)
# ---------------------------------------------------------------------------


class TestRun:
    def _write_processed(self, tmp_path: Path):
        write_csv(tmp_path / "correspondences.csv", [
            make_corr_row(language="lang2", lang1_grapheme="a", langN_grapheme="à",
                          is_identity="False", count="10", regularity="0.9"),
            make_corr_row(language="lang2", lang1_grapheme="o", langN_grapheme="o",
                          is_identity="True", count="8", regularity="1.0"),
        ])
        write_csv(tmp_path / "cognate_sets.csv", [
            {"english": "student", "pos": "NN", "lang1": "dóruma",
             "lang2": "doruma", "lang3": "doruma", "lang4": "doruma",
             "lang5": "doruma", "lang6": "", "lang7": ""},
        ])
        write_csv(tmp_path / "dictionary.csv", [
            {"english": "student", "native_form": "dóruma", "pos": "NN", "compound_explanation": ""},
        ])

    def test_returns_expected_keys(self, tmp_path):
        self._write_processed(tmp_path)
        result = run(processed_dir=tmp_path)
        assert set(result.keys()) == {"sound_laws", "proto_lexicon", "summary"}

    def test_sound_laws_are_instances(self, tmp_path):
        self._write_processed(tmp_path)
        result = run(processed_dir=tmp_path, regularity_threshold=0.0, count_threshold=0)
        for law in result["sound_laws"]:
            assert isinstance(law, SoundLaw)

    def test_proto_lexicon_are_instances(self, tmp_path):
        self._write_processed(tmp_path)
        result = run(processed_dir=tmp_path)
        for e in result["proto_lexicon"]:
            assert isinstance(e, ProtoEntry)

    def test_summary_is_string(self, tmp_path):
        self._write_processed(tmp_path)
        result = run(processed_dir=tmp_path)
        assert isinstance(result["summary"], str)

    def test_output_files_created(self, tmp_path):
        self._write_processed(tmp_path)
        run(processed_dir=tmp_path, regularity_threshold=0.0, count_threshold=0)
        assert (tmp_path / "sound_laws.csv").exists()
        assert (tmp_path / "proto_lexicon.csv").exists()
        assert (tmp_path / "reconstruction_summary.md").exists()

    def test_summary_md_always_written(self, tmp_path):
        # Even with no correspondences, reconstruction_summary.md is created
        run(processed_dir=tmp_path)
        assert (tmp_path / "reconstruction_summary.md").exists()

    def test_count_threshold_filters_laws(self, tmp_path):
        self._write_processed(tmp_path)
        result_strict = run(processed_dir=tmp_path, regularity_threshold=0.0, count_threshold=100)
        result_loose = run(processed_dir=tmp_path, regularity_threshold=0.0, count_threshold=1)
        assert len(result_strict["sound_laws"]) < len(result_loose["sound_laws"])

    def test_regularity_threshold_filters_laws(self, tmp_path):
        self._write_processed(tmp_path)
        result_strict = run(processed_dir=tmp_path, regularity_threshold=0.99, count_threshold=0)
        result_loose = run(processed_dir=tmp_path, regularity_threshold=0.0, count_threshold=0)
        assert len(result_strict["sound_laws"]) <= len(result_loose["sound_laws"])

    def test_proto_lexicon_length_equals_dictionary(self, tmp_path):
        self._write_processed(tmp_path)
        result = run(processed_dir=tmp_path)
        assert len(result["proto_lexicon"]) == 1

    def test_empty_processed_dir_returns_empty_laws_and_lexicon(self, tmp_path):
        result = run(processed_dir=tmp_path)
        assert result["sound_laws"] == []
        assert result["proto_lexicon"] == []

    def test_no_csv_written_when_empty(self, tmp_path):
        run(processed_dir=tmp_path)
        assert not (tmp_path / "sound_laws.csv").exists()
        assert not (tmp_path / "proto_lexicon.csv").exists()


# ---------------------------------------------------------------------------
# Integration tests (require actual processed data)
# ---------------------------------------------------------------------------


@integration
class TestIntegration:
    @pytest.fixture(scope="class")
    def results(self):
        return run(processed_dir=PROCESSED_DIR)

    def test_sound_laws_nonempty(self, results):
        assert len(results["sound_laws"]) > 0

    def test_proto_lexicon_nonempty(self, results):
        assert len(results["proto_lexicon"]) > 0

    def test_all_sound_laws_above_thresholds(self, results):
        for law in results["sound_laws"]:
            assert law.regularity >= REGULARITY_THRESHOLD
            assert law.count >= COUNT_THRESHOLD

    def test_sound_law_languages_valid(self, results):
        for law in results["sound_laws"]:
            assert law.language in LANGUAGES[1:]

    def test_proto_forms_start_with_asterisk(self, results):
        for e in results["proto_lexicon"]:
            assert e.proto_form.startswith("*"), f"bad proto_form: {e.proto_form!r}"

    def test_confidence_values_valid(self, results):
        valid = {"high", "medium", "low"}
        for e in results["proto_lexicon"]:
            assert e.confidence in valid

    def test_n_cognates_consistent_with_forms(self, results):
        for e in results["proto_lexicon"]:
            actual = sum(1 for lang in LANGUAGES[1:] if getattr(e, lang))
            assert e.n_cognates == actual, f"mismatch for {e.english!r}"

    def test_summary_contains_all_sections(self, results):
        summary = results["summary"]
        for section in [
            "## Overview", "## Proto-Phoneme Inventory", "## Sound Laws",
            "## Proto-Lexicon Sample", "## Coverage Statistics",
            "## Candidate Proto-Form Corrections", "## Limitations"
        ]:
            assert section in summary, f"missing section: {section!r}"

    def test_sound_laws_csv_readable(self):
        path = PROCESSED_DIR / "sound_laws.csv"
        if not path.exists():
            pytest.skip("sound_laws.csv not yet generated")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) > 0
        assert "proto_grapheme" in rows[0]
        assert "regularity" in rows[0]

    def test_proto_lexicon_csv_readable(self):
        path = PROCESSED_DIR / "proto_lexicon.csv"
        if not path.exists():
            pytest.skip("proto_lexicon.csv not yet generated")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) > 0
        assert "proto_form" in rows[0]
        assert "confidence" in rows[0]

    def test_reconstruction_summary_md_exists(self):
        path = PROCESSED_DIR / "reconstruction_summary.md"
        if not path.exists():
            pytest.skip("reconstruction_summary.md not yet generated")
        content = path.read_text(encoding="utf-8")
        assert "# Proto-Language Reconstruction Summary" in content

    def test_sound_laws_sorted_by_language_then_count(self, results):
        laws = results["sound_laws"]
        for i in range(len(laws) - 1):
            a, b = laws[i], laws[i + 1]
            if a.language == b.language:
                assert a.count >= b.count, f"count order violated: {a} before {b}"
            else:
                assert a.language <= b.language, f"language order violated: {a.language} before {b.language}"
