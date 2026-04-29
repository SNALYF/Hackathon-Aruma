"""
Tests for src/validate.py.

Unit tests use inline fixtures and tmp_path so they do not depend on real data.
Integration tests are marked @pytest.mark.integration and skipped when the
processed data directory is absent.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.validate import (
    CLOSE_THRESHOLD,
    EXACT_THRESHOLD,
    LANGUAGES,
    PARTIAL_THRESHOLD,
    PROCESSED_DIR,
    ExceptionRecord,
    PredictionResult,
    _build_proto_lookup,
    _content_gloss,
    _diagnose,
    _first_missing_grapheme,
    _lcs_length,
    _read_csv,
    _strip,
    analyze_exceptions,
    build_laws_map,
    compute_accuracy_stats,
    evaluate_predictions,
    export_csv,
    generate_proto_texts,
    generate_report,
    load_data,
    lcs_similarity,
    match_class,
    predict_form,
    predicted_coverage,
    run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROCESSED_DATA_AVAILABLE = (PROCESSED_DIR / "proto_lexicon.csv").exists()
integration = pytest.mark.skipif(
    not PROCESSED_DATA_AVAILABLE, reason="processed data directory not found"
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_prediction(
    english="student",
    pos="NN",
    proto_form="*dóruma",
    language="lang2",
    predicted_form="doruma",
    actual_form="doruma",
    raw_similarity=1.0,
    predicted_coverage=1.0,
    match_class="exact",
) -> PredictionResult:
    return PredictionResult(
        english=english, pos=pos, proto_form=proto_form,
        language=language, predicted_form=predicted_form,
        actual_form=actual_form, raw_similarity=raw_similarity,
        predicted_coverage=predicted_coverage, match_class=match_class,
    )


def make_exception(
    english="student",
    pos="NN",
    proto_form="*dóruma",
    language="lang2",
    predicted_form="doruma",
    actual_form="xyz",
    predicted_coverage=0.1,
    mismatched_grapheme="d",
    diagnosis="irregular",
) -> ExceptionRecord:
    return ExceptionRecord(
        english=english, pos=pos, proto_form=proto_form,
        language=language, predicted_form=predicted_form,
        actual_form=actual_form, predicted_coverage=predicted_coverage,
        mismatched_grapheme=mismatched_grapheme, diagnosis=diagnosis,
    )


def make_laws_row(
    language="lang2", proto_grapheme="a", reflex="à",
    regularity="0.9", count="10", is_identity="False", examples="student"
) -> dict:
    return {
        "language": language, "proto_grapheme": proto_grapheme,
        "reflex": reflex, "regularity": regularity, "count": count,
        "is_identity": is_identity, "examples": examples,
    }


# ---------------------------------------------------------------------------
# _strip
# ---------------------------------------------------------------------------


class TestStrip:
    def test_removes_diacritics(self):
        assert _strip("dóruma") == "doruma"

    def test_returns_lowercase(self):
        assert _strip("DÓRUMA") == "doruma"

    def test_empty_string(self):
        assert _strip("") == ""

    def test_plain_ascii_unchanged(self):
        assert _strip("doruma") == "doruma"

    def test_grave_removed(self):
        assert _strip("àgùùg") == "aguug"

    def test_ipa_thorn_preserved(self):
        assert "þ" in _strip("þadoma")


# ---------------------------------------------------------------------------
# _lcs_length
# ---------------------------------------------------------------------------


class TestLcsLength:
    def test_identical(self):
        assert _lcs_length("abc", "abc") == 3

    def test_empty_first(self):
        assert _lcs_length("", "abc") == 0

    def test_empty_second(self):
        assert _lcs_length("abc", "") == 0

    def test_both_empty(self):
        assert _lcs_length("", "") == 0

    def test_no_common(self):
        assert _lcs_length("abc", "xyz") == 0

    def test_subsequence(self):
        assert _lcs_length("ace", "abcde") == 3

    def test_symmetry(self):
        assert _lcs_length("doruma", "dorumaguug") == _lcs_length("dorumaguug", "doruma")


# ---------------------------------------------------------------------------
# lcs_similarity
# ---------------------------------------------------------------------------


class TestLcsSimilarity:
    def test_identical(self):
        assert lcs_similarity("doruma", "doruma") == 1.0

    def test_empty_both(self):
        assert lcs_similarity("", "") == 0.0

    def test_empty_first(self):
        assert lcs_similarity("", "abc") == 0.0

    def test_result_rounded_to_4(self):
        sim = lcs_similarity("doruma", "dorumaguug")
        assert len(str(sim).split(".")[-1]) <= 4

    def test_diacritics_stripped_before_comparison(self):
        # "dóruma" vs "doruma" — same after stripping → 1.0
        assert lcs_similarity("dóruma", "doruma") == 1.0

    def test_result_between_0_and_1(self):
        sim = lcs_similarity("abc", "xyz")
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# _read_csv / load_data
# ---------------------------------------------------------------------------


class TestReadCsv:
    def test_existing_file(self, tmp_path):
        p = tmp_path / "f.csv"
        write_csv(p, [{"a": "1"}])
        assert _read_csv(p) == [{"a": "1"}]

    def test_missing_file(self, tmp_path):
        assert _read_csv(tmp_path / "nope.csv") == []


class TestLoadData:
    def test_returns_four_keys(self, tmp_path):
        data = load_data(tmp_path)
        assert set(data.keys()) == {"sound_laws", "proto_lexicon", "cognate_sets", "corpus"}

    def test_missing_dir_all_empty(self, tmp_path):
        data = load_data(tmp_path / "nonexistent")
        for v in data.values():
            assert v == []

    def test_loads_each_file(self, tmp_path):
        write_csv(tmp_path / "sound_laws.csv", [make_laws_row()])
        write_csv(tmp_path / "proto_lexicon.csv",
                  [{"english": "student", "pos": "NN", "proto_form": "*dóruma",
                    "confidence": "high", "n_cognates": "4",
                    "lang1": "dóruma", "lang2": "doruma", "lang3": "", "lang4": "",
                    "lang5": "", "lang6": "", "lang7": ""}])
        write_csv(tmp_path / "cognate_sets.csv",
                  [{"english": "student", "lang1": "dóruma", "lang2": "doruma",
                    "lang3": "", "lang4": "", "lang5": "", "lang6": "", "lang7": ""}])
        write_csv(tmp_path / "corpus.csv",
                  [{"language": "lang1", "sentence_id": "1", "surface": "dóruma.",
                    "segmented": "dóruma", "gloss": "student", "translation": "The student."}])
        data = load_data(tmp_path)
        assert len(data["sound_laws"]) == 1
        assert len(data["proto_lexicon"]) == 1
        assert len(data["cognate_sets"]) == 1
        assert len(data["corpus"]) == 1


# ---------------------------------------------------------------------------
# build_laws_map
# ---------------------------------------------------------------------------


class TestBuildLawsMap:
    def test_basic_mapping(self):
        rows = [make_laws_row(language="lang2", proto_grapheme="a", reflex="à", regularity="0.9")]
        m = build_laws_map(rows)
        assert m[("lang2", "a")] == "à"

    def test_empty_input(self):
        assert build_laws_map([]) == {}

    def test_highest_regularity_wins(self):
        rows = [
            make_laws_row(language="lang2", proto_grapheme="a", reflex="à", regularity="0.6"),
            make_laws_row(language="lang2", proto_grapheme="a", reflex="ā", regularity="0.9"),
        ]
        m = build_laws_map(rows)
        assert m[("lang2", "a")] == "ā"

    def test_different_languages_independent(self):
        rows = [
            make_laws_row(language="lang2", proto_grapheme="a", reflex="à"),
            make_laws_row(language="lang3", proto_grapheme="a", reflex="â"),
        ]
        m = build_laws_map(rows)
        assert m[("lang2", "a")] == "à"
        assert m[("lang3", "a")] == "â"

    def test_different_graphemes_independent(self):
        rows = [
            make_laws_row(language="lang2", proto_grapheme="a", reflex="à"),
            make_laws_row(language="lang2", proto_grapheme="o", reflex="ó"),
        ]
        m = build_laws_map(rows)
        assert m[("lang2", "a")] == "à"
        assert m[("lang2", "o")] == "ó"


# ---------------------------------------------------------------------------
# predict_form
# ---------------------------------------------------------------------------


class TestPredictForm:
    def test_applies_law(self):
        laws = {("lang2", "a"): "à"}
        assert predict_form("*dama", "lang2", laws) == "dàmà"

    def test_strips_asterisk(self):
        laws = {}
        # No laws → identity; asterisk stripped
        assert predict_form("*doruma", "lang2", laws) == "doruma"

    def test_unknown_grapheme_kept_unchanged(self):
        laws = {("lang2", "a"): "à"}
        # "o" has no law → preserved
        result = predict_form("*doma", "lang2", laws)
        assert "o" in result

    def test_empty_form_after_strip(self):
        # Only asterisk
        result = predict_form("*", "lang2", {})
        assert result == ""

    def test_all_graphemes_transformed(self):
        laws = {("lang2", "a"): "à", ("lang2", "o"): "ó"}
        result = predict_form("*ao", "lang2", laws)
        assert result == "àó"

    def test_form_without_asterisk_still_works(self):
        laws = {("lang2", "a"): "à"}
        result = predict_form("ama", "lang2", laws)
        assert result == "àmà"


# ---------------------------------------------------------------------------
# predicted_coverage
# ---------------------------------------------------------------------------


class TestPredictedCoverage:
    def test_identical_forms(self):
        assert predicted_coverage("doruma", "doruma") == 1.0

    def test_empty_predicted(self):
        assert predicted_coverage("", "doruma") == 0.0

    def test_full_coverage_with_suffix(self):
        # predicted "dor" is fully embedded in "dórumali"
        cov = predicted_coverage("dor", "dórumali")
        assert cov == 1.0

    def test_partial_coverage(self):
        # "abc" in "axc" → LCS=2, 2/3
        cov = predicted_coverage("abc", "axc")
        assert abs(cov - 2 / 3) < 1e-4

    def test_zero_coverage(self):
        cov = predicted_coverage("zzz", "aaa")
        assert cov == 0.0

    def test_result_rounded_to_4(self):
        cov = predicted_coverage("abc", "abcde")
        assert len(str(cov).split(".")[-1]) <= 4

    def test_asymmetric_predicted_over_actual(self):
        # Predicted longer than actual; coverage can be < 1
        cov = predicted_coverage("dorumaguug", "doruma")
        assert cov < 1.0

    def test_diacritics_ignored_in_coverage(self):
        # "dóruma" vs "doruma" should be full coverage
        cov = predicted_coverage("dóruma", "doruma")
        assert cov == 1.0


# ---------------------------------------------------------------------------
# match_class
# ---------------------------------------------------------------------------


class TestMatchClass:
    def test_exact_at_threshold(self):
        assert match_class(EXACT_THRESHOLD) == "exact"

    def test_exact_above(self):
        assert match_class(1.0) == "exact"

    def test_close_at_threshold(self):
        assert match_class(CLOSE_THRESHOLD) == "close"

    def test_close_below_exact(self):
        assert match_class(EXACT_THRESHOLD - 0.01) == "close"

    def test_partial_at_threshold(self):
        assert match_class(PARTIAL_THRESHOLD) == "partial"

    def test_partial_below_close(self):
        assert match_class(CLOSE_THRESHOLD - 0.01) == "partial"

    def test_miss_below_partial(self):
        assert match_class(PARTIAL_THRESHOLD - 0.01) == "miss"

    def test_miss_at_zero(self):
        assert match_class(0.0) == "miss"

    @pytest.mark.parametrize("cov,expected", [
        (1.00, "exact"),
        (0.95, "exact"),
        (0.94, "close"),
        (0.75, "close"),
        (0.74, "partial"),
        (0.50, "partial"),
        (0.49, "miss"),
        (0.00, "miss"),
    ])
    def test_boundary_table(self, cov, expected):
        assert match_class(cov) == expected


# ---------------------------------------------------------------------------
# evaluate_predictions
# ---------------------------------------------------------------------------


class TestEvaluatePredictions:
    def _lexicon_row(self, english="student", proto="*doruma", lang2="doruma", **kwargs):
        row = {"english": english, "pos": "NN", "proto_form": proto,
               "lang1": "doruma", "lang2": lang2,
               "lang3": "", "lang4": "", "lang5": "", "lang6": "", "lang7": ""}
        row.update(kwargs)
        return row

    def test_empty_lexicon(self):
        assert evaluate_predictions([], {}) == []

    def test_skips_empty_actual(self):
        row = self._lexicon_row(lang2="")
        results = evaluate_predictions([row], {})
        assert results == []

    def test_produces_prediction_result(self):
        row = self._lexicon_row(lang2="doruma")
        results = evaluate_predictions([row], {})
        assert len(results) == 1
        assert isinstance(results[0], PredictionResult)

    def test_fields_populated(self):
        row = self._lexicon_row(english="student", proto="*doruma", lang2="doruma")
        results = evaluate_predictions([row], {})
        r = results[0]
        assert r.english == "student"
        assert r.proto_form == "*doruma"
        assert r.language == "lang2"
        assert r.actual_form == "doruma"

    def test_law_applied_in_prediction(self):
        row = self._lexicon_row(proto="*ama", lang2="àmà")
        laws = {("lang2", "a"): "à"}
        results = evaluate_predictions([row], laws)
        assert results[0].predicted_form == "àmà"

    def test_match_class_assigned(self):
        row = self._lexicon_row(proto="*doruma", lang2="doruma")
        results = evaluate_predictions([row], {})
        assert results[0].match_class in {"exact", "close", "partial", "miss"}

    def test_only_lang2_to_lang7_evaluated(self):
        row = self._lexicon_row(lang2="doruma")
        results = evaluate_predictions([row], {})
        for r in results:
            assert r.language != "lang1"

    def test_multiple_languages_per_entry(self):
        row = self._lexicon_row(lang2="doruma", lang3="doruma")
        results = evaluate_predictions([row], {})
        langs = {r.language for r in results}
        assert "lang2" in langs
        assert "lang3" in langs


# ---------------------------------------------------------------------------
# _first_missing_grapheme
# ---------------------------------------------------------------------------


class TestFirstMissingGrapheme:
    def test_all_matched_returns_empty(self):
        assert _first_missing_grapheme("abc", "abc") == ""

    def test_returns_first_unmatched(self):
        # "xyz" not in "abc"
        result = _first_missing_grapheme("xyz", "abc")
        assert result == "x"

    def test_empty_predicted_returns_empty(self):
        assert _first_missing_grapheme("", "abc") == ""

    def test_empty_actual_returns_first_predicted(self):
        result = _first_missing_grapheme("abc", "")
        assert result == "a"

    def test_partial_match(self):
        # "b" is matched, "z" is not
        result = _first_missing_grapheme("bz", "bcd")
        assert result == "z"

    def test_diacritics_handled(self):
        # "ó" base "o" matched with "o"
        result = _first_missing_grapheme("ó", "o")
        assert result == ""


# ---------------------------------------------------------------------------
# _diagnose
# ---------------------------------------------------------------------------


class TestDiagnose:
    def _result(self, predicted, actual, coverage=0.3):
        return make_prediction(
            predicted_form=predicted, actual_form=actual,
            predicted_coverage=coverage, match_class="miss"
        )

    def test_alignment_noise_when_actual_3x_longer(self):
        # predicted "abc" (3), actual "abcdefghij" (10) → 10 >= 3*3
        r = self._result("abc", "abcdefghij")
        assert _diagnose(r) == "alignment_noise"

    def test_conditioned_when_missing_grapheme_is_vowel(self):
        # predicted "xaz" (3), actual "xbz" (3) — missing "a" is a vowel
        r = self._result("xaz", "xbz")
        assert _diagnose(r) == "conditioned"

    def test_irregular_otherwise(self):
        # predicted "xyz" (3), actual "bbb" (3) — "x" is a consonant
        r = self._result("xyz", "bbb")
        assert _diagnose(r) == "irregular"

    def test_alignment_noise_takes_priority(self):
        # actual is 3x longer AND has vowel mismatch — noise takes priority
        r = self._result("ao", "aobcdefghijklm")
        assert _diagnose(r) == "alignment_noise"

    def test_accented_vowel_is_conditioned(self):
        # "ó" base "o" is a vowel
        r = self._result("óbc", "zbc")
        assert _diagnose(r) == "conditioned"


# ---------------------------------------------------------------------------
# analyze_exceptions
# ---------------------------------------------------------------------------


class TestAnalyzeExceptions:
    def test_only_miss_class_included(self):
        results = [
            make_prediction(match_class="exact", predicted_coverage=1.0),
            make_prediction(match_class="close", predicted_coverage=0.8),
            make_prediction(match_class="miss", predicted_coverage=0.3,
                            predicted_form="xyz", actual_form="abc"),
        ]
        exceptions = analyze_exceptions(results)
        assert len(exceptions) == 1

    def test_empty_results(self):
        assert analyze_exceptions([]) == []

    def test_no_misses_returns_empty(self):
        results = [make_prediction(match_class="exact", predicted_coverage=1.0)]
        assert analyze_exceptions(results) == []

    def test_returns_exception_records(self):
        results = [make_prediction(match_class="miss", predicted_coverage=0.2,
                                   predicted_form="xyz", actual_form="abc")]
        exceptions = analyze_exceptions(results)
        assert isinstance(exceptions[0], ExceptionRecord)

    def test_sorted_by_coverage_ascending(self):
        results = [
            make_prediction(match_class="miss", predicted_coverage=0.4,
                            predicted_form="xyz", actual_form="abc"),
            make_prediction(match_class="miss", predicted_coverage=0.1,
                            predicted_form="xyz", actual_form="abc"),
            make_prediction(match_class="miss", predicted_coverage=0.3,
                            predicted_form="xyz", actual_form="abc"),
        ]
        exceptions = analyze_exceptions(results)
        covs = [e.predicted_coverage for e in exceptions]
        assert covs == sorted(covs)

    def test_diagnosis_field_populated(self):
        results = [make_prediction(match_class="miss", predicted_coverage=0.2,
                                   predicted_form="xyz", actual_form="abc")]
        exceptions = analyze_exceptions(results)
        assert exceptions[0].diagnosis in {"alignment_noise", "conditioned", "irregular"}

    def test_mismatched_grapheme_field_populated(self):
        results = [make_prediction(match_class="miss", predicted_coverage=0.2,
                                   predicted_form="xyz", actual_form="abc")]
        exceptions = analyze_exceptions(results)
        assert exceptions[0].mismatched_grapheme in {"x", "y", "z", ""}


# ---------------------------------------------------------------------------
# _content_gloss
# ---------------------------------------------------------------------------


class TestContentGloss:
    def test_plain_content_word(self):
        assert _content_gloss("student") == "student"

    def test_content_with_suffix(self):
        assert _content_gloss("student-PL") == "student"

    def test_grammatical_prefix_skipped(self):
        assert _content_gloss("INAN-welcome-WIT") == "welcome"

    def test_purely_grammatical_returns_none(self):
        assert _content_gloss("ERG") is None
        assert _content_gloss("1SG") is None

    def test_trailing_comma_stripped(self):
        assert _content_gloss("student,") == "student"

    def test_empty_returns_none(self):
        assert _content_gloss("") is None


# ---------------------------------------------------------------------------
# _build_proto_lookup
# ---------------------------------------------------------------------------


class TestBuildProtoLookup:
    def _lexicon(self):
        return [
            {"english": "student", "proto_form": "*dóruma"},
            {"english": "to create", "proto_form": "*góruamu"},
            {"english": "point of view", "proto_form": "*xxx"},
        ]

    def test_plain_noun_indexed(self):
        lookup = _build_proto_lookup(self._lexicon())
        assert lookup["student"] == "*dóruma"

    def test_verb_to_prefix_stripped(self):
        lookup = _build_proto_lookup(self._lexicon())
        assert lookup["create"] == "*góruamu"

    def test_full_english_also_indexed(self):
        lookup = _build_proto_lookup(self._lexicon())
        assert lookup["to create"] == "*góruamu"

    def test_case_insensitive_key(self):
        lookup = _build_proto_lookup(self._lexicon())
        assert "student" in lookup

    def test_empty_lexicon(self):
        assert _build_proto_lookup([]) == {}


# ---------------------------------------------------------------------------
# generate_proto_texts
# ---------------------------------------------------------------------------


class TestGenerateProtoTexts:
    def _corpus_row(self, surface, segmented, gloss, translation="A sentence.", sid=1):
        return {"language": "lang1", "sentence_id": str(sid),
                "surface": surface, "segmented": segmented,
                "gloss": gloss, "translation": translation}

    def _lexicon(self):
        return [
            {"english": "student", "proto_form": "*dóruma"},
            {"english": "to welcome", "proto_form": "*iáramami"},
        ]

    def test_empty_corpus_returns_empty(self):
        assert generate_proto_texts([], self._lexicon()) == []

    def test_empty_lexicon_returns_empty(self):
        row = self._corpus_row("dóruma iáramami.", "dóruma iáramami", "student-ERG welcome-WIT")
        # No lexicon → no mappings → mapped < 2 → excluded
        assert generate_proto_texts([row], []) == []

    def test_row_without_gloss_skipped(self):
        row = {"language": "lang1", "sentence_id": "1", "surface": "dóruma.",
               "segmented": "", "gloss": "", "translation": "t."}
        assert generate_proto_texts([row], self._lexicon()) == []

    def test_non_lang1_rows_skipped(self):
        row = self._corpus_row("dóruma iáramami.", "dóruma iáramami", "student welcome")
        row["language"] = "lang2"
        assert generate_proto_texts([row], self._lexicon()) == []

    def test_mismatched_token_counts_skipped(self):
        # 2 surface tokens, 1 seg token → mismatch
        row = self._corpus_row("a b", "only_one", "ONE")
        assert generate_proto_texts([row], self._lexicon()) == []

    def test_requires_at_least_2_mapped(self):
        # Only 1 content word maps → excluded
        row = self._corpus_row("dóruma ERG.", "dóruma ERG", "student-ERG ERG")
        assert generate_proto_texts([row], self._lexicon()) == []

    def test_valid_row_produces_result(self):
        row = self._corpus_row(
            "dóruma iáramami.",
            "dóruma iáramami",
            "student welcome",
        )
        results = generate_proto_texts([row], self._lexicon(), n=1)
        assert len(results) == 1

    def test_result_keys(self):
        row = self._corpus_row("dóruma iáramami.", "dóruma iáramami", "student welcome")
        results = generate_proto_texts([row], self._lexicon(), n=1)
        assert set(results[0].keys()) == {"original", "proto_sentence", "translation", "coverage"}

    def test_coverage_format(self):
        row = self._corpus_row("dóruma iáramami.", "dóruma iáramami", "student welcome")
        results = generate_proto_texts([row], self._lexicon(), n=1)
        # coverage is formatted as e.g. "100%"
        assert "%" in results[0]["coverage"]

    def test_n_limits_output(self):
        rows = [
            self._corpus_row("dóruma iáramami.", "dóruma iáramami", "student welcome", sid=i)
            for i in range(10)
        ]
        results = generate_proto_texts(rows, self._lexicon(), n=3)
        assert len(results) <= 3

    def test_original_stored(self):
        row = self._corpus_row("dóruma iáramami.", "dóruma iáramami", "student welcome")
        results = generate_proto_texts([row], self._lexicon(), n=1)
        assert results[0]["original"] == "dóruma iáramami."


# ---------------------------------------------------------------------------
# compute_accuracy_stats
# ---------------------------------------------------------------------------


class TestComputeAccuracyStats:
    def test_empty_results(self):
        stats = compute_accuracy_stats([])
        assert stats["overall"]["total"] == 0
        assert stats["overall"]["exact_pct"] == 0

    def test_all_exact(self):
        results = [make_prediction(match_class="exact", predicted_coverage=1.0,
                                   language="lang2") for _ in range(4)]
        stats = compute_accuracy_stats(results)
        assert stats["overall"]["exact_pct"] == 100.0
        assert stats["overall"]["miss_pct"] == 0.0

    def test_all_miss(self):
        results = [make_prediction(match_class="miss", predicted_coverage=0.2,
                                   language="lang2") for _ in range(3)]
        stats = compute_accuracy_stats(results)
        assert stats["overall"]["miss_pct"] == 100.0

    def test_total_count(self):
        results = [make_prediction(language="lang2") for _ in range(5)]
        stats = compute_accuracy_stats(results)
        assert stats["overall"]["total"] == 5

    def test_per_language_key(self):
        results = [make_prediction(language="lang3", match_class="exact",
                                   predicted_coverage=1.0)]
        stats = compute_accuracy_stats(results)
        assert "lang3" in stats["by_language"]
        assert stats["by_language"]["lang3"]["total"] == 1

    def test_empty_language_gives_empty_dict(self):
        results = [make_prediction(language="lang2")]
        stats = compute_accuracy_stats(results)
        # lang7 has no results
        assert stats["by_language"]["lang7"] == {}

    def test_avg_coverage_rounded(self):
        results = [
            make_prediction(language="lang2", predicted_coverage=0.8),
            make_prediction(language="lang2", predicted_coverage=0.6),
        ]
        stats = compute_accuracy_stats(results)
        assert stats["overall"]["avg_coverage"] == round(0.7, 3)

    def test_close_or_better_includes_exact(self):
        results = [
            make_prediction(language="lang2", match_class="exact", predicted_coverage=1.0),
            make_prediction(language="lang2", match_class="close", predicted_coverage=0.8),
            make_prediction(language="lang2", match_class="miss", predicted_coverage=0.2),
        ]
        stats = compute_accuracy_stats(results)
        assert abs(stats["overall"]["close_or_better_pct"] - 200 / 3) < 0.1


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def _inputs(self):
        results = [
            make_prediction(match_class="exact", predicted_coverage=1.0, language="lang2"),
            make_prediction(match_class="miss", predicted_coverage=0.2,
                            predicted_form="xyz", actual_form="abc", language="lang2"),
        ]
        exceptions = [make_exception()]
        proto_texts = [{"original": "surface.", "proto_sentence": "*form.",
                        "translation": "Translation.", "coverage": "50%"}]
        stats = compute_accuracy_stats(results)
        return results, exceptions, proto_texts, stats

    def test_returns_string(self):
        r, e, p, s = self._inputs()
        assert isinstance(generate_report(r, e, p, s), str)

    def test_ends_with_newline(self):
        r, e, p, s = self._inputs()
        assert generate_report(r, e, p, s).endswith("\n")

    def test_overall_accuracy_section(self):
        r, e, p, s = self._inputs()
        assert "## 1. Overall Accuracy" in generate_report(r, e, p, s)

    def test_per_language_section(self):
        r, e, p, s = self._inputs()
        assert "## 2. Per-Language Accuracy" in generate_report(r, e, p, s)

    def test_best_predictions_section(self):
        r, e, p, s = self._inputs()
        assert "## 3. Best Predictions" in generate_report(r, e, p, s)

    def test_worst_predictions_section(self):
        r, e, p, s = self._inputs()
        assert "## 4. Worst Predictions" in generate_report(r, e, p, s)

    def test_exception_analysis_section(self):
        r, e, p, s = self._inputs()
        assert "## 5. Exception Analysis" in generate_report(r, e, p, s)

    def test_proto_sentences_section(self):
        r, e, p, s = self._inputs()
        assert "## 6. Sample Proto-Language Sentences" in generate_report(r, e, p, s)

    def test_overall_assessment_section(self):
        r, e, p, s = self._inputs()
        assert "## 7. Overall Assessment" in generate_report(r, e, p, s)

    def test_proto_text_appears_in_report(self):
        r, e, p, s = self._inputs()
        report = generate_report(r, e, p, s)
        assert "*form." in report

    def test_empty_results_no_crash(self):
        stats = compute_accuracy_stats([])
        report = generate_report([], [], [], stats)
        assert isinstance(report, str)


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_writes_prediction_records(self, tmp_path):
        record = make_prediction()
        out = tmp_path / "predictions.csv"
        export_csv([record], out)
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 1
        assert rows[0]["english"] == "student"
        assert rows[0]["match_class"] == "exact"

    def test_writes_exception_records(self, tmp_path):
        record = make_exception()
        out = tmp_path / "exceptions.csv"
        export_csv([record], out)
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 1
        assert rows[0]["diagnosis"] == "irregular"

    def test_empty_list_no_file(self, tmp_path):
        out = tmp_path / "empty.csv"
        export_csv([], out)
        assert not out.exists()

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "a" / "b" / "out.csv"
        export_csv([make_prediction()], out)
        assert out.exists()

    def test_unicode_in_forms(self, tmp_path):
        record = make_prediction(proto_form="*dóruma", predicted_form="þàdórúmà")
        out = tmp_path / "out.csv"
        export_csv([record], out)
        content = out.read_text(encoding="utf-8")
        assert "dóruma" in content
        assert "þàdórúmà" in content

    def test_prediction_header(self, tmp_path):
        out = tmp_path / "predictions.csv"
        export_csv([make_prediction()], out)
        header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == [
            "english", "pos", "proto_form", "language",
            "predicted_form", "actual_form", "raw_similarity",
            "predicted_coverage", "match_class"
        ]

    def test_exception_header(self, tmp_path):
        out = tmp_path / "exceptions.csv"
        export_csv([make_exception()], out)
        header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == [
            "english", "pos", "proto_form", "language",
            "predicted_form", "actual_form", "predicted_coverage",
            "mismatched_grapheme", "diagnosis"
        ]


# ---------------------------------------------------------------------------
# run (end-to-end with minimal fixture data)
# ---------------------------------------------------------------------------


class TestRun:
    def _write_minimal(self, tmp_path: Path):
        write_csv(tmp_path / "sound_laws.csv", [
            make_laws_row(language="lang2", proto_grapheme="a", reflex="à",
                          regularity="0.9", count="10"),
        ])
        write_csv(tmp_path / "proto_lexicon.csv", [
            {"english": "student", "pos": "NN", "proto_form": "*doruma",
             "confidence": "high", "n_cognates": "1",
             "lang1": "doruma", "lang2": "doruma", "lang3": "", "lang4": "",
             "lang5": "", "lang6": "", "lang7": ""},
        ])
        write_csv(tmp_path / "cognate_sets.csv", [
            {"english": "student", "lang1": "doruma", "lang2": "doruma",
             "lang3": "", "lang4": "", "lang5": "", "lang6": "", "lang7": ""},
        ])
        write_csv(tmp_path / "corpus.csv", [
            {"language": "lang1", "sentence_id": "1",
             "surface": "doruma welcoma.", "segmented": "doruma welcoma",
             "gloss": "student welcome", "translation": "The student welcomes."},
        ])

    def test_returns_expected_keys(self, tmp_path):
        self._write_minimal(tmp_path)
        result = run(processed_dir=tmp_path)
        assert set(result.keys()) == {"results", "exceptions", "proto_texts", "stats"}

    def test_results_are_prediction_instances(self, tmp_path):
        self._write_minimal(tmp_path)
        result = run(processed_dir=tmp_path)
        for r in result["results"]:
            assert isinstance(r, PredictionResult)

    def test_exceptions_are_exception_records(self, tmp_path):
        self._write_minimal(tmp_path)
        result = run(processed_dir=tmp_path)
        for e in result["exceptions"]:
            assert isinstance(e, ExceptionRecord)

    def test_output_files_created(self, tmp_path):
        self._write_minimal(tmp_path)
        run(processed_dir=tmp_path)
        assert (tmp_path / "validation_report.md").exists()

    def test_validation_report_always_written(self, tmp_path):
        # Even with no data, the report file is created
        run(processed_dir=tmp_path)
        assert (tmp_path / "validation_report.md").exists()

    def test_empty_dir_no_crash(self, tmp_path):
        result = run(processed_dir=tmp_path)
        assert result["results"] == []
        assert result["exceptions"] == []

    def test_predictions_csv_created_when_nonempty(self, tmp_path):
        self._write_minimal(tmp_path)
        run(processed_dir=tmp_path)
        assert (tmp_path / "predictions.csv").exists()

    def test_stats_has_overall_key(self, tmp_path):
        self._write_minimal(tmp_path)
        result = run(processed_dir=tmp_path)
        assert "overall" in result["stats"]
        assert "by_language" in result["stats"]


# ---------------------------------------------------------------------------
# Integration tests (require actual processed data)
# ---------------------------------------------------------------------------


@integration
class TestIntegration:
    @pytest.fixture(scope="class")
    def results(self):
        return run(processed_dir=PROCESSED_DIR)

    def test_results_nonempty(self, results):
        assert len(results["results"]) > 0

    def test_all_match_classes_valid(self, results):
        valid = {"exact", "close", "partial", "miss"}
        for r in results["results"]:
            assert r.match_class in valid

    def test_coverages_in_range(self, results):
        for r in results["results"]:
            assert 0.0 <= r.predicted_coverage <= 1.0

    def test_similarities_in_range(self, results):
        for r in results["results"]:
            assert 0.0 <= r.raw_similarity <= 1.0

    def test_languages_valid(self, results):
        for r in results["results"]:
            assert r.language in LANGUAGES[1:]

    def test_exceptions_are_all_miss(self, results):
        for e in results["exceptions"]:
            assert e.diagnosis in {"alignment_noise", "conditioned", "irregular"}

    def test_exceptions_sorted_ascending(self, results):
        covs = [e.predicted_coverage for e in results["exceptions"]]
        assert covs == sorted(covs)

    def test_stats_total_matches_results(self, results):
        assert results["stats"]["overall"]["total"] == len(results["results"])

    def test_predictions_csv_readable(self):
        path = PROCESSED_DIR / "predictions.csv"
        if not path.exists():
            pytest.skip("predictions.csv not yet generated")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) > 0
        assert "match_class" in rows[0]
        assert "predicted_coverage" in rows[0]

    def test_exceptions_csv_readable(self):
        path = PROCESSED_DIR / "exceptions.csv"
        if not path.exists():
            pytest.skip("exceptions.csv not yet generated")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert "diagnosis" in rows[0]

    def test_validation_report_md_exists(self):
        path = PROCESSED_DIR / "validation_report.md"
        if not path.exists():
            pytest.skip("validation_report.md not yet generated")
        content = path.read_text(encoding="utf-8")
        assert "# Validation Report" in content

    def test_overall_close_or_better_above_50pct(self, results):
        # A basic sanity check: the reconstruction should be >50% close-or-better
        pct = results["stats"]["overall"]["close_or_better_pct"]
        assert pct > 50.0, f"close_or_better_pct unexpectedly low: {pct}"
