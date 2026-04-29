"""
Tests for src/morphology.py.

Unit tests use inline fixtures and tmp_path so they do not depend on real data.
Integration tests are marked @pytest.mark.integration and skipped when the
processed data directory is absent.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from src.morphology import (
    CATEGORY_MAP,
    CROSS_LANG_PATTERNS,
    LANGUAGES,
    MIN_FREQUENCY,
    PROCESSED_DIR,
    MorphemeEntry,
    _is_gram,
    _table,
    build_inventory,
    check_cross_language,
    export_csv,
    extract_morphemes,
    generate_grammar,
    run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROCESSED_DATA_AVAILABLE = (PROCESSED_DIR / "corpus.csv").exists()
integration = pytest.mark.skipif(
    not PROCESSED_DATA_AVAILABLE, reason="processed data directory not found"
)


def write_corpus_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["language", "sentence_id", "surface", "segmented", "gloss", "translation"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full = {f: row.get(f, "") for f in fieldnames}
            writer.writerow(full)


def lang1_row(surface, segmented, gloss, translation="A sentence.", sid=1):
    return {
        "language": "lang1", "sentence_id": str(sid),
        "surface": surface, "segmented": segmented,
        "gloss": gloss, "translation": translation,
    }


def make_entry(
    gloss_tag="WIT",
    canonical_form="mi",
    category="evidentiality",
    frequency=10,
    n_forms=2,
    all_forms="mi(8)  mí(2)",
    cross_lang_attested=True,
    attested_in="lang2, lang3",
    example_word="iáramami",
) -> MorphemeEntry:
    return MorphemeEntry(
        gloss_tag=gloss_tag,
        canonical_form=canonical_form,
        category=category,
        frequency=frequency,
        n_forms=n_forms,
        all_forms=all_forms,
        cross_lang_attested=cross_lang_attested,
        attested_in=attested_in,
        example_word=example_word,
    )


# ---------------------------------------------------------------------------
# _is_gram
# ---------------------------------------------------------------------------


class TestIsGram:
    def test_standard_case_tag(self):
        assert _is_gram("ERG") is True

    def test_evidentiality_tag(self):
        assert _is_gram("WIT") is True

    def test_dotted_tag(self):
        assert _is_gram("DEF.NEAR") is True

    def test_number_tag(self):
        assert _is_gram("PL") is True

    def test_lowercase_content_word(self):
        assert _is_gram("student") is False

    def test_mixed_case_false(self):
        assert _is_gram("Erg") is False

    def test_empty_string_false(self):
        assert _is_gram("") is False

    def test_single_char_false(self):
        assert _is_gram("E") is False

    def test_starts_with_digit_false(self):
        # "1SG".upper() == "1SG" but starts with digit
        assert _is_gram("1SG") is False

    def test_two_char_tag_true(self):
        assert _is_gram("PL") is True

    def test_all_dots_stripped_before_check(self):
        # ".ERG." → strip(".") → "ERG" → True
        assert _is_gram(".ERG.") is True

    def test_only_dots_false(self):
        # "..." → strip(".") → "" → len 0 → False
        assert _is_gram("...") is False

    @pytest.mark.parametrize("tag", [
        t for t in CATEGORY_MAP.keys() if not t[0].isdigit()
    ])
    def test_non_digit_category_map_keys_are_gram(self, tag):
        # All CATEGORY_MAP keys that don't start with a digit should pass _is_gram
        assert _is_gram(tag) is True, f"Expected _is_gram({tag!r}) to be True"

    @pytest.mark.parametrize("tag", [
        t for t in CATEGORY_MAP.keys() if t[0].isdigit()
    ])
    def test_digit_starting_person_tags_fail_is_gram(self, tag):
        # Person tags like "1SG", "2PL" start with a digit → _is_gram returns False
        assert _is_gram(tag) is False, f"Expected _is_gram({tag!r}) to be False"


# ---------------------------------------------------------------------------
# extract_morphemes
# ---------------------------------------------------------------------------


class TestExtractMorphemes:
    def test_empty_input(self):
        forms, examples, pairs = extract_morphemes([])
        assert forms == {}
        assert examples == {}
        assert len(pairs) == 0

    def test_basic_grammatical_tag_collected(self):
        row = lang1_row(
            "iáramami.",
            "i-árama-mi",
            "INAN-welcome-WIT",
        )
        forms, _, _ = extract_morphemes([row])
        assert "WIT" in forms
        assert "mi" in forms["WIT"]
        assert "INAN" in forms

    def test_content_words_not_in_forms(self):
        row = lang1_row(
            "dóruma.",
            "dóruma",
            "student",
        )
        forms, _, _ = extract_morphemes([row])
        # "student" is not a grammatical tag
        assert "student" not in forms

    def test_pair_counts_all_pairs(self):
        row = lang1_row(
            "iáramami.",
            "i-árama-mi",
            "INAN-welcome-WIT",
        )
        _, _, pairs = extract_morphemes([row])
        assert pairs[("i", "INAN")] == 1
        assert pairs[("árama", "welcome")] == 1
        assert pairs[("mi", "WIT")] == 1

    def test_counts_accumulate_across_rows(self):
        rows = [
            lang1_row("iáramami.", "i-árama-mi", "INAN-welcome-WIT", sid=1),
            lang1_row("iáramami.", "i-árama-mi", "INAN-welcome-WIT", sid=2),
        ]
        forms, _, _ = extract_morphemes(rows)
        assert forms["WIT"]["mi"] == 2

    def test_examples_capped_at_5(self):
        rows = [
            lang1_row(f"iáramami.", "i-árama-mi", "INAN-welcome-WIT", sid=i)
            for i in range(10)
        ]
        _, examples, _ = extract_morphemes(rows)
        assert len(examples.get("WIT", [])) <= 5

    def test_trailing_comma_stripped_from_tokens(self):
        row = lang1_row(
            "dóruma iáramami.",
            "dóruma, i-árama-mi",
            "student, INAN-welcome-WIT",
        )
        forms, _, _ = extract_morphemes([row])
        # "mi" should still be found after comma stripping
        assert "WIT" in forms

    def test_dots_stripped_from_morphemes(self):
        # ".ERG." should be cleaned to "ERG"
        row = lang1_row("word.", "word-.ERG.", "content-.ERG.")
        forms, _, _ = extract_morphemes([row])
        assert "ERG" in forms

    def test_mismatched_seg_gls_zips_to_shorter(self):
        # seg has 2 tokens, gls has 1 token → only 1 pair processed
        row = lang1_row("a b.", "a-ERG b-WIT", "content-ERG")
        forms, _, _ = extract_morphemes([row])
        # WIT should not be found since gls only has "content-ERG"
        assert "WIT" not in forms

    def test_example_word_is_surface_token(self):
        row = lang1_row(
            "SURFACEWORD.",
            "root-ERG",
            "content-ERG",
        )
        _, examples, _ = extract_morphemes([row])
        assert examples.get("ERG", [""])[0] == "SURFACEWORD."

    def test_multiple_morphemes_per_token(self):
        row = lang1_row(
            "dórumakoli.",
            "dóruma-ko-li",
            "student-DEF.NEAR-ELAT",
        )
        forms, _, _ = extract_morphemes([row])
        assert "DEF.NEAR" in forms
        assert "ELAT" in forms


# ---------------------------------------------------------------------------
# check_cross_language
# ---------------------------------------------------------------------------


class TestCheckCrossLanguage:
    def _corpus(self, lang: str, tokens: list[str]) -> dict[str, list[dict]]:
        rows = [{"language": lang, "sentence_id": str(i + 1),
                 "surface": tok, "segmented": "", "gloss": "", "translation": "t."}
                for i, tok in enumerate(tokens)]
        return {lang: rows}

    def test_empty_corpus_returns_empty(self):
        assert check_cross_language({}) == {}

    def test_lang1_not_searched(self):
        # Even if lang1 has matching tokens, it should not be counted
        corpus = {"lang1": [
            {"language": "lang1", "sentence_id": "1",
             "surface": "imi", "segmented": "", "gloss": "", "translation": "t."}
        ] * 10}
        result = check_cross_language(corpus)
        for langs in result.values():
            assert "lang1" not in langs

    def test_wit_pattern_detected(self):
        # WIT pattern: suffix r"m[iíĩ]$"  — need >= 5 matches
        corpus = {
            "lang2": [
                {"language": "lang2", "sentence_id": str(i), "surface": "wordmi",
                 "segmented": "", "gloss": "", "translation": "t."}
                for i in range(10)
            ]
        }
        result = check_cross_language(corpus)
        assert "lang2" in result.get("WIT", [])

    def test_fewer_than_5_matches_not_attested(self):
        corpus = {
            "lang2": [
                {"language": "lang2", "sentence_id": str(i), "surface": "wordmi",
                 "segmented": "", "gloss": "", "translation": "t."}
                for i in range(3)  # only 3 matches
            ]
        }
        result = check_cross_language(corpus)
        assert "lang2" not in result.get("WIT", [])

    def test_exactly_5_matches_is_attested(self):
        corpus = {
            "lang2": [
                {"language": "lang2", "sentence_id": str(i), "surface": "wordmi",
                 "segmented": "", "gloss": "", "translation": "t."}
                for i in range(5)
            ]
        }
        result = check_cross_language(corpus)
        assert "lang2" in result.get("WIT", [])

    def test_inan_prefix_pattern_detected(self):
        # INAN pattern: prefix r"^[iíĩ]"
        corpus = {
            "lang3": [
                {"language": "lang3", "sentence_id": str(i), "surface": "iwordfoo",
                 "segmented": "", "gloss": "", "translation": "t."}
                for i in range(10)
            ]
        }
        result = check_cross_language(corpus)
        assert "lang3" in result.get("INAN", [])

    def test_punctuation_stripped_before_matching(self):
        # token "wordmi." should still match the WIT suffix
        corpus = {
            "lang2": [
                {"language": "lang2", "sentence_id": str(i), "surface": "wordmi.",
                 "segmented": "", "gloss": "", "translation": "t."}
                for i in range(10)
            ]
        }
        result = check_cross_language(corpus)
        assert "lang2" in result.get("WIT", [])

    def test_multiple_languages_independent(self):
        row = {"language": "lang2", "sentence_id": "1", "surface": "wordmi",
               "segmented": "", "gloss": "", "translation": "t."}
        corpus = {"lang2": [row] * 10, "lang3": []}
        result = check_cross_language(corpus)
        assert "lang2" in result.get("WIT", [])
        assert "lang3" not in result.get("WIT", [])


# ---------------------------------------------------------------------------
# build_inventory
# ---------------------------------------------------------------------------


class TestBuildInventory:
    def _inputs(self):
        tag_to_forms = {
            "WIT": Counter({"mi": 8, "mí": 2}),
            "ERG": Counter({"o": 5}),
            "RARE": Counter({"x": 1}),
        }
        tag_to_examples = {
            "WIT": ["wordmi"],
            "ERG": ["wordo"],
            "RARE": ["wordx"],
        }
        cross_lang = {"WIT": ["lang2", "lang3"]}
        return tag_to_forms, tag_to_examples, cross_lang

    def test_returns_morpheme_entry_instances(self):
        forms, examples, cross = self._inputs()
        entries = build_inventory(forms, examples, cross, min_freq=1)
        assert all(isinstance(e, MorphemeEntry) for e in entries)

    def test_below_min_freq_excluded(self):
        forms, examples, cross = self._inputs()
        entries = build_inventory(forms, examples, cross, min_freq=2)
        tags = {e.gloss_tag for e in entries}
        assert "RARE" not in tags   # only 1 occurrence

    def test_at_min_freq_included(self):
        forms = {"WIT": Counter({"mi": 2})}
        entries = build_inventory(forms, {}, {}, min_freq=2)
        assert len(entries) == 1

    def test_canonical_form_is_most_common(self):
        forms, examples, cross = self._inputs()
        entries = build_inventory(forms, examples, cross, min_freq=1)
        wit = next(e for e in entries if e.gloss_tag == "WIT")
        assert wit.canonical_form == "mi"

    def test_frequency_is_total_count(self):
        forms, examples, cross = self._inputs()
        entries = build_inventory(forms, examples, cross, min_freq=1)
        wit = next(e for e in entries if e.gloss_tag == "WIT")
        assert wit.frequency == 10

    def test_n_forms_count(self):
        forms, examples, cross = self._inputs()
        entries = build_inventory(forms, examples, cross, min_freq=1)
        wit = next(e for e in entries if e.gloss_tag == "WIT")
        assert wit.n_forms == 2

    def test_all_forms_format(self):
        forms, examples, cross = self._inputs()
        entries = build_inventory(forms, examples, cross, min_freq=1)
        wit = next(e for e in entries if e.gloss_tag == "WIT")
        assert "mi(8)" in wit.all_forms
        assert "mí(2)" in wit.all_forms

    def test_category_from_category_map(self):
        forms = {"WIT": Counter({"mi": 5})}
        entries = build_inventory(forms, {}, {}, min_freq=1)
        assert entries[0].category == "evidentiality"

    def test_unknown_tag_gets_unknown_category(self):
        forms = {"CUSTOM": Counter({"x": 5})}
        entries = build_inventory(forms, {}, {}, min_freq=1)
        assert entries[0].category == "unknown"

    def test_cross_lang_attested_true(self):
        forms, examples, cross = self._inputs()
        entries = build_inventory(forms, examples, cross, min_freq=1)
        wit = next(e for e in entries if e.gloss_tag == "WIT")
        assert wit.cross_lang_attested is True

    def test_cross_lang_attested_false_when_absent(self):
        forms = {"ERG": Counter({"o": 5})}
        entries = build_inventory(forms, {}, {}, min_freq=1)
        assert entries[0].cross_lang_attested is False

    def test_attested_in_sorted_and_joined(self):
        forms = {"WIT": Counter({"mi": 5})}
        cross = {"WIT": ["lang3", "lang2"]}
        entries = build_inventory(forms, {}, cross, min_freq=1)
        assert entries[0].attested_in == "lang2, lang3"

    def test_attested_in_empty_when_no_cross(self):
        forms = {"ERG": Counter({"o": 5})}
        entries = build_inventory(forms, {}, {}, min_freq=1)
        assert entries[0].attested_in == ""

    def test_sorted_by_frequency_descending(self):
        forms = {
            "ERG": Counter({"o": 3}),
            "WIT": Counter({"mi": 10}),
            "NEG": Counter({"ne": 5}),
        }
        entries = build_inventory(forms, {}, {}, min_freq=1)
        freqs = [e.frequency for e in entries]
        assert freqs == sorted(freqs, reverse=True)

    def test_example_word_from_examples(self):
        forms = {"WIT": Counter({"mi": 5})}
        examples = {"WIT": ["exampleword"]}
        entries = build_inventory(forms, examples, {}, min_freq=1)
        assert entries[0].example_word == "exampleword"

    def test_example_word_empty_when_no_examples(self):
        forms = {"WIT": Counter({"mi": 5})}
        entries = build_inventory(forms, {}, {}, min_freq=1)
        assert entries[0].example_word == ""

    def test_all_forms_capped_at_5(self):
        # More than 5 forms → only top 5 shown
        forms = {"WIT": Counter({f"form{i}": 10 - i for i in range(8)})}
        entries = build_inventory(forms, {}, {}, min_freq=1)
        assert entries[0].all_forms.count("(") <= 5


# ---------------------------------------------------------------------------
# _table
# ---------------------------------------------------------------------------


class TestTable:
    def test_returns_list_of_strings(self):
        result = _table(["A", "B"], [["1", "2"]])
        assert all(isinstance(s, str) for s in result)

    def test_first_line_is_header(self):
        result = _table(["Col1", "Col2"], [])
        assert "Col1" in result[0]
        assert "Col2" in result[0]

    def test_second_line_is_separator(self):
        result = _table(["A", "B"], [])
        assert "---" in result[1]

    def test_data_row_present(self):
        result = _table(["A", "B"], [["x", "y"]])
        assert any("x" in line and "y" in line for line in result[2:])

    def test_empty_rows_only_header_and_sep(self):
        result = _table(["H1", "H2", "H3"], [])
        assert len(result) == 2

    def test_multiple_rows(self):
        result = _table(["A"], [["1"], ["2"], ["3"]])
        # header + sep + 3 data rows = 5
        assert len(result) == 5

    def test_uses_pipe_separator(self):
        result = _table(["A", "B"], [["x", "y"]])
        assert "|" in result[0]
        assert "|" in result[2]

    def test_numeric_cells_converted(self):
        result = _table(["N"], [[42]])
        assert "42" in result[2]


# ---------------------------------------------------------------------------
# generate_grammar
# ---------------------------------------------------------------------------


class TestGenerateGrammar:
    def _inputs(self):
        tag_to_forms = {
            "WIT": Counter({"mi": 10}),
            "ERG": Counter({"o": 5}),
            "INAN": Counter({"i": 20}),
            "ANIM": Counter({"a": 15}),
            "FUT": Counter({"ke": 8}),
            "INFER": Counter({"me": 4}),
            "NEG": Counter({"ne": 6}),
            "CAUS": Counter({"su": 3}),
            "DIR.EVID": Counter({"me": 2}),
            "MIR": Counter({"ha": 2}),
        }
        tag_to_examples = {t: ["example"] for t in tag_to_forms}
        cross_lang = {"WIT": ["lang2", "lang3"], "INAN": ["lang2"]}
        inventory = build_inventory(tag_to_forms, tag_to_examples, cross_lang, min_freq=1)
        lang1_rows = [
            lang1_row(
                "dóruma iáramami.",
                "dóruma i-árama-mi",
                "student INAN-welcome-WIT",
            )
        ]
        return tag_to_forms, tag_to_examples, cross_lang, inventory, lang1_rows

    def test_returns_string(self):
        result = generate_grammar(*self._inputs())
        assert isinstance(result, str)

    def test_ends_with_newline(self):
        assert generate_grammar(*self._inputs()).endswith("\n")

    def test_main_title_present(self):
        assert "# Proto-Language Grammar" in generate_grammar(*self._inputs())

    def test_nominal_morphology_section(self):
        assert "## 1. Nominal Morphology" in generate_grammar(*self._inputs())

    def test_verbal_morphology_section(self):
        assert "## 2. Verbal Morphology" in generate_grammar(*self._inputs())

    def test_person_section(self):
        assert "## 3. Person" in generate_grammar(*self._inputs())

    def test_particles_section(self):
        assert "## 4. Particles" in generate_grammar(*self._inputs())

    def test_proto_morpheme_summary_section(self):
        assert "## 5. Proto-Morpheme Summary" in generate_grammar(*self._inputs())

    def test_sample_analysis_section(self):
        assert "## 6. Sample Morphological Analysis" in generate_grammar(*self._inputs())

    def test_limitations_section(self):
        assert "## 7. Limitations" in generate_grammar(*self._inputs())

    def test_morphological_template_present(self):
        result = generate_grammar(*self._inputs())
        assert "VERB:" in result
        assert "NOUN:" in result

    def test_cross_lang_attested_entries_in_summary(self):
        result = generate_grammar(*self._inputs())
        # WIT is cross-lang attested → should appear in proto-morpheme summary table
        assert "WIT" in result

    def test_example_sentences_from_lang1_rows(self):
        t, te, cl, inv, _ = self._inputs()
        # generate_grammar only picks rows where segmented has >= 4 tokens
        rows = [lang1_row(
            "dóruma iáramami kogo izamimi.",
            "dóruma i-árama-mi ko-ko i-sami-mi",
            "student INAN-welcome-WIT this-DEF.NEAR INAN-be-WIT",
        )]
        result = generate_grammar(t, te, cl, inv, rows)
        assert "dóruma iáramami kogo izamimi." in result

    def test_empty_lang1_rows_no_crash(self):
        t, te, cl, inv, _ = self._inputs()
        result = generate_grammar(t, te, cl, inv, [])
        assert isinstance(result, str)

    def test_empty_inventory_no_crash(self):
        t, te, cl, _, rows = self._inputs()
        result = generate_grammar(t, te, cl, [], rows)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_writes_morpheme_entry(self, tmp_path):
        record = make_entry()
        out = tmp_path / "inventory.csv"
        export_csv([record], out)
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 1
        assert rows[0]["gloss_tag"] == "WIT"
        assert rows[0]["canonical_form"] == "mi"

    def test_empty_list_no_file(self, tmp_path):
        out = tmp_path / "empty.csv"
        export_csv([], out)
        assert not out.exists()

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "a" / "b" / "inventory.csv"
        export_csv([make_entry()], out)
        assert out.exists()

    def test_header_matches_dataclass_fields(self, tmp_path):
        out = tmp_path / "inventory.csv"
        export_csv([make_entry()], out)
        header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == [
            "gloss_tag", "canonical_form", "category", "frequency",
            "n_forms", "all_forms", "cross_lang_attested", "attested_in", "example_word"
        ]

    def test_unicode_preserved(self, tmp_path):
        record = make_entry(canonical_form="mĩ́", example_word="iáramami")
        out = tmp_path / "unicode.csv"
        export_csv([record], out)
        content = out.read_text(encoding="utf-8")
        assert "mĩ́" in content
        assert "iáramami" in content

    def test_multiple_records_written(self, tmp_path):
        records = [make_entry(gloss_tag="WIT"), make_entry(gloss_tag="ERG", category="case")]
        out = tmp_path / "multi.csv"
        export_csv(records, out)
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# run (end-to-end with minimal fixture data)
# ---------------------------------------------------------------------------


class TestRun:
    def _write_corpus(self, tmp_path: Path):
        rows = [
            # lang1 rows with segmented+gloss
            {**lang1_row("dóruma iáramami.", "dóruma i-árama-mi",
                         "student INAN-welcome-WIT", sid=1)},
            {**lang1_row("dóruma iáramami.", "dóruma i-árama-mi",
                         "student INAN-welcome-WIT", sid=2)},
            {**lang1_row("kogo izamimi.", "ko-ko i-sami-mi",
                         "this-DEF.NEAR INAN-be-WIT", sid=3)},
            # lang2 rows with tokens that match WIT pattern
            *[{"language": "lang2", "sentence_id": str(10 + i), "surface": f"wordmi{i}",
               "segmented": "", "gloss": "", "translation": "t."}
              for i in range(10)],
        ]
        write_corpus_csv(tmp_path / "corpus.csv", rows)

    def test_missing_corpus_returns_empty(self, tmp_path):
        result = run(processed_dir=tmp_path)
        assert result["inventory"] == []
        assert result["grammar"] == ""

    def test_returns_expected_keys(self, tmp_path):
        self._write_corpus(tmp_path)
        result = run(processed_dir=tmp_path)
        assert set(result.keys()) == {"inventory", "grammar"}

    def test_inventory_contains_morpheme_entries(self, tmp_path):
        self._write_corpus(tmp_path)
        result = run(processed_dir=tmp_path)
        for e in result["inventory"]:
            assert isinstance(e, MorphemeEntry)

    def test_grammar_is_string(self, tmp_path):
        self._write_corpus(tmp_path)
        result = run(processed_dir=tmp_path)
        assert isinstance(result["grammar"], str)

    def test_morpheme_inventory_csv_created(self, tmp_path):
        self._write_corpus(tmp_path)
        run(processed_dir=tmp_path)
        # File created only if inventory is non-empty
        inv = run(processed_dir=tmp_path)
        if inv["inventory"]:
            assert (tmp_path / "morpheme_inventory.csv").exists()

    def test_proto_grammar_md_created(self, tmp_path):
        self._write_corpus(tmp_path)
        run(processed_dir=tmp_path)
        assert (tmp_path / "proto_grammar.md").exists()

    def test_grammar_contains_title(self, tmp_path):
        self._write_corpus(tmp_path)
        result = run(processed_dir=tmp_path)
        assert "# Proto-Language Grammar" in result["grammar"]

    def test_wit_tag_in_inventory_when_sufficient_frequency(self, tmp_path):
        self._write_corpus(tmp_path)
        result = run(processed_dir=tmp_path)
        tags = {e.gloss_tag for e in result["inventory"]}
        assert "WIT" in tags

    def test_inventory_entries_above_min_frequency(self, tmp_path):
        self._write_corpus(tmp_path)
        result = run(processed_dir=tmp_path)
        for e in result["inventory"]:
            assert e.frequency >= MIN_FREQUENCY


# ---------------------------------------------------------------------------
# CATEGORY_MAP spot-checks
# ---------------------------------------------------------------------------


class TestCategoryMap:
    @pytest.mark.parametrize("tag,expected", [
        ("ERG", "case"),
        ("ABS", "case"),
        ("ELAT", "case"),
        ("ILL", "case"),
        ("DEF", "definiteness"),
        ("DEF.NEAR", "definiteness"),
        ("PL", "number"),
        ("SG", "number"),
        ("WIT", "evidentiality"),
        ("INFER", "evidentiality"),
        ("FUT", "tense"),
        ("MIR", "mirative"),
        ("CAUS", "valence"),
        ("NEG", "negation"),
        ("INAN", "agreement"),
        ("ANIM", "agreement"),
        ("1SG", "person"),
        ("ADJ", "derivation"),
        ("INSTR", "adposition"),
    ])
    def test_known_category(self, tag, expected):
        assert CATEGORY_MAP[tag] == expected

    def test_unknown_tag_not_in_map(self):
        assert "CUSTOM_TAG" not in CATEGORY_MAP


# ---------------------------------------------------------------------------
# CROSS_LANG_PATTERNS
# ---------------------------------------------------------------------------


class TestCrossLangPatterns:
    def test_all_entries_have_three_elements(self):
        for entry in CROSS_LANG_PATTERNS:
            assert len(entry) == 3

    def test_positions_are_valid(self):
        valid_positions = {"suffix", "prefix", "infix"}
        for _tag, position, _pattern in CROSS_LANG_PATTERNS:
            assert position in valid_positions

    def test_wit_pattern_present(self):
        tags = [t for t, _, _ in CROSS_LANG_PATTERNS]
        assert "WIT" in tags

    def test_inan_pattern_present(self):
        tags = [t for t, _, _ in CROSS_LANG_PATTERNS]
        assert "INAN" in tags


# ---------------------------------------------------------------------------
# Integration tests (require actual processed data)
# ---------------------------------------------------------------------------


@integration
class TestIntegration:
    @pytest.fixture(scope="class")
    def results(self):
        return run(processed_dir=PROCESSED_DIR)

    def test_inventory_nonempty(self, results):
        assert len(results["inventory"]) > 0

    def test_grammar_nonempty(self, results):
        assert len(results["grammar"]) > 0

    def test_wit_in_inventory(self, results):
        tags = {e.gloss_tag for e in results["inventory"]}
        assert "WIT" in tags, "WIT should be in inventory"

    def test_inan_in_inventory(self, results):
        tags = {e.gloss_tag for e in results["inventory"]}
        assert "INAN" in tags

    def test_all_frequencies_above_min(self, results):
        for e in results["inventory"]:
            assert e.frequency >= MIN_FREQUENCY

    def test_categories_valid(self, results):
        valid = set(CATEGORY_MAP.values()) | {"unknown", "lexical"}
        for e in results["inventory"]:
            assert e.category in valid, f"unknown category: {e.category!r}"

    def test_cross_lang_attested_consistent(self, results):
        for e in results["inventory"]:
            if e.attested_in:
                assert e.cross_lang_attested is True
            else:
                assert e.cross_lang_attested is False

    def test_attested_in_only_valid_languages(self, results):
        for e in results["inventory"]:
            if e.attested_in:
                for lang in e.attested_in.split(", "):
                    assert lang in LANGUAGES

    def test_grammar_contains_all_sections(self, results):
        grammar = results["grammar"]
        for section in [
            "## 1. Nominal Morphology",
            "## 2. Verbal Morphology",
            "## 3. Person",
            "## 4. Particles",
            "## 5. Proto-Morpheme Summary",
            "## 6. Sample Morphological Analysis",
            "## 7. Limitations",
        ]:
            assert section in grammar, f"missing section: {section!r}"

    def test_morpheme_inventory_csv_readable(self):
        path = PROCESSED_DIR / "morpheme_inventory.csv"
        if not path.exists():
            pytest.skip("morpheme_inventory.csv not yet generated")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) > 0
        assert "gloss_tag" in rows[0]
        assert "frequency" in rows[0]

    def test_proto_grammar_md_exists(self):
        path = PROCESSED_DIR / "proto_grammar.md"
        if not path.exists():
            pytest.skip("proto_grammar.md not yet generated")
        content = path.read_text(encoding="utf-8")
        assert "# Proto-Language Grammar" in content

    def test_wit_cross_lang_attested(self, results):
        wit = next((e for e in results["inventory"] if e.gloss_tag == "WIT"), None)
        if wit is None:
            pytest.skip("WIT not in inventory")
        assert wit.cross_lang_attested is True, "WIT should be cross-language attested"
