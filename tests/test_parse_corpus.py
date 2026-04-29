"""
Tests for src/parse_corpus.py.

Unit tests use inline fixture strings written to tmp_path so they do not
depend on the presence of the real data files.  Integration tests run
against the actual raw data and are marked with @pytest.mark.integration;
they are skipped automatically when the raw data directory is absent.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.parse_corpus import (
    LANGUAGES,
    RAW_DIR,
    CorpusEntry,
    DictionaryEntry,
    DocEntry,
    _POS_PATTERN,
    _clean_translation,
    _extract_compound,
    _parse_lang1_block,
    _parse_lang2_7_block,
    _split_blocks,
    _split_dict_line,
    export_csv,
    parse_all,
    parse_corpus,
    parse_dictionary,
    parse_doc,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RAW_DATA_AVAILABLE = RAW_DIR.exists()
integration = pytest.mark.skipif(
    not RAW_DATA_AVAILABLE, reason="raw data directory not found"
)


def write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _split_dict_line
# ---------------------------------------------------------------------------


class TestSplitDictLine:
    def test_tab_separated(self):
        assert _split_dict_line("student\tdóruma\tNN") == ["student", "dóruma", "NN"]

    def test_multi_space_separated(self):
        parts = _split_dict_line("student        dóruma        NN")
        assert parts == ["student", "dóruma", "NN"]

    def test_two_space_minimum(self):
        parts = _split_dict_line("to be  sami  VB")
        assert parts == ["to be", "sami", "VB"]

    def test_single_space_not_split(self):
        # single internal space must NOT split the field
        parts = _split_dict_line("to create\tgóruamu\tVB")
        assert parts[0] == "to create"

    # --- fallback (POS-based) ---

    def test_fallback_single_space_simple(self):
        # "ring/circle kólo NN" — no 2+ spaces anywhere
        parts = _split_dict_line("ring/circle kólo NN")
        assert parts == ["ring/circle", "kólo", "NN"]

    def test_fallback_single_space_verb(self):
        parts = _split_dict_line("judge téme NN")
        assert parts == ["judge", "téme", "NN"]

    def test_fallback_two_spaces_before_pos_only(self):
        # 2 spaces before POS, 1 space before native form
        parts = _split_dict_line("normal sólumalu  JJ")
        assert parts == ["normal", "sólumalu", "JJ"]

    def test_fallback_two_spaces_before_native_only(self):
        # 2 spaces before native form, 1 space before POS
        parts = _split_dict_line("to say/to tell  tálimi VB")
        assert parts == ["to say/to tell", "tálimi", "VB"]

    def test_fallback_compound_annotation_single_space_before_pos(self):
        # compound annotation with only 1 space before POS
        parts = _split_dict_line("to cool xázumumórumu (ie, to kill burning) VB")
        assert parts[0] == "to cool"
        assert "xázumumórumu" in parts[1]
        assert "(ie, to kill burning)" in parts[1]
        assert parts[2] == "VB"

    def test_fallback_no_pos_returns_unparsed(self):
        # entry without POS — returned as-is so caller can skip it
        parts = _split_dict_line("soul  máranivéramidágeri (ie, death-muse)")
        assert len(parts) < 3


# ---------------------------------------------------------------------------
# _extract_compound
# ---------------------------------------------------------------------------


class TestExtractCompound:
    def test_no_annotation(self):
        form, explanation = _extract_compound("dóruma")
        assert form == "dóruma"
        assert explanation == ""

    def test_with_annotation(self):
        form, explanation = _extract_compound("fálurunadóruma (ie, star-scholar)")
        assert form == "fálurunadóruma"
        assert explanation == "star-scholar"

    def test_multi_word_explanation(self):
        form, explanation = _extract_compound("múranuzólunu (ie, life time)")
        assert form == "múranuzólunu"
        assert explanation == "life time"

    def test_whitespace_around_annotation(self):
        form, explanation = _extract_compound("sómething  (ie, compound word) ")
        assert form == "sómething"
        assert explanation == "compound word"


# ---------------------------------------------------------------------------
# _clean_translation
# ---------------------------------------------------------------------------


class TestCleanTranslation:
    def test_strips_quotes(self):
        assert _clean_translation('"Hello world."') == "Hello world."

    def test_no_quotes(self):
        assert _clean_translation("The hawk chased the crow.") == "The hawk chased the crow."

    def test_strips_whitespace(self):
        assert _clean_translation('  "Spaced."  ') == "Spaced."

    def test_empty_after_stripping(self):
        assert _clean_translation('""') == ""

    def test_empty_string(self):
        assert _clean_translation("") == ""


# ---------------------------------------------------------------------------
# _split_blocks
# ---------------------------------------------------------------------------


class TestSplitBlocks:
    def test_single_blank_line(self):
        text = "block one\n\nblock two"
        blocks = _split_blocks(text)
        assert len(blocks) == 2
        assert blocks[0] == "block one"
        assert blocks[1] == "block two"

    def test_double_blank_line(self):
        text = "block one\n\n\nblock two"
        blocks = _split_blocks(text)
        assert len(blocks) == 2

    def test_leading_trailing_blanks_ignored(self):
        text = "\n\nblock one\n\nblock two\n\n"
        blocks = _split_blocks(text)
        assert len(blocks) == 2

    def test_empty_text(self):
        assert _split_blocks("") == []

    def test_whitespace_only(self):
        assert _split_blocks("   \n\n   ") == []


# ---------------------------------------------------------------------------
# _parse_lang1_block
# ---------------------------------------------------------------------------


class TestParseLang1Block:
    def _make_block(self, n_lines=4):
        lines = [
            "Mógogo línenili Aarumalu nie iáramami.",
            "mógo-ko líneni-li Aaruma-lu-ru ni-ERG i-árama-mi",
            "world-DEF.NEAR linguistics-ELAT Aaruma-ELAT-ILL 1SG-ERG INAN-welcome-WIT",
            '"(I) welcome (you) into the world of Aaruma linguistics."',
        ]
        return lines[:n_lines]

    def test_valid_block(self):
        entry = _parse_lang1_block(self._make_block(), "lang1", 1)
        assert entry is not None
        assert entry.surface == "Mógogo línenili Aarumalu nie iáramami."
        assert "mógo-ko" in entry.segmented
        assert "world-DEF.NEAR" in entry.gloss
        assert entry.translation == "(I) welcome (you) into the world of Aaruma linguistics."
        assert entry.language == "lang1"
        assert entry.sentence_id == 1

    def test_too_few_lines_returns_none(self):
        for n in (0, 1, 2, 3):
            assert _parse_lang1_block(self._make_block(n), "lang1", 1) is None

    def test_empty_translation_returns_none(self):
        lines = self._make_block()
        lines[3] = '""'
        assert _parse_lang1_block(lines, "lang1", 1) is None

    def test_sentence_id_stored(self):
        entry = _parse_lang1_block(self._make_block(), "lang1", 42)
        assert entry.sentence_id == 42


# ---------------------------------------------------------------------------
# _parse_lang2_7_block
# ---------------------------------------------------------------------------


class TestParseLang2_7Block:
    def _lines(self, translation='"The students drink coffee."'):
        return [
            "ngdórúmõ̀dórúmõ̀gòòg tìgèèd kòvúrú ínĩ́ngímĩ́mĩ́.",
            translation,
        ]

    def test_valid_block(self):
        entry = _parse_lang2_7_block(self._lines(), "lang2", 1)
        assert entry is not None
        assert "kòvúrú" in entry.surface
        assert entry.translation == "The students drink coffee."
        assert entry.segmented == ""
        assert entry.gloss == ""
        assert entry.language == "lang2"

    def test_unquoted_translation(self):
        entry = _parse_lang2_7_block(self._lines("The hawk chased the crow."), "lang3", 1)
        assert entry is not None
        assert entry.translation == "The hawk chased the crow."

    def test_one_line_returns_none(self):
        assert _parse_lang2_7_block(["only surface"], "lang2", 1) is None

    def test_empty_translation_returns_none(self):
        assert _parse_lang2_7_block(self._lines('""'), "lang2", 1) is None

    def test_sentence_id_stored(self):
        entry = _parse_lang2_7_block(self._lines(), "lang5", 7)
        assert entry.sentence_id == 7


# ---------------------------------------------------------------------------
# parse_dictionary
# ---------------------------------------------------------------------------


DICT_FIXTURE = """\
student\tdóruma\tNN
to create\tgóruamu\tVB
astronomer\tfálurunadóruma (ie, star-scholar)\tNN
personal time\tmúranuzólunu (ie, life time)\tNN
\t
"""


class TestParseDictionary:
    def test_basic_entries(self, tmp_path):
        p = write(tmp_path, "dict.txt", DICT_FIXTURE)
        entries = parse_dictionary(p)
        assert len(entries) == 4

    def test_simple_entry_fields(self, tmp_path):
        p = write(tmp_path, "dict.txt", DICT_FIXTURE)
        entry = parse_dictionary(p)[0]
        assert entry.english == "student"
        assert entry.native_form == "dóruma"
        assert entry.compound_explanation == ""
        assert entry.pos == "NN"

    def test_compound_entry(self, tmp_path):
        p = write(tmp_path, "dict.txt", DICT_FIXTURE)
        entry = parse_dictionary(p)[2]
        assert entry.english == "astronomer"
        assert entry.native_form == "fálurunadóruma"
        assert entry.compound_explanation == "star-scholar"
        assert entry.pos == "NN"

    def test_multi_word_compound(self, tmp_path):
        p = write(tmp_path, "dict.txt", DICT_FIXTURE)
        entry = parse_dictionary(p)[3]
        assert entry.compound_explanation == "life time"

    def test_blank_lines_skipped(self, tmp_path):
        p = write(tmp_path, "dict.txt", "\n\nstudent\tdóruma\tNN\n\n")
        assert len(parse_dictionary(p)) == 1

    def test_incomplete_line_skipped(self, tmp_path):
        content = "only_one_field\n"
        p = write(tmp_path, "dict.txt", content)
        assert parse_dictionary(p) == []

    def test_space_separated_fallback(self, tmp_path):
        content = "student        dóruma        NN\n"
        p = write(tmp_path, "dict.txt", content)
        entries = parse_dictionary(p)
        assert len(entries) == 1
        assert entries[0].english == "student"
        assert entries[0].native_form == "dóruma"


# ---------------------------------------------------------------------------
# parse_corpus
# ---------------------------------------------------------------------------


LANG1_CORPUS = """\
Mógogo línenili Aarumalu nie iáramami.
mógo-ko líneni-li Aaruma-lu-ru ni-ERG i-árama-mi
world-DEF.NEAR linguistics-ELAT Aaruma-ELAT-ILL 1SG-ERG INAN-welcome-WIT
"(I) welcome (you) into the world of Aaruma linguistics."


Kogo dénilili hírinili Aarumalu izamimi.
ko-ko dénili-li hírini-li Aaruma-lu i-sami-mi
this documentation-ELAT history-ELAT Aaruma-ELAT INAN-be-WIT
"This is the documentation of the history of Aaruma."
"""

LANG2_CORPUS = """\
Lííngírìù tóràlúnàrúmàgà nàrúmànàrúmàlú púúrúmúnú llímíréngàmí ídàlímíngíngímí.
"Who made the actor tell the secret message to the people?"

ħħàrúmúshkùùg màrúnú élìlí mìlímìlí úbúúrúmàmú.
The hawk hid the swan's feathers in the valley.

ngélírí-kù-ì lédè-lí púúdòngò mõ̀rúgúlú í-nẽ́mĩ́-hú-mĩ́
"""


class TestParseCorpus:
    def test_lang1_sentence_count(self, tmp_path):
        p = write(tmp_path, "lang1-corpus.txt", LANG1_CORPUS)
        entries = parse_corpus(p, "lang1")
        assert len(entries) == 2

    def test_lang1_fields_populated(self, tmp_path):
        p = write(tmp_path, "lang1-corpus.txt", LANG1_CORPUS)
        entry = parse_corpus(p, "lang1")[0]
        assert entry.language == "lang1"
        assert entry.sentence_id == 1
        assert "Mógogo" in entry.surface
        assert "mógo-ko" in entry.segmented
        assert "world-DEF.NEAR" in entry.gloss
        assert "world of Aaruma" in entry.translation

    def test_lang1_segmented_gloss_not_empty(self, tmp_path):
        p = write(tmp_path, "lang1-corpus.txt", LANG1_CORPUS)
        for entry in parse_corpus(p, "lang1"):
            assert entry.segmented != ""
            assert entry.gloss != ""

    def test_lang2_sentence_count(self, tmp_path):
        p = write(tmp_path, "lang2-corpus.txt", LANG2_CORPUS)
        # 3rd block is a bare morpheme line with no translation — should be skipped
        entries = parse_corpus(p, "lang2")
        assert len(entries) == 2

    def test_lang2_segmented_gloss_empty(self, tmp_path):
        p = write(tmp_path, "lang2-corpus.txt", LANG2_CORPUS)
        for entry in parse_corpus(p, "lang2"):
            assert entry.segmented == ""
            assert entry.gloss == ""

    def test_lang2_unquoted_translation_accepted(self, tmp_path):
        p = write(tmp_path, "lang2-corpus.txt", LANG2_CORPUS)
        entry = parse_corpus(p, "lang2")[1]
        assert "hawk" in entry.translation

    def test_sentence_ids_sequential(self, tmp_path):
        p = write(tmp_path, "lang1-corpus.txt", LANG1_CORPUS)
        entries = parse_corpus(p, "lang1")
        assert [e.sentence_id for e in entries] == [1, 2]

    def test_incomplete_lang1_block_skipped(self, tmp_path):
        content = "Surface only.\nSegmented only.\n"
        p = write(tmp_path, "lang1-corpus.txt", content)
        assert parse_corpus(p, "lang1") == []

    def test_empty_file(self, tmp_path):
        p = write(tmp_path, "corpus.txt", "")
        assert parse_corpus(p, "lang2") == []


# ---------------------------------------------------------------------------
# parse_doc
# ---------------------------------------------------------------------------


DOC_FIXTURE = """\
Vúlunudonu nideni Tára, lízirede vúlunudonu izamimi.
Zidaba zólurulu izamimi.
Máguzúraspáraba uvóvumagomo.

"""


class TestParseDoc:
    def test_entry_count(self, tmp_path):
        p = write(tmp_path, "lang1-doc1.txt", DOC_FIXTURE)
        entries = parse_doc(p, "lang1", 1)
        assert len(entries) == 3

    def test_fields(self, tmp_path):
        p = write(tmp_path, "lang1-doc1.txt", DOC_FIXTURE)
        entry = parse_doc(p, "lang1", 1)[0]
        assert entry.language == "lang1"
        assert entry.doc_id == 1
        assert entry.line_no == 1
        assert "Vúlunudonu" in entry.text

    def test_blank_lines_skipped(self, tmp_path):
        p = write(tmp_path, "lang1-doc1.txt", DOC_FIXTURE)
        entries = parse_doc(p, "lang1", 1)
        for e in entries:
            assert e.text.strip() != ""

    def test_line_no_tracks_original_file_position(self, tmp_path):
        content = "Line one.\n\nLine three.\n"
        p = write(tmp_path, "doc.txt", content)
        entries = parse_doc(p, "lang1", 2)
        assert entries[0].line_no == 1
        assert entries[1].line_no == 3

    def test_doc_id_stored(self, tmp_path):
        p = write(tmp_path, "lang2-doc2.txt", DOC_FIXTURE)
        entry = parse_doc(p, "lang2", 2)[0]
        assert entry.doc_id == 2

    def test_empty_file(self, tmp_path):
        p = write(tmp_path, "doc.txt", "")
        assert parse_doc(p, "lang1", 1) == []


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_writes_header_and_rows(self, tmp_path):
        records = [
            DictionaryEntry("student", "dóruma", "", "NN"),
            DictionaryEntry("language", "lóruma", "", "NN"),
        ]
        out = tmp_path / "out.csv"
        export_csv(records, out)

        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 2
        assert rows[0]["english"] == "student"
        assert rows[0]["native_form"] == "dóruma"
        assert rows[0]["pos"] == "NN"

    def test_header_matches_dataclass_fields(self, tmp_path):
        records = [CorpusEntry("lang1", 1, "surface", "seg", "gloss", "translation")]
        out = tmp_path / "corpus.csv"
        export_csv(records, out)

        header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == ["language", "sentence_id", "surface", "segmented", "gloss", "translation"]

    def test_creates_parent_directories(self, tmp_path):
        records = [DocEntry("lang1", 1, 1, "text")]
        out = tmp_path / "nested" / "dir" / "docs.csv"
        export_csv(records, out)
        assert out.exists()

    def test_empty_records_writes_nothing(self, tmp_path):
        out = tmp_path / "empty.csv"
        export_csv([], out)
        assert not out.exists()

    def test_unicode_preserved(self, tmp_path):
        records = [DictionaryEntry("student", "dóruma", "", "NN")]
        out = tmp_path / "unicode.csv"
        export_csv(records, out)
        content = out.read_text(encoding="utf-8")
        assert "dóruma" in content


# ---------------------------------------------------------------------------
# parse_all
# ---------------------------------------------------------------------------


class TestParseAll:
    def _setup_minimal_raw(self, raw_dir: Path):
        raw_dir.mkdir(parents=True)
        (raw_dir / "lang1-dictionary.txt").write_text(
            "student\tdóruma\tNN\nastrónomer\tfálurunadóruma (ie, star-scholar)\tNN\n",
            encoding="utf-8",
        )
        (raw_dir / "lang1-corpus.txt").write_text(
            "Surface one.\nSeg one.\nGloss one.\n\"Translation one.\"\n\n"
            "Surface two.\nSeg two.\nGloss two.\n\"Translation two.\"\n",
            encoding="utf-8",
        )
        (raw_dir / "lang2-corpus.txt").write_text(
            "Surface lang2 one.\n\"Lang2 translation one.\"\n",
            encoding="utf-8",
        )
        (raw_dir / "lang1-doc1.txt").write_text(
            "Prose line one.\nProse line two.\n",
            encoding="utf-8",
        )

    def test_returns_all_three_keys(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        self._setup_minimal_raw(raw_dir)
        results = parse_all(raw_dir=raw_dir, processed_dir=out_dir)
        assert set(results.keys()) == {"dictionary", "corpus", "docs"}

    def test_dictionary_count(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        self._setup_minimal_raw(raw_dir)
        results = parse_all(raw_dir=raw_dir, processed_dir=out_dir)
        assert len(results["dictionary"]) == 2

    def test_corpus_combines_all_languages(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        self._setup_minimal_raw(raw_dir)
        results = parse_all(raw_dir=raw_dir, processed_dir=out_dir)
        langs = {e.language for e in results["corpus"]}
        assert "lang1" in langs
        assert "lang2" in langs

    def test_csv_files_created(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        self._setup_minimal_raw(raw_dir)
        parse_all(raw_dir=raw_dir, processed_dir=out_dir)
        assert (out_dir / "dictionary.csv").exists()
        assert (out_dir / "corpus.csv").exists()
        assert (out_dir / "docs.csv").exists()

    def test_missing_raw_dir_returns_empty(self, tmp_path):
        results = parse_all(
            raw_dir=tmp_path / "nonexistent",
            processed_dir=tmp_path / "out",
        )
        assert results["dictionary"] == []
        assert results["corpus"] == []
        assert results["docs"] == []


# ---------------------------------------------------------------------------
# Integration tests (require actual raw data)
# ---------------------------------------------------------------------------


@integration
class TestIntegration:
    """Run against the real data/raw/Hackaton 2026/ directory."""

    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("processed")
        return parse_all(processed_dir=out)

    def test_dictionary_has_expected_entry_count(self, results):
        # lang1-dictionary.txt has 579 lines; 578 have parseable POS tags
        assert len(results["dictionary"]) >= 570

    def test_dictionary_entry_types(self, results):
        for e in results["dictionary"]:
            assert isinstance(e, DictionaryEntry)

    def test_corpus_covers_all_seven_languages(self, results):
        langs = {e.language for e in results["corpus"]}
        assert langs == set(LANGUAGES)

    def test_corpus_entry_types(self, results):
        for e in results["corpus"]:
            assert isinstance(e, CorpusEntry)

    def test_lang1_corpus_has_segmented_gloss(self, results):
        lang1_entries = [e for e in results["corpus"] if e.language == "lang1"]
        assert len(lang1_entries) > 0
        for e in lang1_entries:
            assert e.segmented != "", f"sentence_id={e.sentence_id} missing segmented"
            assert e.gloss != "", f"sentence_id={e.sentence_id} missing gloss"

    def test_lang2_7_corpus_no_segmented_gloss(self, results):
        non_lang1 = [e for e in results["corpus"] if e.language != "lang1"]
        for e in non_lang1:
            assert e.segmented == ""
            assert e.gloss == ""

    def test_all_translations_non_empty(self, results):
        for e in results["corpus"]:
            assert e.translation != "", f"{e.language} sentence_id={e.sentence_id}"

    def test_docs_covers_all_seven_languages(self, results):
        langs = {e.language for e in results["docs"]}
        assert langs == set(LANGUAGES)

    def test_docs_both_doc_ids_present(self, results):
        doc_ids = {e.doc_id for e in results["docs"]}
        assert doc_ids == {1, 2}

    def test_doc_entry_types(self, results):
        for e in results["docs"]:
            assert isinstance(e, DocEntry)

    def test_no_empty_text_in_docs(self, results):
        for e in results["docs"]:
            assert e.text.strip() != ""

    def test_sentence_ids_per_language_are_sequential(self, results):
        from itertools import groupby

        for lang in LANGUAGES:
            lang_entries = sorted(
                [e for e in results["corpus"] if e.language == lang],
                key=lambda e: e.sentence_id,
            )
            if not lang_entries:
                continue
            ids = [e.sentence_id for e in lang_entries]
            assert ids == list(range(1, len(ids) + 1)), f"{lang}: non-sequential ids {ids[:10]}"

    def test_csv_dictionary_readable(self, tmp_path_factory):
        import csv as csv_mod

        out = tmp_path_factory.mktemp("csv_check")
        parse_all(processed_dir=out)
        rows = list(
            csv_mod.DictReader(
                (out / "dictionary.csv").read_text(encoding="utf-8").splitlines()
            )
        )
        assert len(rows) >= 570
        assert "english" in rows[0]
        assert "native_form" in rows[0]
        assert "pos" in rows[0]
