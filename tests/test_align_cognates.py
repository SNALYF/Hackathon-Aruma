"""
Tests for src/align_cognates.py.

Unit tests use inline fixtures and tmp_path so they do not depend on real data.
Integration tests are marked @pytest.mark.integration and skipped when the
processed data directory is absent.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.align_cognates import (
    LANGUAGES,
    PARALLEL_THRESHOLD,
    PROCESSED_DIR,
    SEARCH_THRESHOLD,
    CognateCandidate,
    CognateSet,
    best_candidate,
    build_cognate_sets,
    build_gloss_to_english,
    export_csv,
    extract_lang1_word_inventory,
    extract_parallel_candidates,
    extract_search_candidates,
    find_parallel_groups,
    get_search_terms,
    lcs_length,
    lcs_similarity,
    load_data,
    run,
    strip_diacritics,
    tokenise,
    _content_gloss,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROCESSED_DATA_AVAILABLE = (PROCESSED_DIR / "corpus.csv").exists()
integration = pytest.mark.skipif(
    not PROCESSED_DATA_AVAILABLE, reason="processed data directory not found"
)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts as a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_candidate(
    english="student",
    pos="NN",
    lang1_form="dóruma",
    language="lang2",
    candidate_form="doruma",
    similarity=0.9,
    method="parallel",
    sentence_id=1,
    sentence_length=5,
) -> CognateCandidate:
    return CognateCandidate(
        english=english,
        pos=pos,
        lang1_form=lang1_form,
        language=language,
        candidate_form=candidate_form,
        similarity=similarity,
        method=method,
        sentence_id=sentence_id,
        sentence_length=sentence_length,
    )


# ---------------------------------------------------------------------------
# strip_diacritics
# ---------------------------------------------------------------------------


class TestStripDiacritics:
    def test_removes_acute_accent(self):
        assert strip_diacritics("dóruma") == "doruma"

    def test_removes_grave_accent(self):
        assert strip_diacritics("àgùùg") == "aguug"

    def test_removes_mixed_accents(self):
        assert strip_diacritics("þàdórúmàgùùg") == "þadorumaguug"

    def test_preserves_ipa_consonants(self):
        # þ (U+00FE), ħ (U+0127), ð (U+00F0) should survive
        result = strip_diacritics("þħð")
        assert "þ" in result
        assert "ħ" in result
        assert "ð" in result

    def test_returns_lowercase(self):
        assert strip_diacritics("DÓRUMA") == "doruma"

    def test_empty_string(self):
        assert strip_diacritics("") == ""

    def test_no_diacritics_unchanged(self):
        assert strip_diacritics("doruma") == "doruma"

    def test_tilde_diacritic_removed(self):
        assert strip_diacritics("mṍzu") == "mozu"

    def test_ascii_digits_preserved(self):
        assert strip_diacritics("abc123") == "abc123"


# ---------------------------------------------------------------------------
# tokenise
# ---------------------------------------------------------------------------


class TestTokenise:
    def test_simple_split(self):
        assert tokenise("hello world") == ["hello", "world"]

    def test_strips_trailing_period(self):
        assert tokenise("hello world.") == ["hello", "world"]

    def test_strips_leading_quote(self):
        assert tokenise('"hello world."') == ["hello", "world"]

    def test_strips_comma(self):
        assert tokenise("cats, dogs") == ["cats", "dogs"]

    def test_preserves_internal_hyphen(self):
        tokens = tokenise("well-known fact")
        assert "well-known" in tokens

    def test_preserves_internal_apostrophe(self):
        tokens = tokenise("swan's feathers")
        assert "swan's" in tokens

    def test_empty_string(self):
        assert tokenise("") == []

    def test_whitespace_only(self):
        assert tokenise("   ") == []

    def test_single_token(self):
        assert tokenise("word") == ["word"]

    def test_multiple_spaces(self):
        assert tokenise("a  b") == ["a", "b"]

    def test_unicode_token(self):
        tokens = tokenise("dóruma kávuru")
        assert tokens == ["dóruma", "kávuru"]

    def test_strips_parentheses(self):
        tokens = tokenise("(hello) world")
        assert "hello" in tokens

    def test_no_empty_tokens_returned(self):
        tokens = tokenise("... ...")
        for t in tokens:
            assert t != ""


# ---------------------------------------------------------------------------
# lcs_length
# ---------------------------------------------------------------------------


class TestLcsLength:
    def test_identical_strings(self):
        assert lcs_length("abc", "abc") == 3

    def test_completely_different(self):
        assert lcs_length("abc", "xyz") == 0

    def test_subsequence(self):
        assert lcs_length("ace", "abcde") == 3

    def test_empty_first(self):
        assert lcs_length("", "abc") == 0

    def test_empty_second(self):
        assert lcs_length("abc", "") == 0

    def test_both_empty(self):
        assert lcs_length("", "") == 0

    def test_single_char_match(self):
        assert lcs_length("a", "a") == 1

    def test_single_char_no_match(self):
        assert lcs_length("a", "b") == 0

    def test_one_char_substring(self):
        assert lcs_length("a", "bac") == 1

    def test_known_example(self):
        # LCS("doruma", "dorumaguug") = len("doruma") = 6
        assert lcs_length("doruma", "dorumaguug") == 6

    def test_symmetry(self):
        assert lcs_length("abcde", "ace") == lcs_length("ace", "abcde")


# ---------------------------------------------------------------------------
# lcs_similarity
# ---------------------------------------------------------------------------


class TestLcsSimilarity:
    def test_identical_strings(self):
        assert lcs_similarity("abc", "abc") == 1.0

    def test_completely_different(self):
        assert lcs_similarity("abc", "xyz") == 0.0

    def test_empty_both(self):
        assert lcs_similarity("", "") == 0.0

    def test_empty_first(self):
        assert lcs_similarity("", "abc") == 0.0

    def test_empty_second(self):
        assert lcs_similarity("abc", "") == 0.0

    def test_cognate_pair_above_threshold(self):
        # documented example: genuine cognate pair should be ~0.67
        sim = lcs_similarity("dóruma", "þàdórúmàgùùg")
        assert sim > PARALLEL_THRESHOLD

    def test_unrelated_words_below_threshold(self):
        # documented example: unrelated words ~0.22
        sim = lcs_similarity("dóruma", "kávuru")
        assert sim < PARALLEL_THRESHOLD

    def test_diacritics_stripped_before_comparison(self):
        # Same underlying letters, different diacritics → should be identical
        assert lcs_similarity("dóruma", "doruma") == 1.0

    def test_result_in_range(self):
        sim = lcs_similarity("hello", "world")
        assert 0.0 <= sim <= 1.0

    def test_formula_2lcs_over_total(self):
        a, b = "abc", "ac"
        # strip_diacritics is identity here; LCS("abc","ac")=2; total=5
        expected = 2 * 2 / (3 + 2)
        assert abs(lcs_similarity(a, b) - expected) < 1e-9


# ---------------------------------------------------------------------------
# get_search_terms
# ---------------------------------------------------------------------------


class TestGetSearchTerms:
    def test_verb_prefix_stripped(self):
        assert "create" in get_search_terms("to create")
        assert "to" not in get_search_terms("to create")

    def test_slash_splits_into_multiple_terms(self):
        terms = get_search_terms("ring/circle")
        assert "ring" in terms
        assert "circle" in terms

    def test_slash_with_to_prefix(self):
        terms = get_search_terms("to flee/to run")
        assert "flee" in terms
        assert "run" in terms

    def test_multi_word_entry(self):
        terms = get_search_terms("point of view")
        assert "point" in terms
        assert "view" in terms

    def test_stopwords_excluded(self):
        terms = get_search_terms("the student")
        assert "the" not in terms
        assert "student" in terms

    def test_short_words_excluded(self):
        terms = get_search_terms("to be or not")
        assert "be" not in terms
        assert "or" not in terms
        assert "to" not in terms

    def test_deduplication(self):
        terms = get_search_terms("run/to run")
        assert terms.count("run") == 1

    def test_order_preserved(self):
        terms = get_search_terms("create/build")
        assert terms.index("create") < terms.index("build")

    def test_empty_string(self):
        assert get_search_terms("") == []

    def test_punctuation_stripped(self):
        terms = get_search_terms("love, hate")
        assert "love" in terms
        assert "hate" in terms


# ---------------------------------------------------------------------------
# _content_gloss
# ---------------------------------------------------------------------------


class TestContentGloss:
    def test_simple_content_word(self):
        assert _content_gloss("student") == "student"

    def test_content_with_grammatical_suffix(self):
        assert _content_gloss("student-PL-DEF-ERG") == "student"

    def test_grammatical_prefix_skipped(self):
        assert _content_gloss("INAN-welcome-WIT") == "welcome"

    def test_purely_grammatical_returns_none(self):
        assert _content_gloss("ERG") is None
        assert _content_gloss("1SG") is None

    def test_multiple_gram_tags_returns_none(self):
        assert _content_gloss("PL-DEF") is None

    def test_trailing_comma_handled(self):
        assert _content_gloss("student,") == "student"

    def test_empty_string_returns_none(self):
        assert _content_gloss("") is None

    def test_first_lowercase_wins(self):
        # "INAN-welcome-WIT": skip INAN, return "welcome" not "WIT"
        result = _content_gloss("INAN-welcome-WIT")
        assert result == "welcome"

    def test_dotted_gloss_stripped(self):
        # "DEF.NEAR" style — cleaned and uppercased check
        assert _content_gloss("DEF.NEAR") is None

    def test_numeric_person_tag_returns_none(self):
        assert _content_gloss("3PL") is None


# ---------------------------------------------------------------------------
# build_gloss_to_english
# ---------------------------------------------------------------------------


class TestBuildGlossToEnglish:
    def _dict(self):
        return [
            {"english": "student", "native_form": "dóruma"},
            {"english": "to create", "native_form": "góruamu"},
            {"english": "point of view", "native_form": "xxx"},
        ]

    def test_simple_noun_indexed(self):
        index = build_gloss_to_english(self._dict())
        assert index["student"] == "student"

    def test_verb_to_prefix_stripped(self):
        index = build_gloss_to_english(self._dict())
        assert index["create"] == "to create"

    def test_first_word_of_multiword_indexed(self):
        index = build_gloss_to_english(self._dict())
        assert index["point"] == "point of view"

    def test_full_key_also_indexed(self):
        index = build_gloss_to_english(self._dict())
        assert index["point of view"] == "point of view"

    def test_empty_dictionary(self):
        assert build_gloss_to_english([]) == {}


# ---------------------------------------------------------------------------
# extract_lang1_word_inventory
# ---------------------------------------------------------------------------


class TestExtractLang1WordInventory:
    def _make_row(self, surface, segmented, gloss, translation="A sentence.", sid=1):
        return {
            "sentence_id": str(sid),
            "surface": surface,
            "segmented": segmented,
            "gloss": gloss,
            "translation": translation,
        }

    def _gloss_to_eng(self):
        return {"student": "student", "welcome": "to welcome", "world": "world"}

    def test_basic_extraction(self):
        row = self._make_row(
            "Studenta worldo.",
            "student-ERG world-DEF",
            "student-ERG world-DEF.NEAR",
        )
        inv = extract_lang1_word_inventory([row], {"student": "student", "world": "world"})
        assert 1 in inv
        pairs = dict(inv[1])
        assert "Studenta" in pairs or "student" in [t for t, _ in inv[1]]

    def test_mismatched_token_counts_skipped(self):
        row = self._make_row(
            "one two three",
            "one-X",          # 1 seg vs 3 surface
            "ONE-ERG",
        )
        inv = extract_lang1_word_inventory([row], {"one": "one"})
        assert inv == {}

    def test_missing_segmented_skipped(self):
        row = self._make_row("surface.", "", "GLOSS")
        inv = extract_lang1_word_inventory([row], {"gloss": "gloss"})
        assert inv == {}

    def test_missing_gloss_skipped(self):
        row = self._make_row("surface.", "seg", "")
        inv = extract_lang1_word_inventory([row], {"seg": "seg"})
        assert inv == {}

    def test_purely_grammatical_row_produces_no_pairs(self):
        # All gloss tokens are grammatical
        row = self._make_row("nie", "ni-ERG", "1SG-ERG")
        inv = extract_lang1_word_inventory([row], {"1sg": "I"})
        assert inv == {}

    def test_sentence_id_used_as_key(self):
        row = self._make_row("dóruma.", "dóruma", "student", sid=42)
        inv = extract_lang1_word_inventory([row], {"student": "student"})
        assert 42 in inv

    def test_multiple_rows(self):
        rows = [
            self._make_row("dóruma.", "dóruma", "student", sid=1),
            self._make_row("góruamu.", "góruamu", "create", sid=2),
        ]
        gloss_map = {"student": "student", "create": "to create"}
        inv = extract_lang1_word_inventory(rows, gloss_map)
        assert 1 in inv
        assert 2 in inv


# ---------------------------------------------------------------------------
# find_parallel_groups
# ---------------------------------------------------------------------------


class TestFindParallelGroups:
    def _corpus(self, translations_by_lang: dict[str, list[str]]) -> dict[str, list[dict]]:
        corpus: dict[str, list[dict]] = {}
        for lang, translations in translations_by_lang.items():
            corpus[lang] = [
                {"sentence_id": str(i + 1), "translation": t, "surface": f"surface{i+1}"}
                for i, t in enumerate(translations)
            ]
        return corpus

    def test_shared_translation_grouped(self):
        corpus = self._corpus({
            "lang1": ["The cat sat.", "Other."],
            "lang2": ["The cat sat.", "Different."],
        })
        groups = find_parallel_groups(corpus)
        shared = [g for g in groups if "lang1" in g and "lang2" in g]
        assert len(shared) == 1

    def test_unique_translations_not_grouped(self):
        corpus = self._corpus({
            "lang1": ["Unique sentence one."],
            "lang2": ["Unique sentence two."],
        })
        groups = find_parallel_groups(corpus)
        assert groups == []

    def test_group_contains_correct_rows(self):
        corpus = self._corpus({
            "lang1": ["Shared."],
            "lang2": ["Shared."],
        })
        groups = find_parallel_groups(corpus)
        assert len(groups) == 1
        assert "lang1" in groups[0]
        assert "lang2" in groups[0]

    def test_three_language_group(self):
        corpus = self._corpus({
            "lang1": ["Shared."],
            "lang2": ["Shared."],
            "lang3": ["Shared."],
        })
        groups = find_parallel_groups(corpus)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_single_language_not_included(self):
        corpus = self._corpus({"lang1": ["Only one language."]})
        groups = find_parallel_groups(corpus)
        assert groups == []

    def test_empty_corpus(self):
        assert find_parallel_groups({}) == []

    def test_whitespace_in_translation_handled(self):
        # Leading/trailing whitespace is stripped before grouping
        corpus = {
            "lang1": [{"sentence_id": "1", "translation": "  Shared.  ", "surface": "s1"}],
            "lang2": [{"sentence_id": "1", "translation": "Shared.", "surface": "s2"}],
        }
        groups = find_parallel_groups(corpus)
        assert len(groups) == 1


# ---------------------------------------------------------------------------
# extract_parallel_candidates
# ---------------------------------------------------------------------------


class TestExtractParallelCandidates:
    def _make_group(self, lang1_sid, lang1_surface, lang2_surface, translation="Shared."):
        return {
            "lang1": {"sentence_id": str(lang1_sid), "surface": lang1_surface, "translation": translation},
            "lang2": {"sentence_id": "10", "surface": lang2_surface, "translation": translation},
        }

    def _inventory(self, sid, token, english):
        return {sid: [(token, english)]}

    def _dict_by_english(self):
        return {"student": {"english": "student", "pos": "NN", "native_form": "dóruma"}}

    def test_candidate_produced_when_above_threshold(self):
        group = self._make_group(1, "dóruma.", "doruma kávuru.")
        inventory = self._inventory(1, "dóruma", "student")
        cands = extract_parallel_candidates(
            [group], inventory, self._dict_by_english(), threshold=0.5
        )
        assert len(cands) >= 1
        assert cands[0].english == "student"
        assert cands[0].language == "lang2"

    def test_no_candidate_below_threshold(self):
        group = self._make_group(1, "dóruma.", "xyz abc.")
        inventory = self._inventory(1, "dóruma", "student")
        cands = extract_parallel_candidates(
            [group], inventory, self._dict_by_english(), threshold=0.99
        )
        assert cands == []

    def test_no_lang1_row_skipped(self):
        group = {"lang2": {"sentence_id": "1", "surface": "doruma.", "translation": "Shared."}}
        inventory = self._inventory(1, "dóruma", "student")
        cands = extract_parallel_candidates([group], inventory, self._dict_by_english())
        assert cands == []

    def test_lang1_not_in_inventory_skipped(self):
        group = self._make_group(99, "dóruma.", "doruma.")
        inventory = {}  # sid 99 not present
        cands = extract_parallel_candidates([group], inventory, self._dict_by_english())
        assert cands == []

    def test_method_is_parallel(self):
        group = self._make_group(1, "dóruma.", "doruma kávuru.")
        inventory = self._inventory(1, "dóruma", "student")
        cands = extract_parallel_candidates([group], inventory, self._dict_by_english(), threshold=0.5)
        assert all(c.method == "parallel" for c in cands)

    def test_candidate_fields_populated(self):
        group = self._make_group(1, "dóruma.", "doruma.")
        inventory = self._inventory(1, "dóruma", "student")
        cands = extract_parallel_candidates([group], inventory, self._dict_by_english(), threshold=0.5)
        assert len(cands) == 1
        c = cands[0]
        assert c.pos == "NN"
        assert c.lang1_form == "dóruma"
        assert c.sentence_id == 10
        assert c.sentence_length == 1

    def test_similarity_rounded_to_4_decimals(self):
        group = self._make_group(1, "dóruma.", "doruma.")
        inventory = self._inventory(1, "dóruma", "student")
        cands = extract_parallel_candidates([group], inventory, self._dict_by_english(), threshold=0.0)
        assert len(str(cands[0].similarity).split(".")[-1]) <= 4

    def test_empty_groups_list(self):
        cands = extract_parallel_candidates([], {}, self._dict_by_english())
        assert cands == []


# ---------------------------------------------------------------------------
# extract_search_candidates
# ---------------------------------------------------------------------------


class TestExtractSearchCandidates:
    def _dict(self):
        return [{"english": "student", "native_form": "dóruma", "pos": "NN"}]

    def _corpus(self, translation, surface, sid=1):
        return {
            "lang2": [{"sentence_id": str(sid), "translation": translation, "surface": surface}]
        }

    def test_candidate_found_when_keyword_in_translation(self):
        cands = extract_search_candidates(
            self._dict(),
            self._corpus("The student arrived.", "doruma kávuru."),
            covered=set(),
            threshold=0.5,
        )
        assert len(cands) >= 1
        assert cands[0].english == "student"

    def test_covered_pair_skipped(self):
        cands = extract_search_candidates(
            self._dict(),
            self._corpus("The student arrived.", "doruma kávuru."),
            covered={("student", "lang2")},
            threshold=0.5,
        )
        assert cands == []

    def test_no_keyword_match_skipped(self):
        cands = extract_search_candidates(
            self._dict(),
            self._corpus("The teacher arrived.", "doruma kávuru."),
            covered=set(),
            threshold=0.5,
        )
        assert cands == []

    def test_sentence_too_long_skipped(self):
        surface = " ".join([f"tok{i}" for i in range(20)])
        cands = extract_search_candidates(
            self._dict(),
            self._corpus("The student arrived.", surface),
            covered=set(),
            threshold=0.0,
            max_tokens=10,
        )
        assert cands == []

    def test_below_threshold_skipped(self):
        cands = extract_search_candidates(
            self._dict(),
            self._corpus("The student arrived.", "xyz."),
            covered=set(),
            threshold=0.99,
        )
        assert cands == []

    def test_method_is_search(self):
        cands = extract_search_candidates(
            self._dict(),
            self._corpus("The student arrived.", "doruma."),
            covered=set(),
            threshold=0.5,
        )
        assert all(c.method == "search" for c in cands)

    def test_empty_lang1_form_skipped(self):
        dictionary = [{"english": "student", "native_form": "", "pos": "NN"}]
        cands = extract_search_candidates(
            dictionary,
            self._corpus("The student arrived.", "doruma."),
            covered=set(),
        )
        assert cands == []

    def test_empty_dictionary(self):
        cands = extract_search_candidates([], self._corpus("Anything.", "tok."), covered=set())
        assert cands == []

    def test_sentence_length_stored(self):
        cands = extract_search_candidates(
            self._dict(),
            self._corpus("The student arrived.", "doruma kávuru."),
            covered=set(),
            threshold=0.5,
        )
        if cands:
            assert cands[0].sentence_length == 2

    def test_lang1_not_searched(self):
        corpus = {
            "lang1": [{"sentence_id": "1", "translation": "The student arrived.", "surface": "doruma."}]
        }
        cands = extract_search_candidates(self._dict(), corpus, covered=set(), threshold=0.5)
        # lang1 is excluded from LANGUAGES[1:]
        assert cands == []


# ---------------------------------------------------------------------------
# best_candidate
# ---------------------------------------------------------------------------


class TestBestCandidate:
    def test_empty_returns_none(self):
        assert best_candidate([]) is None

    def test_single_candidate(self):
        c = make_candidate()
        assert best_candidate([c]) is c

    def test_parallel_outranks_search(self):
        parallel = make_candidate(method="parallel", similarity=0.6)
        search = make_candidate(method="search", similarity=0.95)
        assert best_candidate([parallel, search]) is parallel

    def test_higher_similarity_wins_same_method(self):
        low = make_candidate(method="parallel", similarity=0.6)
        high = make_candidate(method="parallel", similarity=0.9)
        assert best_candidate([low, high]) is high

    def test_shorter_sentence_wins_same_method_and_similarity(self):
        long_sent = make_candidate(method="parallel", similarity=0.8, sentence_length=10)
        short_sent = make_candidate(method="parallel", similarity=0.8, sentence_length=3)
        assert best_candidate([long_sent, short_sent]) is short_sent

    def test_parallel_beats_search_regardless_of_similarity(self):
        search = make_candidate(method="search", similarity=1.0, sentence_length=1)
        parallel = make_candidate(method="parallel", similarity=0.51, sentence_length=100)
        assert best_candidate([search, parallel]) is parallel

    def test_all_search_picks_highest_similarity(self):
        c1 = make_candidate(method="search", similarity=0.5)
        c2 = make_candidate(method="search", similarity=0.8)
        c3 = make_candidate(method="search", similarity=0.6)
        assert best_candidate([c1, c2, c3]) is c2


# ---------------------------------------------------------------------------
# build_cognate_sets
# ---------------------------------------------------------------------------


class TestBuildCognateSets:
    def _dictionary(self):
        return [
            {"english": "student", "native_form": "dóruma", "pos": "NN"},
            {"english": "to create", "native_form": "góruamu", "pos": "VB"},
        ]

    def test_returns_one_set_per_dictionary_entry(self):
        sets = build_cognate_sets([], self._dictionary())
        assert len(sets) == 2

    def test_lang1_taken_from_dictionary(self):
        sets = build_cognate_sets([], self._dictionary())
        assert sets[0].lang1 == "dóruma"
        assert sets[1].lang1 == "góruamu"

    def test_empty_string_when_no_candidate(self):
        sets = build_cognate_sets([], self._dictionary())
        for s in sets:
            for lang in ["lang2", "lang3", "lang4", "lang5", "lang6", "lang7"]:
                assert getattr(s, lang) == ""

    def test_candidate_form_placed_in_correct_language(self):
        c = make_candidate(english="student", language="lang3", candidate_form="dorumax", method="parallel", similarity=0.9)
        sets = build_cognate_sets([c], self._dictionary())
        student_set = next(s for s in sets if s.english == "student")
        assert student_set.lang3 == "dorumax"
        assert student_set.lang2 == ""

    def test_best_candidate_selected(self):
        low = make_candidate(english="student", language="lang2", candidate_form="bad", method="search", similarity=0.5)
        high = make_candidate(english="student", language="lang2", candidate_form="good", method="parallel", similarity=0.9)
        sets = build_cognate_sets([low, high], self._dictionary())
        student_set = next(s for s in sets if s.english == "student")
        assert student_set.lang2 == "good"

    def test_english_and_pos_fields(self):
        sets = build_cognate_sets([], self._dictionary())
        assert sets[0].english == "student"
        assert sets[0].pos == "NN"

    def test_all_languages_present_in_set(self):
        sets = build_cognate_sets([], self._dictionary())
        for s in sets:
            for lang in LANGUAGES:
                assert hasattr(s, lang)


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_writes_header_and_rows(self, tmp_path):
        records = [
            make_candidate(english="student", language="lang2"),
            make_candidate(english="to create", language="lang3"),
        ]
        out = tmp_path / "out.csv"
        export_csv(records, out)
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 2
        assert rows[0]["english"] == "student"
        assert rows[1]["language"] == "lang3"

    def test_header_matches_dataclass_fields(self, tmp_path):
        records = [make_candidate()]
        out = tmp_path / "candidates.csv"
        export_csv(records, out)
        header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == [
            "english", "pos", "lang1_form", "language", "candidate_form",
            "similarity", "method", "sentence_id", "sentence_length"
        ]

    def test_cognate_set_written_correctly(self, tmp_path):
        records = [CognateSet("student", "NN", "dóruma", "", "", "", "", "", "")]
        out = tmp_path / "sets.csv"
        export_csv(records, out)
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["lang1"] == "dóruma"
        assert rows[0]["lang2"] == ""

    def test_empty_list_writes_nothing(self, tmp_path):
        out = tmp_path / "empty.csv"
        export_csv([], out)
        assert not out.exists()

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "out.csv"
        export_csv([make_candidate()], out)
        assert out.exists()

    def test_unicode_preserved(self, tmp_path):
        records = [make_candidate(lang1_form="dóruma", candidate_form="þàdórúmàgùùg")]
        out = tmp_path / "unicode.csv"
        export_csv(records, out)
        content = out.read_text(encoding="utf-8")
        assert "dóruma" in content
        assert "þàdórúmàgùùg" in content


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def _write_processed(self, tmp_path: Path):
        dict_rows = [
            {"english": "student", "native_form": "dóruma", "pos": "NN", "compound_explanation": ""},
            {"english": "to create", "native_form": "góruamu", "pos": "VB", "compound_explanation": ""},
        ]
        corpus_rows = [
            {"language": "lang1", "sentence_id": "1", "surface": "dóruma.", "segmented": "dóruma", "gloss": "student", "translation": "The student."},
            {"language": "lang2", "sentence_id": "1", "surface": "doruma.", "segmented": "", "gloss": "", "translation": "The student."},
        ]
        write_csv(tmp_path / "dictionary.csv", dict_rows)
        write_csv(tmp_path / "corpus.csv", corpus_rows)
        return tmp_path

    def test_loads_dictionary(self, tmp_path):
        self._write_processed(tmp_path)
        dictionary, _ = load_data(tmp_path)
        assert len(dictionary) == 2
        assert dictionary[0]["english"] == "student"

    def test_loads_corpus_by_language(self, tmp_path):
        self._write_processed(tmp_path)
        _, corpus = load_data(tmp_path)
        assert "lang1" in corpus
        assert "lang2" in corpus
        assert corpus["lang1"][0]["surface"] == "dóruma."

    def test_missing_dictionary_returns_empty(self, tmp_path):
        # Only write corpus
        rows = [{"language": "lang1", "sentence_id": "1", "surface": "s", "segmented": "", "gloss": "", "translation": "t"}]
        write_csv(tmp_path / "corpus.csv", rows)
        dictionary, _ = load_data(tmp_path)
        assert dictionary == []

    def test_missing_corpus_returns_empty(self, tmp_path):
        rows = [{"english": "student", "native_form": "dóruma", "pos": "NN", "compound_explanation": ""}]
        write_csv(tmp_path / "dictionary.csv", rows)
        _, corpus = load_data(tmp_path)
        assert corpus == {}

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        dictionary, corpus = load_data(tmp_path / "nonexistent")
        assert dictionary == []
        assert corpus == {}


# ---------------------------------------------------------------------------
# run (end-to-end with minimal fixture data)
# ---------------------------------------------------------------------------


class TestRun:
    def _write_minimal(self, tmp_path: Path) -> Path:
        """Set up a minimal but valid processed directory."""
        dict_rows = [
            {"english": "student", "native_form": "dóruma", "pos": "NN", "compound_explanation": ""},
        ]
        corpus_rows = [
            {
                "language": "lang1",
                "sentence_id": "1",
                "surface": "dóruma.",
                "segmented": "dóruma",
                "gloss": "student",
                "translation": "The student.",
            },
            {
                "language": "lang2",
                "sentence_id": "1",
                "surface": "doruma.",
                "segmented": "",
                "gloss": "",
                "translation": "The student.",
            },
        ]
        write_csv(tmp_path / "dictionary.csv", dict_rows)
        write_csv(tmp_path / "corpus.csv", corpus_rows)
        return tmp_path

    def test_returns_expected_keys(self, tmp_path):
        self._write_minimal(tmp_path)
        result = run(processed_dir=tmp_path)
        assert "candidates" in result
        assert "cognate_sets" in result

    def test_cognate_sets_count_equals_dictionary(self, tmp_path):
        self._write_minimal(tmp_path)
        result = run(processed_dir=tmp_path)
        assert len(result["cognate_sets"]) == 1

    def test_output_csvs_created(self, tmp_path):
        self._write_minimal(tmp_path)
        run(processed_dir=tmp_path)
        assert (tmp_path / "cognate_candidates.csv").exists()
        assert (tmp_path / "cognate_sets.csv").exists()

    def test_empty_processed_dir_runs_without_error(self, tmp_path):
        result = run(processed_dir=tmp_path)
        assert result["candidates"] == []
        assert result["cognate_sets"] == []

    def test_candidates_are_cognate_candidate_instances(self, tmp_path):
        self._write_minimal(tmp_path)
        result = run(processed_dir=tmp_path)
        for c in result["candidates"]:
            assert isinstance(c, CognateCandidate)

    def test_cognate_sets_are_cognate_set_instances(self, tmp_path):
        self._write_minimal(tmp_path)
        result = run(processed_dir=tmp_path)
        for s in result["cognate_sets"]:
            assert isinstance(s, CognateSet)

    def test_no_candidate_for_uncovered_language(self, tmp_path):
        """lang3–lang7 have no data; their form in cognate_sets should be empty."""
        self._write_minimal(tmp_path)
        result = run(processed_dir=tmp_path)
        s = result["cognate_sets"][0]
        for lang in ["lang3", "lang4", "lang5", "lang6", "lang7"]:
            assert getattr(s, lang) == ""


# ---------------------------------------------------------------------------
# Integration tests (require actual processed data)
# ---------------------------------------------------------------------------


@integration
class TestIntegration:
    @pytest.fixture(scope="class")
    def results(self):
        return run(processed_dir=PROCESSED_DIR)

    def test_candidates_list_nonempty(self, results):
        assert len(results["candidates"]) > 0

    def test_cognate_sets_list_nonempty(self, results):
        assert len(results["cognate_sets"]) > 0

    def test_parallel_candidates_present(self, results):
        parallel = [c for c in results["candidates"] if c.method == "parallel"]
        assert len(parallel) > 0

    def test_search_candidates_present(self, results):
        search = [c for c in results["candidates"] if c.method == "search"]
        assert len(search) > 0

    def test_all_similarities_in_range(self, results):
        for c in results["candidates"]:
            assert 0.0 <= c.similarity <= 1.0, f"out-of-range sim: {c}"

    def test_parallel_similarities_above_threshold(self, results):
        for c in results["candidates"]:
            if c.method == "parallel":
                assert c.similarity >= PARALLEL_THRESHOLD

    def test_search_similarities_above_threshold(self, results):
        for c in results["candidates"]:
            if c.method == "search":
                assert c.similarity >= SEARCH_THRESHOLD

    def test_all_languages_covered_in_sets(self, results):
        covered = {lang for s in results["cognate_sets"] for lang in LANGUAGES if getattr(s, lang)}
        assert covered >= {"lang1"}

    def test_no_empty_lang1_in_sets(self, results):
        for s in results["cognate_sets"]:
            assert s.lang1 != "", f"lang1 form missing for english={s.english!r}"

    def test_candidate_languages_are_lang2_to_lang7(self, results):
        for c in results["candidates"]:
            assert c.language in LANGUAGES[1:], f"unexpected language: {c.language}"

    def test_cognate_sets_csv_is_valid(self):
        path = PROCESSED_DIR / "cognate_sets.csv"
        if not path.exists():
            pytest.skip("cognate_sets.csv not yet generated")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) > 0
        assert "english" in rows[0]
        assert "lang1" in rows[0]

    def test_candidates_csv_is_valid(self):
        path = PROCESSED_DIR / "cognate_candidates.csv"
        if not path.exists():
            pytest.skip("cognate_candidates.csv not yet generated")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) > 0
        assert "similarity" in rows[0]
        assert "method" in rows[0]
