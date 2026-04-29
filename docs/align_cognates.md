# align_cognates — Stage 2: Cognate Identification

`src/align_cognates.py` takes the structured outputs of Stage 1 and identifies
which words across the seven daughter languages descend from the same proto-form.
These cognate sets are the direct input for Stage 3 (sound correspondence analysis).

---

## Usage

```bash
# defaults — reads from data/processed/, writes back to data/processed/
python src/align_cognates.py

# custom path
python src/align_cognates.py --processed-dir path/to/processed
```

Sample stdout:

```
candidates      :  3914  (parallel: 2786, search: 1128)
cognate sets    :   578
fully covered   :    86  (form found in all 7 languages)
coverage per language:
  lang1   578 / 578  (100%)
  lang2   268 / 578  ( 46%)
  lang3   290 / 578  ( 50%)
  lang4   275 / 578  ( 48%)
  lang5   292 / 578  ( 51%)
  lang6   308 / 578  ( 53%)
  lang7   268 / 578  ( 46%)
```

The per-language coverage of ~46–53 % reflects the corpus size (~115 sentences
per language).  Meanings that do not appear in any sentence receive empty cells
and should be treated as gaps rather than errors.

---

## Inputs

Both files are expected in `data/processed/` (produced by Stage 1).

| File | Used for |
|---|---|
| `dictionary.csv` | Ground-truth lang1 citation forms and POS tags |
| `corpus.csv` | Surface sentences + translations (all languages); interlinear data (lang1 only) |

---

## Algorithm

### Similarity metric

All comparisons use **LCS (longest common subsequence) similarity**:

```
sim(a, b) = 2 × LCS_length(a, b) / (len(a) + len(b))
```

computed on **diacritic-stripped lowercase** strings.  Stripping diacritics
is essential because the daughter languages encode the same underlying vowels
with different accent marks (e.g. `ó`, `ò`, `õ`, `ô` all reduce to `o`).
IPA-derived consonants (þ, ħ, ð) are retained because they represent distinct
phonemes.

LCS is preferred over edit distance here because the daughter-language forms
are inflected surface tokens that share a common root but carry different
prefixes and suffixes.  The root appears as a **subsequence** of the inflected
form regardless of which affixes surround it, so LCS similarity stays high
even when the full strings differ in length.

Example — "student":

| Language | Form | Stripped | Similarity to lang1 |
|---|---|---|---|
| lang1 | `dóruma` | `doruma` | 1.00 (reference) |
| lang2 | `þàdórúmàdórúmàgù` | `þadorumadorumagu` | 0.67 |
| lang3 | `þàdrúmã̀drúmã̀go` | `þadrumadrumago` | 0.62 |

### Method 1 — Parallel sentence alignment (primary)

Because ~93 % of corpus translations appear in two or more languages, most
sentences have a direct parallel.

Steps:
1. **Group sentences by translation** — every set of rows sharing the same
   English translation forms a parallel group.
2. **Parse lang1 interlinear glosses** — the `segmented` and `gloss` columns
   of lang1 sentences are aligned token-by-token.  For each token, the first
   lowercase morpheme gloss is extracted as the content meaning (e.g.
   `student-PL-DEF-ERG` → `student`).  Purely grammatical glosses (`ERG`,
   `INAN`, `WIT`, etc.) are skipped.
3. **Align tokens within each group** — for each lang1 surface token whose
   meaning maps to a dictionary entry, find the highest-similarity token in
   the parallel lang2–7 sentence.  Record as a candidate if `similarity ≥ 0.50`.

This method is more reliable because positional context limits the candidate
pool to tokens in the same sentence, not the whole corpus.

### Method 2 — Translation-keyword search (fallback)

For any `(english, language)` pair not covered by Method 1, the entire target-
language corpus is searched:

1. Extract content keywords from the English meaning (`"to create"` → `["create"]`,
   `"ring/circle"` → `["ring", "circle"]`).
2. Find sentences whose translation contains at least one keyword.
3. Ignore sentences longer than 10 tokens (too much token ambiguity).
4. Score every token in each matching sentence against the lang1 citation form;
   keep tokens with `similarity ≥ 0.45`.

The lower threshold (0.45 vs 0.50) reflects reduced confidence from the lack
of positional context.

### Candidate aggregation

All candidates for the same `(english, language)` pair are ranked by:

1. **Method** — `parallel` > `search`
2. **Similarity** — higher is better
3. **Sentence length** — shorter sentences preferred (less token ambiguity)

The top-ranked candidate becomes the entry in `cognate_sets.csv`.

---

## Output files

Both files are written to `data/processed/`.

### cognate_candidates.csv

Every candidate pair that cleared its method's threshold.  Useful for
inspecting confidence, auditing decisions, or re-running Stage 3 with a
different threshold.

| Column | Type | Description |
|---|---|---|
| `english` | str | Dictionary English meaning |
| `pos` | str | Part-of-speech tag |
| `lang1_form` | str | Lang1 citation form from dictionary |
| `language` | str | Target language (`lang2`–`lang7`) |
| `candidate_form` | str | Surface token from the target language |
| `similarity` | float | LCS similarity (0–1) |
| `method` | str | `"parallel"` or `"search"` |
| `sentence_id` | int | Sentence ID in the target language's corpus |
| `sentence_length` | int | Token count of the source sentence |

### cognate_sets.csv

One row per dictionary entry; the best candidate form per language in wide format.

| Column | Type | Description |
|---|---|---|
| `english` | str | Dictionary English meaning |
| `pos` | str | Part-of-speech tag |
| `lang1`–`lang7` | str | Best candidate form for each language; `""` if none found |

Sample rows:

| english | pos | lang1 | lang2 | lang3 |
|---|---|---|---|---|
| student | NN | dóruma | þàdórúmàdórúmàgù | þàdrúmã̀drúmã̀go |
| world | NN | mógo | Mógùgù | mṍgonṹ |
| together | RB | tírimi | tíírímí | tíírímĩ |

---

## Public API

```python
from src.align_cognates import (
    strip_diacritics,    # str -> str
    tokenise,            # str -> list[str]
    lcs_length,          # str, str -> int
    lcs_similarity,      # str, str -> float
    load_data,           # Path -> (list[dict], dict[str, list[dict]])
    get_search_terms,    # str -> list[str]
    find_parallel_groups,       # dict -> list[dict]
    extract_parallel_candidates,
    extract_search_candidates,
    best_candidate,      # list[CognateCandidate] -> CognateCandidate | None
    build_cognate_sets,  # list[CognateCandidate], list[dict] -> list[CognateSet]
    run,                 # Path -> dict
)
```

### Data models

```python
@dataclass
class CognateCandidate:
    english: str
    pos: str
    lang1_form: str
    language: str
    candidate_form: str
    similarity: float
    method: str           # "parallel" or "search"
    sentence_id: int
    sentence_length: int

@dataclass
class CognateSet:
    english: str
    pos: str
    lang1: str            # citation form from dictionary
    lang2: str            # "" if not found
    lang3: str
    lang4: str
    lang5: str
    lang6: str
    lang7: str
```

---

## Important notes for Stage 3

**Candidate forms are inflected surface tokens**, not citation (root) forms.
For example, the lang2 candidate for "student" is `þàdórúmàdórúmàgù` (the
full noun phrase with definiteness and case marking) rather than just the bare
stem.

Stage 3 (sound correspondence analysis) should treat these as-is: the shared
root is still recoverable as the LCS of the forms, and regular sound changes
apply to entire phoneme positions regardless of whether they occur in the stem
or affix.  If root-only comparison is needed, a morpheme segmentation step
should be added between Stage 2 and Stage 3.

**Coverage gaps** (empty cells in `cognate_sets.csv`) arise because a word
never appears in any sentence of that language's ~115-sentence corpus.  They
are not evidence that the word is absent from the language.

**Threshold tuning**: the thresholds `PARALLEL_THRESHOLD = 0.50` and
`SEARCH_THRESHOLD = 0.45` are conservative.  Lowering them increases recall
but also false positives.  Inspect `cognate_candidates.csv` to understand the
score distribution before changing them.
