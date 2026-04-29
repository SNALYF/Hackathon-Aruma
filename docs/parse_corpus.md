# parse_corpus — Stage 1: Raw Data Normalisation

`src/parse_corpus.py` reads the seven raw daughter-language files from
`data/raw/Hackaton 2026/` and writes three structured CSV files to
`data/processed/`.  It is the first step in the proto-language reconstruction
pipeline.

---

## Usage

```bash
# use defaults (reads from data/raw/, writes to data/processed/)
python src/parse_corpus.py

# override paths
python src/parse_corpus.py --raw-dir path/to/raw --out-dir path/to/out
```

Running the script prints a summary count to stdout:

```
dictionary :  579 entries
corpus     :  716 sentences
docs       :  168 lines
```

---

## Input files

All input files live in `data/raw/Hackaton 2026/`.

| File pattern | Present for | Description |
|---|---|---|
| `lang1-dictionary.txt` | lang1 only | Full vocabulary list |
| `langN-corpus.txt` | lang1–7 | Interlinear glossed sentences |
| `langN-doc{1,2}.txt` | lang1–7 | Continuous prose documents |

### lang1-dictionary.txt

Tab-separated (or 2+ spaces) with three fields per line:

```
english_meaning    native_form    POS
```

The `native_form` may carry a parenthetical compound annotation:

```
astronomer    fálurunadóruma (ie, star-scholar)    NN
```

Lines with fewer than three parseable fields are skipped.

### langN-corpus.txt — lang1 (4-line interlinear blocks)

Blocks are separated by one or more blank lines.  Each block has exactly
four non-empty lines:

```
Mógogo línenili Aarumalu nie iáramami.
mógo-ko líneni-li Aaruma-lu-ru ni-ERG i-árama-mi
world-DEF.NEAR linguistics-ELAT Aaruma-ELAT-ILL 1SG-ERG INAN-welcome-WIT
"(I) welcome (you) into the world of Aaruma linguistics."
```

| Line | Field | Notes |
|---|---|---|
| 1 | `surface` | Orthographic sentence as written |
| 2 | `segmented` | Morpheme boundaries marked with `-` |
| 3 | `gloss` | Leipzig glossing abbreviations per morpheme |
| 4 | `translation` | English free translation, usually in double quotes |

### langN-corpus.txt — lang2–7 (2-line blocks)

```
Lííngírìù tóràlúnàrúmàgà nàrúmànàrúmàlú púúrúmúnú llímíréngàmí ídàlímíngíngímí.
"Who made the actor tell the secret message to the people?"
```

| Line | Field |
|---|---|
| 1 | `surface` |
| 2 | `translation` (quotes are optional in the file) |

Blocks with fewer than two non-empty lines, or where the translation is
empty after stripping quotes, are silently skipped.  This handles a small
number of incomplete morpheme-only entries present in some files.

### langN-doc{1,2}.txt

Continuous prose; no glosses or translations.  Each non-empty line is
treated as one entry.  The seven languages appear to contain parallel
versions of the same two texts.

---

## Output files

All output files are written to `data/processed/` as UTF-8 CSV.

### dictionary.csv

| Column | Type | Description |
|---|---|---|
| `english` | str | English meaning |
| `native_form` | str | Native word (compound explanation stripped) |
| `compound_explanation` | str | Content of `(ie, ...)` annotation; empty when absent |
| `pos` | str | Penn Treebank part-of-speech tag (NN, VB, JJ, …) |

### corpus.csv

| Column | Type | Description |
|---|---|---|
| `language` | str | Source language identifier (`lang1`–`lang7`) |
| `sentence_id` | int | 1-based index within the language |
| `surface` | str | Orthographic sentence |
| `segmented` | str | Morpheme-segmented form; empty for lang2–7 |
| `gloss` | str | Morpheme gloss; empty for lang2–7 |
| `translation` | str | English free translation |

### docs.csv

| Column | Type | Description |
|---|---|---|
| `language` | str | Source language identifier |
| `doc_id` | int | Document number (1 or 2) |
| `line_no` | int | 1-based line number in the original file |
| `text` | str | Non-empty line of prose |

---

## Public API

The module exposes these functions for use by downstream pipeline steps:

```python
from src.parse_corpus import (
    parse_dictionary,   # Path -> list[DictionaryEntry]
    parse_corpus,       # Path, language -> list[CorpusEntry]
    parse_doc,          # Path, language, doc_id -> list[DocEntry]
    export_csv,         # list[dataclass], Path -> None
    parse_all,          # (raw_dir, processed_dir) -> dict
)
```

### `parse_dictionary(path) -> list[DictionaryEntry]`

Returns one `DictionaryEntry` per valid line.

```python
@dataclass
class DictionaryEntry:
    english: str
    native_form: str
    compound_explanation: str   # '' when no annotation
    pos: str
```

### `parse_corpus(path, language) -> list[CorpusEntry]`

Returns one `CorpusEntry` per accepted sentence block.

```python
@dataclass
class CorpusEntry:
    language: str
    sentence_id: int
    surface: str
    segmented: str      # '' for lang2-7
    gloss: str          # '' for lang2-7
    translation: str
```

### `parse_doc(path, language, doc_id) -> list[DocEntry]`

Returns one `DocEntry` per non-empty line.

```python
@dataclass
class DocEntry:
    language: str
    doc_id: int
    line_no: int
    text: str
```

### `parse_all(raw_dir, processed_dir) -> dict`

Convenience function that parses all files and writes all three CSVs.
Returns `{'dictionary': [...], 'corpus': [...], 'docs': [...]}`.

---

## Edge cases

| Situation | Handling |
|---|---|
| Dictionary line with fewer than 3 fields | Skipped |
| Dictionary `native_form` with `(ie, ...)` | Annotation moved to `compound_explanation` |
| lang1 corpus block with fewer than 4 lines | Skipped |
| lang2–7 corpus block with fewer than 2 lines | Skipped (partial morpheme-only lines) |
| Translation with no surrounding quotes | Accepted — quotes are stripped when present |
| Empty translation after stripping quotes | Skipped |
| Blank lines within a doc file | Skipped |
