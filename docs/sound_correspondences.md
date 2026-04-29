# sound_correspondences — Stage 3: Sound Correspondence Analysis

`src/sound_correspondences.py` reads `cognate_sets.csv` (produced by Stage 2)
and extracts the systematic character-level sound correspondences between lang1
and each of lang2–7.  The results are the direct input for Stage 4
(proto-language reconstruction).

---

## Usage

```bash
# defaults — reads data/processed/cognate_sets.csv, writes back to data/processed/
python src/sound_correspondences.py

# raise the minimum-cognate threshold for stricter filtering
python src/sound_correspondences.py --min-cognates 3

# custom data directory
python src/sound_correspondences.py --processed-dir path/to/processed
```

Sample stdout:

```
correspondences reported :   251  (min_count=2)
unique lang1 graphemes   :    29

Identity rate per language (higher = more conservative / less changed):
  lang2  60.68%  ████████████
  lang3  60.38%  ████████████
  lang4  60.02%  ████████████
  lang5  87.66%  █████████████████
  lang6  78.22%  ███████████████
  lang7  66.26%  █████████████

Top 5 non-identity correspondences per language:
  lang2: u→ú(230)  i→í(166)  a→à(110)  á→à(70)  i→ì(12)
  lang3: u→ú(165)  u→ṹ(112)  i→í(82)  a→à(64)  á→à(61)
  lang4: u→ú(157)  u→ṹ(98)   i→í(82)  a→à(61)  á→à(57)
  lang5: a→à(114)  á→à(75)   u→ú(5)   k→K(4)   l→L(3)
  lang6: u→ũ(102)  i→ĩ(91)   a→ã(67)  á→a(65)  á→ã(19)
  lang7: u→ú(131)  u→ṹ(125)  i→í(79)  i→ĩ́(43) i→ĩ(38)
```

---

## Inputs

| File | Description |
|---|---|
| `cognate_sets.csv` | One row per dictionary entry; best candidate form per language (Stage 2 output) |

---

## Algorithm

### Step 1 — Grapheme cluster splitting

Before any comparison, each word form is split into **grapheme clusters** — a
base character plus all immediately following combining diacritical marks
(Unicode category `Mn`).  Each cluster is then re-normalised to NFC.

This step is critical for correctness.  Characters like `ã̀` (a + combining
tilde U+0303 + combining grave U+0300) are a single grapheme and must be kept
together.  Using raw code-point indexing on a different-length stripped string
would silently misalign characters.

```
"dóruma"         -> ["d", "ó", "r", "u", "m", "a"]
"þàdórúmàgù"     -> ["þ", "à", "d", "ó", "r", "ú", "m", "à", "g", "ù"]
"þàdrúmã̀drúmã̀go" -> ["þ", "à", "d", "r", "ú", "m", "ã̀", "d", "r", "ú", "m", "ã̀", "g", "o"]
```

### Step 2 — LCS alignment on base characters

The two grapheme sequences are aligned by the longest common subsequence (LCS)
of their **base characters** (diacritics stripped, lowercase).  The LCS is
computed with standard O(m·n) DP, then backtracked once.

Using stripped bases for the match and original graphemes for the output means:

- `u` and `ú` match (same base `u`), recorded as the change `u → ú`
- `a` and `ã̀` match (same base `a`), recorded as `a → ã̀`
- `r` and `r` match, recorded as identity `r → r`

This correctly handles the inflected surface forms from Stage 2 by finding the
shared root as a common subsequence within the longer langN tokens.

### Step 3 — Counting and regularity

Every matched `(lang1_grapheme, langN_grapheme)` pair increments a per-language
counter.  **Regularity** is then computed as:

```
regularity(a → b, language) = count(a → b) / sum_x(count(a → x))
```

A regularity of 1.0 means the change is exceptionless: every time `a` appears
in lang1, the corresponding langN grapheme is always `b`.  Lower values
indicate split correspondences or noise.

### Step 4 — Filtering

Correspondences with `count < min_count` (default: 2) are excluded from the
output.  This removes one-off alignments that are likely artefacts of noisy
Stage 2 forms rather than genuine sound patterns.

---

## Output files

### correspondences.csv

One row per unique `(language, lang1_grapheme, langN_grapheme)` triple.

| Column | Type | Description |
|---|---|---|
| `language` | str | Target language (`lang2`–`lang7`) |
| `lang1_grapheme` | str | Grapheme as it appears in the lang1 citation form |
| `langN_grapheme` | str | Corresponding grapheme in the langN token |
| `is_identity` | bool | `True` when `lang1_grapheme == langN_grapheme` exactly |
| `count` | int | Number of cognate pairs showing this mapping |
| `regularity` | float | Fraction of lang1_grapheme occurrences mapped to this langN_grapheme |
| `examples` | str | Up to 5 English meanings where this mapping was observed |

### correspondence_table.csv

Wide-format view: one row per lang1 grapheme, one column per daughter language.
Each cell shows the top 3 mappings as `grapheme:count` entries, space-separated.
Empty cell means no mapping met the `min_count` threshold for that language.

Example rows:

| lang1_grapheme | lang2 | lang3 | lang5 | lang6 |
|---|---|---|---|---|
| `u` | `ú:230  ù:11` | `ú:165  ṹ:112` | `u:220  ú:5` | `ũ:102  u:35` |
| `a` | `à:110` | `à:64  ã̀:29` | `à:114  a:83` | `ã:67  a:52  à:18` |
| `r` | `r:196` | `r:194` | `r:199` | `r:199` |

---

## Observed sound patterns

The Stage 3 analysis reveals three main groupings:

### Group 1 — Accent shift languages: lang2, lang3, lang4, lang7
- Unaccented `u → ú`, `i → í` (acquire acute accent)
- Unaccented `a → à` (acquires grave accent)
- Accented `á → à` (acute shifts to grave)
- Consonants are fully stable

### Group 2 — Conservative language: lang5
- Highest identity rate (87.7%)
- Mostly `a → à` and `á → à`; very few other changes
- Closest to the proto-language phonological inventory

### Group 3 — Nasalisation language: lang6
- `u → ũ`, `i → ĩ`, `a → ã` — all three core vowels acquire nasal marking
- Proto-language `á → a` (accent loss alongside nasalisation)
- Intermediate identity rate (78.2%)

---

## Public API

```python
from src.sound_correspondences import (
    grapheme_split,          # str -> list[str]
    grapheme_base,           # str -> str
    lcs_align,               # list[str], list[str] -> list[tuple[str, str]]
    extract_correspondences, # list[dict] -> (counts, examples)
    compute_regularity,      # counts -> regularity dict
    build_correspondence_records,  # -> list[Correspondence]
    build_table_rows,              # -> list[TableRow]
    compute_summary,               # -> dict
    run,                           # Path, int -> dict
)
```

---

## Limitations

**Inflected surface forms from Stage 2** — the `cognate_sets.csv` forms are
surface tokens from sentences (not citation forms), so each form may carry
prefixes and suffixes in addition to the stem.  The LCS alignment naturally
focuses on the shared stem, but the unmatched affix material is simply
discarded rather than analysed.  Affix-level correspondences require a
morpheme segmentation step before Stage 3.

**Unmatched graphemes not tracked** — graphemes in lang1 that have no LCS
counterpart in langN (sound deletions, e.g. *ó → ∅ in lang3) are not recorded
in the current outputs.  They represent an important class of sound change
but require a separate gap-correspondence analysis pass.

**min_count threshold** — the default of 2 is conservative.  With ~270 cognate
pairs per language and word lengths of 5–15 graphemes, a count of 2 can still
arise by chance.  Consider raising to 5 for Stage 4 to restrict the
correspondence set to patterns with stronger evidence.
