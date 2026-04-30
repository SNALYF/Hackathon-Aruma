# reconstruct_proto — Stage 4: Proto-Language Reconstruction

`src/reconstruct_proto.py` synthesises the outputs of Stages 1–3 into a
reconstructed proto-language — a set of explicit sound laws, a *proto-form
for every dictionary entry, and a human-readable summary of findings.

---

## Usage

```bash
# defaults — reads from data/processed/, writes back to data/processed/
python src/reconstruct_proto.py

# stricter thresholds for high-confidence laws only
python src/reconstruct_proto.py --regularity 0.85 --min-count 10

# custom directory
python src/reconstruct_proto.py --processed-dir path/to/processed
```

Sample stdout:

```
sound laws      :  133  (changes: 9, conservation: 124)
proto-lexicon   :  578  (high: 229, medium: 168, low: 181)

Sound changes per language:
  lang2   4 changes  19 stable
  lang3   1 changes  20 stable
  ...

Top sound changes across all languages:
  lang2: *u → ú  reg=95%  n=230
  lang2: *i → í  reg=93%  n=166
  lang5: *a → à  reg=79%  n=114
  ...
```

---

## Inputs

| File | Description |
|---|---|
| `correspondences.csv` | Per-language (lang1_grapheme, langN_grapheme) counts and regularity (Stage 3) |
| `cognate_sets.csv` | Best candidate form per language per meaning (Stage 2) |
| `dictionary.csv` | Lang1 citation forms with POS (Stage 1) |

---

## Reconstruction strategy

Lang1 citation forms (from the dictionary) are used as the direct basis
for all proto-forms:

```
*proto_form = "*" + lang1_citation_form
```

This is justified by two observations:
1. Lang1 has the richest documentation: a 578-entry dictionary with POS
   tags and compound-word annotations, plus a full interlinear corpus.
2. Lang5 — the most phonologically conservative daughter (87.7 % grapheme
   identity with lang1) — preserves lang1 forms almost entirely, confirming
   lang1 as a reliable proxy for the ancestral state.

Sound laws are therefore read as **changes FROM the proto-language (= lang1)
TO each daughter language**.

---

## Algorithm

### Step 1 — Sound law extraction

Every row in `correspondences.csv` that satisfies both:

- `regularity ≥ REGULARITY_THRESHOLD` (default 0.70)
- `count ≥ COUNT_THRESHOLD` (default 5)

becomes a sound law.  Both identity correspondences (phoneme preserved) and
non-identity correspondences (phoneme changed) are included so that the law
table explicitly documents which sounds were **stable** as well as which
**changed** in each daughter.

### Step 2 — Proto-lexicon construction

For each of the 578 dictionary entries:

1. Prepend `*` to the lang1 citation form → `proto_form`
2. Look up the cognate set row for this meaning
3. Count how many of lang2–7 have a non-empty cognate form → `n_cognates`
4. Assign a **confidence tier**:

| Tier | n_cognates | Interpretation |
|---|---|---|
| high | ≥ 4 | Proto-form corroborated by majority of daughters |
| medium | 2–3 | Partial corroboration |
| low | 0–1 | Proto-form rests on lang1 alone |

### Step 3 — Summary generation

`reconstruction_summary.md` is generated programmatically and contains:
- Proto-phoneme inventory table (vowels with stability notes; consonants)
- Sound law tables per daughter language
- A sample of 20 high-confidence reconstructed words
- Coverage statistics
- Candidate proto-form corrections (where lang1 may have innovated)

---

## Output files

### sound_laws.csv

One row per regular correspondence above both thresholds.

| Column | Type | Description |
|---|---|---|
| `language` | str | Daughter language (`lang2`–`lang7`) |
| `proto_grapheme` | str | Grapheme in the proto-language (= lang1) |
| `reflex` | str | Corresponding grapheme in the daughter language |
| `is_change` | bool | `False` = preserved (identity); `True` = changed |
| `regularity` | float | Fraction of proto_grapheme occurrences → this reflex |
| `count` | int | Number of cognate pairs supporting this law |
| `examples` | str | Up to 5 English meanings where this was observed |

### proto_lexicon.csv

One row per dictionary entry.

| Column | Type | Description |
|---|---|---|
| `english` | str | English meaning |
| `pos` | str | Part-of-speech tag |
| `proto_form` | str | Reconstructed proto-form (e.g. `*dóruma`) |
| `confidence` | str | `high` / `medium` / `low` |
| `n_cognates` | int | Number of daughter languages with a cognate |
| `lang1`–`lang7` | str | Form in each language (`""` if not found) |

### reconstruction_summary.md

Human-readable Markdown file written to `data/processed/`.  Contents:
- Overview and confidence distribution
- Proto-phoneme inventory (vowels and consonants)
- Sound-law tables per daughter language
- 20-entry sample lexicon
- Coverage statistics
- Candidate proto-form corrections
- Limitations

---

## Key findings

### Sound changes (9 non-identity laws above threshold)

| Law | Languages | Count | Notes |
|---|---|---|---|
| *u → ú | lang2 | 230 | Bare vowel raises to acute in lang2 |
| *i → í | lang2 | 166 | Same pattern |
| *a → à | lang2, lang5 | 110, 114 | Bare low vowel acquires grave accent |
| *á → à | lang2, lang3, lang4, lang5, lang6 | 70–75 | Acute low vowel lowers to grave |
| *á → a | lang6 | 65 | Lang6 drops accent entirely |

### Stability (124 conservation laws)

Every consonant is stable across all six daughters.  Accented vowels
`é`, `ó`, `í`, `ú` are preserved with ≥ 95 % regularity in all languages.

### Candidate lang1 innovations

Stage 3 and Stage 4 together identify two places where lang1 itself may
have moved away from the proto-language:

| Lang1 grapheme | Likely proto | Basis |
|---|---|---|
| `á` | `*à` | All 5 attested daughters change `á → à` (100 % regularity) |
| `a` | `*à` | Majority of daughters map `a → à`; lang1 may have dropped the grave |

If these corrections are applied, proto-forms containing `a` or `á` should be
written with `à` instead.  The `proto_lexicon.csv` currently retains the lang1
forms for traceability; Stage 5 can apply a correction pass.

---

## Public API

```python
from src.reconstruct_proto import (
    extract_sound_laws,    # list[dict] -> list[SoundLaw]
    build_proto_lexicon,   # list[dict], list[dict] -> list[ProtoEntry]
    generate_summary,      # laws, entries, corrs -> str
    run,                   # Path -> dict
)
```

---

## Tuning the thresholds

| Parameter | Default | Effect of raising |
|---|---|---|
| `--regularity` | 0.70 | Fewer but higher-confidence sound laws |
| `--min-count` | 5 | Laws must be supported by more cognate pairs |

For Stage 5 validation, running with `--regularity 0.85 --min-count 10`
gives a stricter set of laws to test against new data.
