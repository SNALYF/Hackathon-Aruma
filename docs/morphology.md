# morphology — Stage 6: Morphological Reconstruction

`src/morphology.py` extracts and documents the proto-language grammatical system
from the lang1 interlinear corpus (the `segmented` and `gloss` columns of
`corpus.csv`).

---

## Usage

```bash
python -m src.morphology
python -m src.morphology --processed-dir path/to/processed
```

Sample stdout:

```
morpheme categories  :   41
cross-lang attested  :   11

By functional category:
  case                 9
  evidentiality        5
  number               3
  agreement            2
  ...

Cross-language attested proto-morphemes:
  *mi       [WIT       ]  freq=818   in: lang2, lang3, lang4, lang5, lang6, lang7
  *i        [INAN      ]  freq=780   in: lang2, lang3, lang4, lang5, lang6, lang7
  *o        [ERG       ]  freq=499   in: lang2, lang3, lang4, lang5, lang6, lang7
  ...
```

---

## Inputs

| File | Description |
|---|---|
| `corpus.csv` | Lang1 interlinear sentences with `segmented` and `gloss` columns (Stage 1) |

---

## Algorithm

### Step 1 — Morpheme extraction

For every lang1 corpus row with non-empty `segmented` and `gloss` fields:

1. Split both fields on whitespace to get parallel token lists
2. For each (segmented_token, gloss_token) pair, split on `-` to get individual morphemes
3. Classify each morpheme as **grammatical** (all-uppercase gloss, e.g. `WIT`, `ERG`) or **lexical** (lowercase gloss)
4. Count (form, gloss_tag) co-occurrences

### Step 2 — Cross-language attestation

For each grammatical morpheme, a regex pattern is applied to every surface token
in the lang2–7 corpus.  A morpheme is marked as **cross-language attested** if
the pattern matches ≥ 5 tokens in at least one daughter language.

The patterns look for characteristic phonological signatures:

| Morpheme | Pattern | Rationale |
|---|---|---|
| WIT (`*-mi`) | `m[iíĩ]$` at suffix | Final `-mi` with vowel variation |
| INAN (`*i-`) | `^[iíĩ]` at prefix | Initial `i-` |
| ERG (`*-o`) | `[oeóé]$` at suffix | Final open vowel |
| INESS (`*-nu`) | `n[uúũ]$` | Final `-nu` |
| FUT (`*-ke`) | `[kg][eé]?$` | Final velar stop |

### Step 3 — Inventory and grammar generation

The morpheme inventory table and the prose grammar document are generated
from the extracted counts and cross-language evidence.

---

## Output files

### morpheme_inventory.csv

One row per (gloss_tag, canonical_form) pair with frequency ≥ 2.

| Column | Description |
|---|---|
| `gloss_tag` | Grammatical abbreviation (e.g. `WIT`, `ERG`) |
| `canonical_form` | Most frequent surface form for this tag |
| `category` | Functional category (`case`, `evidentiality`, `tense`, …) |
| `frequency` | Total occurrences in lang1 corpus |
| `n_forms` | Number of distinct surface realizations |
| `all_forms` | Top 5 forms with counts |
| `cross_lang_attested` | True if pattern found in lang2–7 |
| `attested_in` | Daughter languages where attested |
| `example_word` | Example full surface token from lang1 |

### proto_grammar.md

Human-readable grammar written to `data/processed/`.  Sections:
1. Overview and morphological template
2. Nominal morphology (case, number, possession)
3. Verbal morphology (agreement, causative, negation, evidentiality/TAM)
4. Person/reference system
5. Particles and other categories
6. Proto-morpheme summary table
7. Sample morphological analyses
8. Limitations

---

## Key findings

### Morphological template

```
VERB:  [AGR]-ROOT[-CAUS[-CAUS]][-NEG]-TAM
NOUN:  ROOT[-PL][-DEF][-CASE]
```

The language is **polysynthetic and head-marking**: verbs are heavily inflected
with agreement, valence, and evidentiality; nouns carry case and definiteness.

### 11 proto-morphemes with cross-language evidence

All 6 daughter languages attest the core verbal morphemes:

| Proto-form | Function | Attested in |
|---|---|---|
| `*-mi` | Witnessed evidential (WIT) | All 6 daughters |
| `*i-` | Inanimate agreement (INAN) | All 6 daughters |
| `*-o` / `*-e` | Ergative case (ERG) | All 6 daughters |
| `*-nu` | Inessive case (INESS) | All 6 daughters |
| `*-ke` / `*-ge` | Future tense (FUT) | All 6 daughters |
| `*-li` | Elative case (ELAT) | All 6 daughters |
| `*-lu` | Illative case (ILL) | All 6 daughters |
| `*a-` | Animate agreement (ANIM) | 5 daughters |
| `*-su-` | Causative (CAUS) | 3 daughters |
| `*-me` | Inferential evidential (INFER) | 3 daughters |
| `*-ne` | Negation (NEG) | 2 daughters |

### The WIT morpheme `*-mi` is the most robustly attested proto-morpheme

It occurs 818 times in lang1 and appears in all six daughters with forms
`-mi`, `-mí`, `-mĩ`, `-mĩ́` — all fully predictable from the Stage 3 vowel
correspondence `*i → í` (lang2) and `*i → ĩ` (lang6).

### 9-case nominal system

ERG, INESS, ELAT, ILL, BEN, POSS, INALIEN, DEF, DEF.NEAR — a rich case system
with both spatial (ELAT, ILL, INESS) and grammatical (ERG, BEN) distinctions.

### Four-way evidentiality

WIT (witnessed) · FUT (future/irrealis) · INFER (inferential) · DIR.EVID
(direct non-visual evidence) — with a separate mirative particle `há`.

---

## Relationship to earlier stages

| Stage | Phonological | Morphological |
|---|---|---|
| Stage 3 | Sound correspondences | — |
| Stage 4 | Proto-lexicon | — |
| Stage 5 | Sound law validation | — |
| **Stage 6** | — | **Morpheme inventory + proto-grammar** |

The morphological reconstruction complements the phonological one: where
Stages 3–5 established *what sounds* the proto-language had and how they
changed, Stage 6 establishes *how words were built* and which grammatical
categories were present in the proto-language.
