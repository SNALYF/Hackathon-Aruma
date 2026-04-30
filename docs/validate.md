# validate — Stage 5: Reconstruction Validation

`src/validate.py` tests whether the proto-language reconstruction produced by
Stages 1–4 is internally consistent and predictively useful.  It answers the
question: *"Do our sound laws generalise, or do they merely describe the data
they were derived from?"*

---

## Usage

```bash
python -m src.validate
python -m src.validate --processed-dir path/to/processed
```

Sample stdout:

```
predictions evaluated :  1701
exact (≥ 0.95)        :   52.1 %
close or better       :   84.4 %
miss (< 0.50)         :    2.6 %
avg predicted coverage:  0.875

Per-language accuracy (close or better %):
  lang2   82.1 %  ████████████████
  lang3   89.0 %  █████████████████
  lang4   89.5 %  █████████████████
  lang5   87.7 %  █████████████████
  lang6   89.9 %  █████████████████
  lang7   66.4 %  █████████████

Exceptions: 44  (alignment_noise: 2, conditioned: 4, irregular: 38)
```

---

## Inputs

| File | Description |
|---|---|
| `sound_laws.csv` | Proposed sound change rules (Stage 4) |
| `proto_lexicon.csv` | Reconstructed proto-forms with daughter cognates (Stage 4) |
| `corpus.csv` | Lang1 interlinear sentences for proto-text generation (Stage 1) |

---

## Algorithm

### Step 1 — Sound law application

For each entry in `proto_lexicon.csv`, the proto-form is split into grapheme
clusters.  For each grapheme, the highest-regularity sound law for the target
language is applied.  If no law exists for a grapheme, it is kept unchanged
(the default conservation assumption).

```
predict("*dóruma", "lang2") → "dórúmà"
    d → d  (no law → keep)
    ó → ó  (no law → keep)
    r → r  (no law → keep)
    u → ú  (law: *u → ú in lang2, reg=0.95)
    m → m  (no law → keep)
    a → à  (law: *a → à in lang2, reg=1.00)
```

### Step 2 — Predicted coverage metric

Stage 2 cognate forms are inflected surface tokens that may include prefixes,
suffixes, and reduplication around the stem.  The predicted form is only the
bare stem.  Symmetric LCS similarity underestimates prediction quality in this
situation.

**Predicted coverage** is used instead:

```
coverage = LCS_length(predicted_graphemes, actual_graphemes)
           / len(predicted_graphemes)
```

A coverage of 1.0 means the entire predicted stem appears as a subsequence
inside the actual (inflected) form — regardless of what affixes surround it.

### Step 3 — Match classification

| Class | Coverage threshold | Interpretation |
|---|---|---|
| exact | ≥ 0.95 | Prediction fully embedded in actual form |
| close | ≥ 0.75 | Minor divergence (one unseen grapheme change) |
| partial | ≥ 0.50 | Partially correct |
| miss | < 0.50 | Prediction fails — see exception analysis |

### Step 4 — Exception analysis

Miss-class predictions are assigned one of three diagnoses:

| Diagnosis | Detection heuristic |
|---|---|
| `alignment_noise` | Actual form is ≥ 3× longer than predicted (Stage 2 captured a multi-word surface complex) |
| `conditioned` | The first mismatched grapheme is a vowel (unresolved split correspondence) |
| `irregular` | Everything else — genuine exception, suppletive form, or Stage 2 error |

### Step 5 — Proto-text generation

Five lang1 sentences with interlinear glosses are selected.  For each token,
the content-word gloss is used to look up the corresponding `*proto_form`.
Grammatical tokens (with all-uppercase glosses like `ERG`, `PL`, `WIT`) keep
their lang1 surface form.  Sentences with fewer than 2 mapped tokens are
skipped.

---

## Output files

### predictions.csv

One row per (proto-entry, language) pair where the actual cognate is non-empty.

| Column | Type | Description |
|---|---|---|
| `english` | str | Dictionary English meaning |
| `pos` | str | Part-of-speech tag |
| `proto_form` | str | Reconstructed proto-form (e.g. `*dóruma`) |
| `language` | str | Target language (`lang2`–`lang7`) |
| `predicted_form` | str | Form produced by applying sound laws |
| `actual_form` | str | Actual cognate from `cognate_sets.csv` |
| `raw_similarity` | float | Symmetric LCS similarity |
| `predicted_coverage` | float | Fraction of predicted stem found in actual |
| `match_class` | str | `exact` / `close` / `partial` / `miss` |

### exceptions.csv

All miss-class rows with additional diagnostic fields.

| Column | Description |
|---|---|
| (inherits prediction columns) | |
| `mismatched_grapheme` | First predicted grapheme not found in actual |
| `diagnosis` | `alignment_noise` / `conditioned` / `irregular` |

### validation_report.md

Full Markdown report written to `data/processed/`.  Sections:
- Overall and per-language accuracy tables
- Best and worst predictions
- Exception analysis with diagnosis breakdown
- 5 sample proto-language sentences
- Overall assessment

---

## Key findings

### Accuracy

| Language | Close or better | Miss |
|---|---|---|
| lang6 | 89.9 % | 3.9 % |
| lang4 | 89.5 % | 1.5 % |
| lang3 | 89.0 % | 2.1 % |
| lang5 | 87.7 % | 2.4 % |
| lang2 | 82.1 % | 3.7 % |
| lang7 | 66.4 % | 1.9 % |
| **overall** | **84.4 %** | **2.6 %** |

Lang7's lower close-or-better rate (66.4 %) is explained by its unresolved
`*u → ú / ṹ` split (roughly 50/50 in Stage 3): the laws map `*u → ú`
(higher regularity), but half the actual lang7 forms have `ṹ`, pushing many
predictions into the partial class rather than exact.

### Exceptions (44 misses)

- **38 irregular** — include suppletive numerals (`*vési` / "thirteen" → `hàlú`
  in lang2/3/6), loanwords, and probable Stage 2 alignment errors.
- **4 conditioned** — vowel mismatches where the conditioning environment for a
  known split has not been identified.
- **2 alignment noise** — Stage 2 retrieved a multi-word surface complex
  instead of just the target root.

The most frequently mismatched proto-grapheme in misses is `v` (15 cases),
suggesting either a `*v → ∅` lenition in some languages not yet captured, or
a systematic Stage 2 error for `v`-initial verbs that acquire an `i-` agreement
prefix (making the `v` non-initial and harder to align).

---

## Public API

```python
from src.validate import (
    build_laws_map,          # list[dict] -> dict
    predict_form,            # str, str, dict -> str
    predicted_coverage,      # str, str -> float
    match_class,             # float -> str
    evaluate_predictions,    # list[dict], dict -> list[PredictionResult]
    analyze_exceptions,      # list[PredictionResult] -> list[ExceptionRecord]
    generate_proto_texts,    # list[dict], list[dict] -> list[dict]
    compute_accuracy_stats,  # list[PredictionResult] -> dict
    run,                     # Path -> dict
)
```
