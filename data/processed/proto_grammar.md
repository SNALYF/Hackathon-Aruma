# Proto-Language Grammar

Reconstructed from the lang1 interlinear corpus (717 sentences, 8 546
morpheme-gloss pairs) by `src/morphology.py` (Stage 6).

Lang1 exhibits a **polysynthetic, head-marking morphology** with
extensive verb agreement, a nine-case nominal system, and a four-way
evidentiality system.  The morphological template is:

```
VERB:  [AGREEMENT]-ROOT[-CAUS][-CAUS][-NEG]-TAM
NOUN:  ROOT[-PL][-DEF][-CASE]
```

- **32** grammatical morpheme categories identified
- **11** of these are attested in lang2–7 corpus data,
  supporting their reconstruction as proto-morphemes

---

## 1. Nominal Morphology

### 1.1 Case system

The proto-language had at least **nine cases**, all expressed as
suffixes on the noun root.  Case stacks after the definiteness suffix.

| Case | Function | Forms (n) | Attested in lang2–7 | Notes |
| --- | --- | --- | --- | --- |
| ERG | Ergative (agent of transitive) | `o`(263), `e`(152), `ke`(48) | lang2, lang3, lang4, lang5, lang6, lang7 | subject of causative / agent NP |
| INESS | Inessive (location in) | `nu`(223), `ni`(2), `xóronu`(2) | lang2, lang3, lang4, lang5, lang6, lang7 | static location |
| ELAT | Elative (motion from) | `li`(57), `lu`(15), `sérimi`(4) | lang2, lang3, lang4, lang5, lang6, lang7 | source of motion / topic |
| ILL | Illative (motion into) | `lu`(71), `sám`(2), `ru`(1) | lang2, lang3, lang4, lang5, lang6, lang7 | goal of motion |
| BEN | Benefactive | `bi`(22), `véri`(8), `géni`(4) | — | recipient / beneficiary |
| POSS | Possessive | `li`(67), `e`(9), `o`(1) | — | alienable possession |
| INALIEN | Inalienable poss. | `ta`(20), `da`(9), `li`(1) | — | body parts, kinship |
| DEF | Definite | `ko`(355), `o`(32), `góruamu`(5) | — | definite / referential NP |
| DEF.NEAR | Definite proximal | `ko`(72), `ke`(9), `máguzúra`(1) | — | proximal demonstrative |

### 1.2 Number

Plurality is expressed by **reduplication of the noun root**, not by a
separate suffix.  The dual is marked by the prefix `po-`.

| Category | Marking | Forms (n) | Attested in |
| --- | --- | --- | --- |
| PL | Root reduplication | `dóruma`(49), `múru`(17), `náruma`(10) | — |
| DUAL | Prefix `po-` | `po`(13) | — |

### 1.3 Possession

Two possession types are distinguished:

- **Alienable** (`POSS`): suffix `-li` / `-lu` — things that can be transferred
- **Inalienable** (`INALIEN`): suffix `-ta` / `-da` — body parts, kinship terms

---

## 2. Verbal Morphology

### 2.1 Agreement prefixes

Every finite verb carries an agreement prefix encoding the animacy class
of its absolutive argument:

| Prefix | Gloss | Canonical form | Frequency | Attested in |
| --- | --- | --- | --- | --- |
| INAN | Inanimate argument | `i`(754), `te`(6) | 780 | lang2, lang3, lang4, lang5, lang6, lang7 |
| ANIM | Animate argument | `a`(132), `u`(19) | 156 | lang2, lang3, lang4, lang5, lang6 |

The INAN prefix `i-` and ANIM prefix `a-` are the most frequent morphemes
in the corpus.  Both are attested in lang2–7 surface tokens (matching the
patterns `i-...` and `a-...`).

### 2.2 Causative

The causative suffix `*-su-` (CAUS, n=140)
is inserted between the verb root and the TAM suffix.  It can stack
(**double causative**: make-someone-make-someone-do):

```
INAN-create-CAUS-CAUS-WIT  'caused to cause to create'
ANIM-be-CAUS-WIT           'caused to be (= became)'
```

### 2.3 Negation

Negation suffix `*-ne-` / `*-no-` / `*-ni-` (NEG, n=39)
precedes the TAM suffix and follows any causative:

```
INAN-speak-NEG-WIT   'did not speak'
OBJ-NEG              'no object / nothing'
```

### 2.4 Evidentiality / TAM system

The final position on the verb encodes both tense and evidentiality in
a **four-way evidential system**:

| TAM | Meaning | Proto-form | Attested forms (n) | Freq | In lang2–7 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| WIT | Witnessed | `*-mi` | `mi`(800), `ERG`(6), `mu`(4) | 818 | lang2, lang3, lang4, lang5, lang6, lang7 | Default past/present; speaker witnessed event |
| FUT | Future | `*-ke` / `*-ge` | `ke`(47), `ge`(36), `go`(2) | 91 | lang2, lang3, lang4, lang5, lang6, lang7 | Future or irrealis |
| INFER | Inferential | `*-me` | `me`(50) | 50 | lang3, lang5, lang6 | Speaker infers from evidence |
| DIR.EVID | Direct evidence | `*-me` | `me`(20) | 20 | — | Speaker has direct non-visual evidence |
| MIR | Mirative | `há` | `há`(21), `há!`(1) | 22 | — | Surprise / unexpected new information (particle) |

The WIT suffix `*-mi` is the most robustly attested proto-morpheme:
it appears in all six non-lang1 daughter languages with minor phonological
variation (`-mi`, `-mí`, `-mĩ`, `-mĩ́`) fully consistent with the vowel
correspondences established in Stage 3.

---

## 3. Person / Reference System

Personal reference is expressed through free pronouns (not agreement
prefixes) combined with the animacy-based verb prefix:

| Person | Form | Attested variants | Freq | Notes |
| --- | --- | --- | --- | --- |
| 1SG | `ni` |  | 0 | First person singular |
| 1PL.INCL | `ni te` |  | 0 | First person plural inclusive |
| 1PL.POSS | `teli` |  | 0 | First person plural possessive |
| 2SG.FORMAL | `tíme` |  | 0 | Second person singular formal |
| DUAL | `po` | `po`(13) | 13 | Dual (two referents) |

---

## 4. Particles and Other Categories

| Tag | Category | Canonical form | Freq | Function |
| --- | --- | --- | --- | --- |
| MIR | Mirative particle | `há` | 22 | Sentence-final; marks surprise or new information |
| Q.POLAR | Polar question | `lú` | 11 | Sentence-final; marks yes/no question |
| INSTR | Instrumental | `géni` | 23 | Adposition 'by means of / using' |
| BEN | Benefactive | `bi-` / `véri` | 35 | Prefix or free morpheme 'for the benefit of' |
| PURP | Purposive | `-li` | 12 | Suffix on nominalized verb 'in order to' |

---

## 5. Proto-Morpheme Summary

Morphemes for which we have cross-language attestation evidence
(found in ≥ 5 tokens per language in at least one daughter):

| Proto-form | Gloss | Category | Lang1 freq | Daughter languages |
| --- | --- | --- | --- | --- |
| `*-mi` | WIT | evidentiality | 818 | lang2, lang3, lang4, lang5, lang6, lang7 |
| `*i-` | INAN | agreement | 780 | lang2, lang3, lang4, lang5, lang6, lang7 |
| `*-o` | ERG | case | 499 | lang2, lang3, lang4, lang5, lang6, lang7 |
| `*-nu` | INESS | case | 233 | lang2, lang3, lang4, lang5, lang6, lang7 |
| `*a-` | ANIM | agreement | 156 | lang2, lang3, lang4, lang5, lang6 |
| `*-su` | CAUS | valence | 140 | lang3, lang4, lang6 |
| `*-ke` | FUT | tense | 91 | lang2, lang3, lang4, lang5, lang6, lang7 |
| `*-li` | ELAT | case | 86 | lang2, lang3, lang4, lang5, lang6, lang7 |
| `*-lu` | ILL | case | 79 | lang2, lang3, lang4, lang5, lang6, lang7 |
| `*-me` | INFER | evidentiality | 50 | lang3, lang5, lang6 |
| `*-ne` | NEG | negation | 39 | lang3, lang5 |

---

## 6. Sample Morphological Analysis

Three example sentences illustrating the morphological system:

### Example 1

**Surface**: ﻿CL dórumadórumagoo lóruma góruamumu, hére zúrumago lóruma.

**Segmented**: `CL dóruma-dóruma-ko-o lóruma gó-ru-a-mu-mu, hére zú-ru-ma-go lóruma`

**Gloss**: `CL student-PL-DEF-ERG language create-WIT, now test-FUT language`

**Translation**: The CL students have created a language; it's now time to test it.

### Example 2

**Surface**: Mógogo línenili Aarumalu nie iáramami.

**Segmented**: `mógo-ko líneni-li Aaruma-lu-ru ni-ERG i-árama-mi`

**Gloss**: `world-DEF.NEAR linguistics-ELAT Aaruma-ELAT-ILL 1SG-ERG INAN-welcome-WIT`

**Translation**: (I) welcome (you) into the world of Aaruma linguistics.

### Example 3

**Surface**: Kogo dénilili hírinili Aarumalu izamimi.

**Segmented**: `ko-ko dénili-li hírini-li Aaruma-lu i-sami-mi`

**Gloss**: `this documentation-ELAT history-ELAT Aaruma-ELAT INAN-be-WIT`

**Translation**: This is the documentation of the history of Aaruma.

---

## 7. Limitations

- **Only lang1 has interlinear data.** The morphological analysis rests
  entirely on lang1 evidence.  Cross-language attestation uses surface
  pattern matching, not interlinear glosses, and can only confirm that
  a morpheme *exists* in a daughter — not its precise function.

- **Morphological conditioning not analysed.** Several suffixes have
  multiple surface forms (e.g. ERG: `-o`, `-e`, `-ke`) whose distribution
  is likely phonologically conditioned but is not modelled here.

- **Derivational morphology partially covered.** Compound-word formation
  (documented in the Stage 1 dictionary's `compound_explanation` field)
  is not addressed here; it warrants a separate analysis.
