# Model card: character-trigram spell checkers for 21 Niger-Congo B languages

This card follows the Hugging Face model card template.

## Model details

### Model description

The trained models of the character-trigram spell checker, for 25 corpora of 21 Niger-Congo B
languages. Every corpus has three files, and the three together make one checker for that
language.

| File | What it holds | Used for |
|---|---|---|
| `<folder>_trigram.pkl` | counts of each character trigram after the two trigrams before it | scoring a word (detection) |
| `<folder>_word.pkl` | every word of the training split with its frequency | suggesting corrections |
| `<folder>_classifier.pkl` | a one-dimensional logistic regression on the score | the threshold of the language |

- **Developed by:** Leruo Hillary Sharp, supervised by C. Maria Keet
- **Model type:** character-trigram count model with a logistic-regression threshold, and a word-frequency model for correction
- **Languages:** the 21 languages of the table below

### Model sources

- **Repository:** this repository
- **Demo:** the desktop application, `app.py`

## Uses

### Direct use

Detecting and correcting spelling errors in text of one of the 21 languages, through the desktop
application or through `SpellChecker` in `src/core/spell_checker.py`.

### Downstream use

A model can be applied to a closely related language that has no model of its own. It keeps most
of its accuracy only on a language of the same Guthrie group.

### Out-of-scope use

The models judge one word at a time. They do not check grammar or word choice, and they are not
language models for generating text.

## Bias, risks and limitations

- The errors used for the threshold and for the evaluation are synthetic single edits, which may
  not reflect the errors people make.
- A word is judged on its own, so a misspelling that is itself a valid word is not detected.
- A word of fewer than three characters has no trigram. The checker skips such words by default.
- A model reflects its corpus. Several corpora are small, come from one domain, or were mined from
  the web and contain noise, and the Silozi corpus is mostly Biblical text.
- Detection is weaker on languages that are written disjunctively, such as Sesotho and Setswana.

### Recommendations

A flagged word is a suggestion to look again, not a verdict. Rare but valid words are the most
common false alarms.

## How to get started with the model

```python
from pathlib import Path
from src.models.trigram_model import TrigramModel

model = TrigramModel.load(Path("data/models/zu_trigram.pkl"))
print(model.get_info())
print(model.score_word("abantu"))
```

A `.pkl` file is a Python pickle and can run code when it is loaded, so load only files from a
source you trust.

## Training details

### Training data

| Folder | Language | Guthrie code | Corpus | Licence of the corpus | Training tokens | Word types | Distinct trigrams | tau | Macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| `zu` | isiZulu | S42 | NCHLT Text Corpora 2.0 | CC BY 2.5 ZA | 3,691,480 | 249,501 | 10,240 | -4.03 | 0.871 |
| `xh` | isiXhosa | S41 | NCHLT Text Corpora 2.0 | CC BY 2.5 ZA | 3,441,801 | 209,673 | 9,779 | -4.40 | 0.856 |
| `nd` | isiNdebele | S44 | NCHLT Text Corpora 2.0 | CC BY 2.5 ZA | 2,392,132 | 156,372 | 9,335 | -4.17 | 0.878 |
| `sw` | siSwati | S43 | NCHLT Text Corpora 2.0 | CC BY 2.5 ZA | 2,353,559 | 174,831 | 9,209 | -4.11 | 0.883 |
| `sotho` | Sesotho | S33 | NCHLT Text Corpora 2.0 | CC BY 2.5 ZA | 4,301,705 | 82,275 | 9,131 | -4.68 | 0.761 |
| `tswana` | Setswana | S31 | NCHLT Text Corpora 2.0 | CC BY 2.5 ZA | 3,292,811 | 83,983 | 9,248 | -4.72 | 0.749 |
| `zulu` | isiZulu | S42 | SADiLaR enriched corpora 1.0.0 and AST isiZulu Text Corpus | CC BY 4.0 and CC BY 2.5 ZA | 365,987 | 85,761 | 6,584 | -5.07 | 0.884 |
| `xhosa` | isiXhosa | S41 | SADiLaR enriched corpora 1.0.0 | CC BY 4.0 | 33,673 | 13,219 | 2,762 | -5.69 | 0.868 |
| `ndebele` | isiNdebele | S44 | SADiLaR enriched corpora 1.0.0 | CC BY 4.0 | 34,604 | 12,620 | 2,423 | -5.07 | 0.893 |
| `swati` | siSwati | S43 | SADiLaR enriched corpora 1.0.0 | CC BY 4.0 | 32,610 | 12,232 | 2,188 | -5.12 | 0.893 |
| `shona` | ciShona | S10 | Leipzig Corpora Collection, cleaned by hand | CC BY 4.0 | 193,528 | 193,528 | 3,414 | -4.49 | 0.854 |
| `swahili` | Kiswahili | G42 | Wikipedia 20231101 and OPUS MultiParaCrawl v7.1 | CC BY-SA 3.0 and GFDL; MultiParaCrawl CC0 (packaging) | 2,419,601 | 158,083 | 15,151 | -5.06 | 0.781 |
| `lozi` | Silozi | K21 | Liseli | per source | 506,293 | 15,327 | 2,878 | -5.73 | 0.733 |
| `kinyarwanda` | Kinyarwanda | JD61 | Wikipedia 20231101 and Leipzig community 2017 | CC BY-SA 3.0 and GFDL; Leipzig CC BY 4.0 | 1,564,642 | 164,525 | 12,598 | -5.26 | 0.801 |
| `kirundi` | Kirundi | JD62 | Wikipedia 20231101 | CC BY-SA 3.0 and GFDL | 43,111 | 12,981 | 4,513 | -7.30 | 0.806 |
| `luganda` | Luganda | JE15 | Wikipedia 20231101 | CC BY-SA 3.0 and GFDL | 528,916 | 82,226 | 7,450 | -5.51 | 0.791 |
| `kikuyu` | Gikuyu | E51 | Wikipedia 20231101 and Leipzig community 2017 | CC BY-SA 3.0 and GFDL; Leipzig CC BY 4.0 | 77,077 | 20,205 | 6,442 | -7.38 | 0.789 |
| `kamba` | Kamba | E55 | OPUS NLLB v1 | ODC-By 1.0 | 624,679 | 89,823 | 13,970 | -6.53 | 0.767 |
| `chichewa` | Chichewa | N31 | Wikipedia 20231101 | CC BY-SA 3.0 and GFDL | 149,409 | 26,595 | 6,569 | -5.85 | 0.851 |
| `tumbuka` | Tumbuka | N21 | Wikipedia 20231101 and Glot500 | CC BY-SA 3.0 and GFDL; Glot500 per source (Glot500 packaging is CC0 1.0) | 1,287,757 | 74,167 | 12,580 | -5.25 | 0.817 |
| `bemba` | Bemba | M42 | OPUS NLLB v1 | ODC-By 1.0 | 2,270,800 | 260,609 | 13,401 | -5.39 | 0.777 |
| `luba` | Luba | L31 | Glot500 and OPUS NLLB v1 | Glot500 per source (Glot500 packaging is CC0 1.0); NLLB ODC-By 1.0 | 2,714,348 | 338,229 | 22,378 | -5.30 | 0.764 |
| `umbundu` | Umbundu | R11 | Glot500 and OPUS NLLB v1 | Glot500 per source (Glot500 packaging is CC0 1.0); NLLB ODC-By 1.0 | 2,274,564 | 238,814 | 18,977 | -5.74 | 0.759 |
| `yao` | Yao | P21 | Glot500 | per source (Glot500 packaging is CC0 1.0) | 78,192 | 13,292 | 1,675 | -5.07 | 0.876 |
| `lumbu` | Lumbu | B44 | Leipzig community 2017 | CC BY 4.0 | 80,238 | 15,884 | 2,362 | -4.94 | 0.856 |

Tokens and word types are counted in the training split. The exact source, version and download date of every corpus are in the main README. Liseli and Glot500 collect text from several sources, and each source keeps its own licence. The
ParaCrawl and NLLB licences cover the packaging of text mined from the web, whose original
sources keep their own terms.

### Training procedure

`setup.py` cleans the raw corpus, splits its word tokens 80/10/10 with seed 42, counts the
training split, and fits the threshold on the validation split, where every word is scored once as
it is and once with one synthetic error. No smoothing is applied. A trigram transition that was
never seen in training contributes a fixed penalty of -20 to the score of a word. A word is
flagged when its score is at or below the boundary tau of its language.

## Evaluation

### Testing data, factors and metrics

Each model is tested on 4,000 tokens sampled from the test split of its own corpus. Each token is
checked once as it is and once with one synthetic error, and the result is the mean of 30 seeds.
Words of fewer than three characters are included. Macro-F1 is the mean of the F1 on correct
words and the F1 on incorrect words.

### Results

The macro-F1 of each model is in the last column of the table above.

## Technical specifications

Each `.pkl` file is a Python pickle of a dict. The trigram and word models hold `n`, `ngrams`,
`counts`, `vocab`, `total_tokens`, `language`, `trained_on` and `tokenization`. The classifier
holds `clf`, `language`, `fallback_threshold` and `trained`. The models were trained with
Python 3.12 and saved with scikit-learn 1.8.0. Another version of scikit-learn loads the
classifiers but prints a warning.
