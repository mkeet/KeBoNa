# ncb-trigram-spellchecker

Character-trigram spell checking for Niger-Congo B languages.

A platform-independent, language-agnostic spell checker for Niger-Congo B (NCB) languages.
Detection is a character-trigram language model plus a one-dimensional logistic-regression
threshold, both trained from raw text. No morphological rules, no hand-built lexicon, no
linguistic expertise required to add a language.

## Key features

- **Add a language** by dropping a plain-text corpus in a folder and running one command.
- **Error detection** from character-trigram conditional probabilities, thresholded per language.
- **Correction suggestions** ranked by edit distance, then corpus frequency, then trigram
  plausibility.

## Quick start

The trained models are included in `data/models/`, so the app runs without training anything.
On Debian or Ubuntu, install into a virtual environment (the system Python refuses `pip install`):

```bash
python3 -m venv venv
source venv/bin/activate

# to run the app only
pip install nicegui pywebview scikit-learn==1.8.0 rapidfuzz numpy langdetect
python app.py                   # desktop app (native window; falls back to http://localhost:8080)
```

The models were saved with scikit-learn 1.8.0; another version loads them but prints an
`InconsistentVersionWarning`. To retrain from your own corpora, or to run the experiments, install
the full requirements instead (`kenlm` in that list needs a C++ toolchain to build):

```bash
pip install -r requirements.txt
python setup.py                 # clean + train every language in data/corpora/
python setup.py --lang zulu     # or just one
```

## Pipeline

What runs, in what order, and what calls what.

```
data/corpora/<lang>/raw/*.txt          your raw text, any number of files
        |
        |  setup.py  (the only thing you run to onboard a language)
        v
  1  CorpusCleaner        strips markup, non-Latin script, digits, punctuation except
     (src/utils)          hyphens; drops English/French/Portuguese sentences via langdetect
        v
        cleaned.txt
        v
  2  DataSplitter         80 / 10 / 10 by seeded shuffle of word tokens, seed 42
     (src/utils)          -> train.txt  test.txt  validation.txt
        v
  3  TrigramModel         whole-word counts -> <lang>_word.pkl
     (src/models)         used for the vocabulary check and suggestion ranking
        v
  4  char-trigram corpus  each training word rewritten as its overlapping 3-char windows
        v                 -> train_trigram.txt  (built from train.txt only)
  5  TrigramModel         trigram counts -> <lang>_trigram.pkl
                          used for detection scoring
        v
  6  ThresholdClassifier  1-D logistic regression on validation words vs corrupted copies
     (src/models)         -> <lang>_classifier.pkl   (the boundary tau)
```

At inference `SpellChecker` (`src/core`) loads all three pickles per language and scores a word:

```
word -> CharTrigramTokenizer -> trigrams
     -> TrigramModel.score_word()   mean log P(trigram | two preceding), -20 floor
     -> ThresholdClassifier.predict_error()   flags when coef*score + intercept <= 0
     -> TrigramModel.suggest_corrections()    only if flagged
```

**Where experiments come in.** They import the *trained pickles* and never retrain, except RQ1
and C10 which deliberately train fresh models at varying corpus sizes. Every experiment lives in
`analysis/`, shares `analysis/exp_common.py` (the evaluation engine, `evaluate_balanced_checker`),
and is driven either individually or by the `experiments.py` marimo notebook.

## Running experiments

Run from the project root so relative paths resolve.

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

venv/bin/python experiments.py           # every experiment, in order (script mode)
venv/bin/marimo edit experiments.py      # or interactively, cell by cell
```

Script mode is preferable for long runs: it survives a closed terminal and, unlike the marimo
GUI, never applies a dark matplotlib theme to the figures.

A single experiment can be run directly:

```bash
venv/bin/python analysis/exp_rq1_corpus_size.py
venv/bin/python analysis/exp_c6_error_types.py --lang zu --corruption-types random
```

`analysis/exp_c9_llm_comparison.py` additionally needs `GROQ_API_KEY`, `OPENROUTER_API_KEY` and
`GEMINI_API_KEY` in `.env`, and [Ollama](https://ollama.com) running at
`http://localhost:11434` (or `OLLAMA_URL`) with the three local models pulled: `gemma4:12b`, `gpt-oss:20b` and `hf.co/mradermacher/AfriqueQwen3.5-9B-50Langs-Instruct-v1-i1-GGUF:Q4_K_M`. `analysis/kenlm_detect.py` needs the KenLM binaries; set `KENLM_BIN` if they are not at `../kenlm/build/bin`.

### Experiments, scripts and outputs

Each experiment, the script that produces it and where it writes. E-numbers are the experiments
and A-numbers the supporting analyses.

| | Experiment | Script | Output |
|---|---|---|---|
| E1 | Detection, 8 configurations | `analysis/evaluate_detection.py` | `results/eval_detection/` |
| E2 | Corpus size (RQ1) | `analysis/exp_rq1_corpus_size.py` | `results/rq1/` |
| E3 | Transfer, 420 pairs (RQ2) | `analysis/exp_rq2_cross_language.py` | `results/rq2/` |
| E3 | Transfer on UDHR parallel text | `analysis/exp_udhr_transfer.py` | `results/udhr_transfer/` |
| E3 | Sensitivity of the distance weights | `analysis/rq2_weight_sensitivity.py` | `results/rq2/` |
| E3 | Transfer failures word by word | `analysis/qual_rq2_transfer_failures.py`, `analysis/qual_missing_trigrams.py --pair` | `results/rq2/` |
| E4 | Comparison with KenLM and SADiLaR (RQ3) | `scripts/make_rq3_eval_set.py`, `analysis/exp_rq3_external_eval.py`, `analysis/kenlm_detect.py`, `analysis/score_sadilar_run.py`, `analysis/plot_rq3_comparison.py` | `results/rq3/` |
| E5 | Correction | `analysis/exp_rq4_correction.py`, `analysis/qual_rq4_corrections.py`, `analysis/qual_rq4_error_types.py`, `analysis/qual_rq4_length.py` | `results/rq4/` |
| A1 | Scoring formulations | `scripts/compare_scoring_methods.py` | `results/scoring_methods/` |
| A2 | Error types | `analysis/exp_c6_error_types.py` | `results/c6/` |
| A3 | Failure analysis | `analysis/exp_c8_fp_fn_analysis.py`, `analysis/qual_missing_trigrams.py` | `results/c8/` |
| A4 | Orthography (UDHR) | `analysis/exp_c7_orthography_udhr.py` | `results/c7/` |
| A5 | Large language models | `analysis/exp_c9_llm_comparison.py`, `analysis/plot_llm_comparison.py` | `results/c9/` |
| A6 | Cost | `analysis/exp_c5_timing.py` | `results/c5/` |
| A7 | Error density and training size | `analysis/exp_c10_scaling.py` | `results/c10/` |
| A8 | Short words | `scripts/compare_min_word_len.py` | `results/min_word_len/` |
| A9 | Real Shona text | `analysis/exp_shona_realtext.py`, `analysis/exp_shona_traintext.py` | `results/shona_realtext/` |

The SADiLaR spelling checker (E4) is a Microsoft Word extension and was run once by hand. Its
output on the 500 words is in `results/rq3/{original,random,qwerty}_sadilar.txt`, one line per word aligned to `results/rq3/rq3_500_groundtruth.csv`, and `analysis/score_sadilar_run.py` scores it. A5 is a single run of the language models; their answers are cached in `results/c9/cache/`, and a new run may give different answers. Claude Opus 5.5 and GPT-6.2 Astra were run separately on the same 500 words with the same prompt; their replies, the prompts they were given and the word list are in `results/c9/paid_models/`.
`analysis/plot_llm_comparison.py` checks that every run used the same words and corruptions, scores all systems and draws the comparison chart.

### MLflow UI

```bash
venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db \
                   --default-artifact-root ./mlartifacts --port 5000
```

The UI's `--backend-store-uri` must match the runs' `MLFLOW_TRACKING_URI`, or it will show an
empty store. Artifacts land in `mlartifacts/`; the same files are also written to `results/`.

## Adding a new language

```bash
mkdir -p data/corpora/<language>/raw
cp mycorpus.txt data/corpora/<language>/raw/
venv/bin/python setup.py --lang <language>
```

That is the whole procedure. `SpellChecker.load_models()` discovers languages by globbing
`data/models/*_word.pkl`, so the new language appears in the app and is available to every
experiment as soon as it is trained. No configuration file needs editing.

To place the language on the Guthrie distance metric used by RQ2, add its code to
`GUTHRIE_CODES` in `src/utils/guthrie_distance.py` (for example `"bemba": "M42"`). Without a code
the language still trains and runs, but RQ2 skips it.

Helper scripts for sourcing corpora for under-resourced languages:

| Script | Source |
|---|---|
| `scripts/download_hf_corpus.py` | any HuggingFace dataset, streamed |
| `scripts/_zone_wiki_dl.py` | Wikipedia via HuggingFace `wikimedia/wikipedia`, snapshot 20231101 |
| `scripts/_zone_glot_dl.py` | Glot500 |
| `scripts/_opus_mono_dl.py` | OPUS monolingual object store (NLLB, CCMatrix, …) |

## Project structure

```
ncb-trigram-spellchecker/
├── app.py                      NiceGUI desktop app
├── setup.py                    corpus cleaning + training pipeline (one command per language)
├── experiments.py              marimo notebook driving every experiment
├── build.spec                  PyInstaller recipe for a standalone app
├── src/
│   ├── core/spell_checker.py       loads models, routes check() calls
│   ├── models/
│   │   ├── trigram_model.py        counting, score_word, suggest_corrections
│   │   └── threshold_classifier.py 1-D logistic regression boundary
│   ├── utils/
│   │   ├── corpus_cleaner.py       raw text -> cleaned word list
│   │   ├── data_splitter.py        80/10/10 split of word tokens
│   │   ├── trigram_tokenizer.py    3-char sliding window
│   │   ├── text_utils.py           normalise, tokenize, metadata-line check
│   │   └── guthrie_distance.py     Guthrie/Maho codes + zone-adjacency distance
│   └── evaluation/
│       └── spelling_wrecker.py     synthetic Damerau error generator
├── analysis/                   the experiments (E1–E5, A1–A9) + exp_common.py
├── scripts/                    corpus downloaders, the RQ3 evaluation set, A1 and A8
├── data/
│   ├── corpora/<lang>/raw/         raw text in, cleaned + split files out
│   ├── models/                     <lang>_{word,trigram,classifier}.pkl
│   ├── models_shona_web/           Shona models trained on the earlier web collection (A9 only)
│   └── udhr/ udhr_xml/             UDHR parallel text (E3 parallel-text step, A4)
├── shona_material/             Shona translated text and typo list (A9)
├── results/                    experiment outputs (JSON, CSV, PDF, PNG)
└── tests/                      pytest suite
```

Running the experiments also creates `mlflow.db` and `mlartifacts/` (the MLflow run archive).

## Data

Corpora are **not** included in this repository. The table gives, for each of the 25 corpora
(21 languages) used, the exact source, the version and the date it was saved on
this machine. Each file goes in `data/corpora/<folder>/raw/`.

| Folder | Language | Source and link | Version | Saved |
|---|---|---|---|---|
| `zu` | isiZulu | NCHLT isiZulu Text Corpora, [hdl.handle.net/20.500.12185/321](https://hdl.handle.net/20.500.12185/321) | 2.0 (`CORP.NCHLT.zu.*.2.0.txt`) | 15 Jun 2026 |
| `xh` | isiXhosa | NCHLT isiXhosa Text Corpora, [hdl.handle.net/20.500.12185/314](https://hdl.handle.net/20.500.12185/314) | 2.0 | 15 Jun 2026 |
| `nd` | isiNdebele | NCHLT isiNdebele Text Corpora, [hdl.handle.net/20.500.12185/308](https://hdl.handle.net/20.500.12185/308) | 2.0 | 17 Jun 2026 |
| `sw` | siSwati | NCHLT Siswati Text Corpora, [hdl.handle.net/20.500.12185/348](https://hdl.handle.net/20.500.12185/348) | 2.0 | 15 Jun 2026 |
| `sotho` | Sesotho | NCHLT Sesotho Text Corpora, [hdl.handle.net/20.500.12185/336](https://hdl.handle.net/20.500.12185/336) | 2.0 | 1 Jul 2026 |
| `tswana` | Setswana | NCHLT Setswana Text Corpora, [hdl.handle.net/20.500.12185/343](https://hdl.handle.net/20.500.12185/343) | 2.0 | 1 Jul 2026 |
| `zulu` | isiZulu | Linguistically enriched corpora for conjunctively written South African languages, [hdl.handle.net/20.500.12185/546](https://hdl.handle.net/20.500.12185/546), plus African Speech Technology isiZulu Text Corpus, [hdl.handle.net/20.500.12185/338](https://hdl.handle.net/20.500.12185/338) (folder `ast/`) | `SADII.ZU.Morph_Lemma_POS.1.0.0.{TRAIN,TEST}.CTexT.TG.2021-09-30` | 1 Mar 2026 |
| `xhosa` | isiXhosa | Linguistically enriched corpora, handle 546 as above | `SADII.XH...1.0.0...2021-09-30` | 1 Mar 2026 |
| `ndebele` | isiNdebele | Linguistically enriched corpora, handle 546 as above | `SADII.NR...1.0.0...2021-09-30` | 1 Mar 2026 |
| `swati` | siSwati | Linguistically enriched corpora, handle 546 as above | `SADII.SS...1.0.0...2021-09-30` | 1 Mar 2026 |
| `shona` | ciShona | Leipzig Corpora Collection, [wortschatz.uni-leipzig.de/en/download](https://wortschatz.uni-leipzig.de/en/download), cleaned by hand for this project into `shona_clean.txt` | not recorded | 12 Aug 2026 (cleaned file) |
| `swahili` | Kiswahili | Wikipedia, [huggingface.co/datasets/wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) config `20231101.sw` (`swahili.txt`), plus OPUS MultiParaCrawl, [object.pouta.csc.fi/OPUS-MultiParaCrawl/v7.1/moses/en-sw.txt.zip](https://object.pouta.csc.fi/OPUS-MultiParaCrawl/v7.1/moses/en-sw.txt.zip), file `MultiParaCrawl.en-sw.sw` | Wikipedia snapshot 20231101; MultiParaCrawl v7.1 | 1 Jul 2026 (Wikipedia); MultiParaCrawl date not recorded |
| `lozi` | Silozi | Liseli, [huggingface.co/datasets/GiJoeHansFranz/Liseli](https://huggingface.co/datasets/GiJoeHansFranz/Liseli), file `parallel-corpus/lozi.parquet`, column `translation`, duplicate lines removed (35,711 rows to 33,693 lines) | revision `512569370059a7a8623b583a4fe80482238af98b` (10 Apr 2026) | 1 Jul 2026 |
| `kinyarwanda` | Kinyarwanda | Wikipedia config `20231101.rw`, plus Leipzig [downloads.wortschatz-leipzig.de/corpora/kin_community_2017.tar.gz](https://downloads.wortschatz-leipzig.de/corpora/kin_community_2017.tar.gz) | 20231101; community_2017 | 14–15 Jul 2026 |
| `kirundi` | Kirundi | Wikipedia config `20231101.rn` | 20231101 | 14 Jul 2026 |
| `luganda` | Luganda | Wikipedia config `20231101.lg` | 20231101 | 14 Jul 2026 |
| `kikuyu` | Gikuyu | Wikipedia config `20231101.ki`, plus Leipzig [downloads.wortschatz-leipzig.de/corpora/kik_community_2017.tar.gz](https://downloads.wortschatz-leipzig.de/corpora/kik_community_2017.tar.gz) | 20231101; community_2017 | 14–15 Jul 2026 |
| `chichewa` | Chichewa | Wikipedia config `20231101.ny` | 20231101 | 14 Jul 2026 |
| `tumbuka` | Tumbuka | Wikipedia config `20231101.tum`, plus Glot500, [huggingface.co/datasets/cis-lmu/Glot500](https://huggingface.co/datasets/cis-lmu/Glot500) config `tum_Latn` | 20231101; Glot500 revision `f7b3e9d11a7e974ec0d021342cd8b41fd67510c1` | 14–15 Jul 2026 |
| `kamba` | Kamba | OPUS-NLLB monolingual, [object.pouta.csc.fi/OPUS-NLLB/v1/mono/kam.txt.gz](https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/kam.txt.gz) | NLLB v1 | 14 Jul 2026 |
| `bemba` | Bemba | OPUS-NLLB parallel, [object.pouta.csc.fi/OPUS-NLLB/v1/moses/bem-en.txt.zip](https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/bem-en.txt.zip), Bemba side `NLLB.bem-en.bem` (saved as `bemba_nllb.txt`) | NLLB v1 | not recorded (before 14 Jul 2026) |
| `luba` | Luba | Glot500 config `lua_Latn`, plus OPUS-NLLB [object.pouta.csc.fi/OPUS-NLLB/v1/mono/lua.txt.gz](https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/lua.txt.gz) | Glot500 revision as above; NLLB v1 | 14 Jul 2026 |
| `umbundu` | Umbundu | Glot500 config `umb_Latn`, plus OPUS-NLLB [object.pouta.csc.fi/OPUS-NLLB/v1/mono/umb.txt.gz](https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/umb.txt.gz) | Glot500 revision as above; NLLB v1 | 14 Jul 2026 |
| `yao` | Yao | Glot500 config `yao_Latn` | Glot500 revision as above | 14 Jul 2026 |
| `lumbu` | Lumbu | Leipzig [downloads.wortschatz-leipzig.de/corpora/lup_community_2017.tar.gz](https://downloads.wortschatz-leipzig.de/corpora/lup_community_2017.tar.gz) | community_2017 | 15 Jul 2026 |

"Saved" is the date of the file on this machine. The NLLB, MultiParaCrawl, Leipzig and Liseli
files above were downloaded again on 6 Nov 2026 and are identical to the files used. Wikipedia
and Glot500 are given at the dataset revision current at download time (Wikipedia last changed
9 Jan 2024, Glot500 10 Dec 2025). The scripts that fetched them are `scripts/_zone_wiki_dl.py`,
`scripts/_zone_glot_dl.py`, `scripts/_opus_mono_dl.py` and `scripts/download_hf_corpus.py`.

`setup.py` reads every `.txt` file under `raw/`. For the NCHLT corpora this includes the
source documents (`1.Source/`), both the CLEAN and RAW corpus files (`2.Corpora/`) and the
lexica (`3.Lexica/`); for Leipzig it includes both the `-sentences.txt` and the `-words.txt`
files.

### Other data

| Data | Source | Version | Saved |
|---|---|---|---|
| UDHR translations (E3 parallel-text step, A4) | [github.com/eric-muller/udhr](https://github.com/eric-muller/udhr), `data/udhr/udhr_<code>.xml`, cached in `data/udhr_xml/` | `main` branch | 26 Jun and 3 Jul 2026 |
| Shona real text and typo list (A9) | `shona_material/` in this repository | | 2 Sep 2026 |
| Shona models trained on the web collection (A9) | `data/models_shona_web/` in this repository | | |
| SADiLaR spelling checker (E4) | [sadilar.org/en/spelling-checkers](https://sadilar.org/en/spelling-checkers/), run by hand; output in `results/rq3/*_sadilar.txt` | 18.00.0004 | 4 Oct 2026 (last accessed) |

### Results

`results/` holds the output of every experiment above, under `type-based/` and `token-based/`
where an experiment counts both ways. Some files cannot be regenerated: the 500-word RQ3
evaluation set and the manual SADiLaR run (`results/rq3/`), the cached language-model answers
(`results/c9/cache/`), the per-word results of the language-model run
(`results/c9/c9_reasoned_words_zu_*.csv`) and the replies of the two paid models
(`results/c9/paid_models/`). Running an experiment again overwrites its files.
`data/models/` holds the trained models, three per corpus
(`<folder>_{word,trigram,classifier}.pkl`). The model card in `data/models/README.md` describes
them.

## Tests

```bash
venv/bin/python -m pytest tests/ -q
```
