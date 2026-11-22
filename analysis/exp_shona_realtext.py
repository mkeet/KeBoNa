"""
exp_shona_realtext.py - a demonstration on a real Shona text with real typing errors (Melokuhle).

Three things are shown. They are specific to Shona and are a demonstration, not a benchmark.
    (1) the proportions of the real error types, against the uniform proportions that the
        synthetic errors assume
    (2) whether the checker finds those errors in running text at their natural density
    (3) the two Shona models on disk compared on the same text, one trained on the web
        collection and one on the corpus cleaned by hand
"""

import argparse
import collections
import csv
import json
import logging
from pathlib import Path
import sys

import mlflow

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import confusion_to_mf1, write_csv, MIN_WORD_LEN
from src.core.spell_checker import SpellChecker
from src.utils.text_utils import TextUtils
from src.models.trigram_model import TrigramModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAT = ROOT / "shona_material"
OUT = ROOT / "results" / "shona_realtext"
MODELS = {"cleaned": ROOT / "data" / "models",
          "dirty":   ROOT / "data" / "models_shona_web"}
ERROR_TYPES = ["insertion", "deletion", "substitution", "transposition"]


def load_typos():
    """Reads the list of typos and their corrections."""
    rows = list(csv.DictReader(open(MAT / "typos.csv", encoding="utf-8")))
    dist = collections.Counter()
    for r in rows:
        for t in [x.strip().lower() for x in r["edit_types"].replace(";", ",").split(",") if x.strip()]:
            dist[t] += 1
    return rows, dist


def real_word_typos(rows, vocab):
    """
    Returns the typed forms that are themselves valid Shona words.

    kuri for kuti and yaiva for chaiva are slips of word choice and not misspellings. The form
    typed is spelled correctly, so no checker that scores words one at a time can flag it. These
    forms are left out of the scoring and counted separately.
    """
    return {r["typed"].strip().lower() for r in rows
            if r["typed"].strip() and r["typed"].strip().lower() in vocab}


def evaluate(models_dir, text, typed_forms, real_words, label):
    """
    Runs detection over the raw text, token by token. A word is a true error if the translator
    corrected it and the form typed is not itself a word of the language.
    """
    checker = SpellChecker()
    checker.load_models(models_dir)
    if "shona" not in checker.detection_models:
        logger.error("no shona model in %s", models_dir); return None
    res = checker.check(text, language="shona",
                        include_suggestions=False, min_word_length=MIN_WORD_LEN)
    flagged = {e["word"].lower() for e in res["errors"]}
    errors = typed_forms - real_words
    # every token is counted once per occurrence, in lower case because detection ignores case
    forms = [w for w in TextUtils.tokenize(TextUtils.normalize(text))
             if len(w) >= MIN_WORD_LEN and w.lower() not in real_words]
    tp = sum(1 for w in forms if w.lower() in errors and w.lower() in flagged)
    fn = sum(1 for w in forms if w.lower() in errors and w.lower() not in flagged)
    fp = sum(1 for w in forms if w.lower() not in errors and w.lower() in flagged)
    tn = sum(1 for w in forms if w.lower() not in errors and w.lower() not in flagged)
    mf1, p_e, r_e, f1_e, f1_c = confusion_to_mf1(tp, fp, fn, tn)
    n_tri = len(checker.detection_models["shona"].vocab)
    return {"corpus": label, "char_trigrams": n_tri,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "real_word_typos_excluded": len(real_words),
            "precision": p_e, "recall": r_e, "incorrect_f1": f1_e,
            "correct_f1": f1_c, "mf1": mf1}


def run():
    """Runs the three parts and writes the JSON and the CSVs to results/shona_realtext/."""
    raw = (MAT / "melokuhle_shona_raw-MK.txt").read_text(encoding="utf-8")
    rows, dist = load_typos()
    typed = {r["typed"].strip().lower() for r in rows if r["typed"].strip()}
    vocab = {w.lower() for w in TrigramModel.load(ROOT / "data" / "models" / "shona_word.pkl").vocab}
    real_words = real_word_typos(rows, vocab)
    n_words = len(raw.split())
    density = 100 * len(rows) / n_words

    total = sum(dist[t] for t in ERROR_TYPES)
    ratio = [{"error_type": t, "count": dist[t], "real_pct": 100*dist[t]/total,
              "wrecker_pct": 25.0} for t in ERROR_TYPES]
    multi = sum(1 for r in rows if r["n_edits"].strip() in ("2", "3"))

    OUT.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("Shona_RealText_Demonstration")
    with mlflow.start_run():
        mlflow.log_params({"experiment": "shona_realtext", "text": "Melokuhle (translated)",
                           "n_typos": len(rows), "n_words": n_words})
        results = [r for r in (evaluate(d, raw, typed, real_words, k) for k, d in MODELS.items()) if r]
        write_csv(OUT / "shona_realtext_detection.csv", results)
        write_csv(OUT / "shona_realtext_error_ratio.csv", ratio)
        (OUT / "shona_realtext.json").write_text(json.dumps(
            {"text": "Melokuhle (translated)", "n_words": n_words, "n_typos": len(rows),
             "error_density_pct": density, "multi_edit_typos": multi,
             "error_type_ratio": ratio, "detection": results}, indent=2))
        for r in results:
            mlflow.log_metrics({f"mf1_{r['corpus']}": r["mf1"],
                                f"recall_{r['corpus']}": r["recall"],
                                f"precision_{r['corpus']}": r["precision"]})
        for p in OUT.glob("shona_realtext*"):
            mlflow.log_artifact(str(p))

    print(f"\n{'='*70}\n  Shona demonstration - Melokuhle (translated), real typos\n{'='*70}")
    print(f"  {n_words:,} words, {len(rows)} typos = {density:.2f}% error density")
    print(f"  {multi} of {len(rows)} typos need more than one edit "
          f"({100*multi/len(rows):.1f}%); the generator only ever applies one\n")
    print(f"  (1) sampling ratio of error types")
    print(f"      {'type':<16}{'real':>8}{'wrecker':>10}")
    for r in ratio:
        print(f"      {r['error_type']:<16}{r['real_pct']:>7.1f}%{r['wrecker_pct']:>9.1f}%")
    print(f"\n  (2,3) detection on the real text, dirty vs cleaned corpus")
    print(f"      {'corpus':<10}{'trigrams':>10}{'P':>8}{'R':>8}{'mF1':>8}")
    for r in results:
        print(f"      {r['corpus']:<10}{r['char_trigrams']:>10,}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['mf1']:>8.3f}")
    print(f"{'='*70}\n  wrote results/shona_realtext/\n")
    return results


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
