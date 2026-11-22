"""
exp_c10_scaling.py - detection as the error density and the training size grow.

    (A) error density  the share of corrupted words in the test text is raised from 5% to
                       100% and macro-F1 is tracked. Real text carries about 2 to 5% errors.
    (B) data size      macro-F1 against the size of the training corpus, at fewer points
                       than exp_rq1_corpus_size.py

Outputs (results/c10/<mode>-based/): c10_scaling.json, c10_error_rate.pdf, c10_data_size.pdf
and the CSVs behind them
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import (ROOT, load_words, corpus_path, sample_unique, sample_tokens,
                        train_detection_model, train_lr, evaluate_balanced_model,
                        evaluate_at_rate_checker, mean_std, write_csv, MIN_WORD_LEN)
from src.evaluation.spelling_wrecker import SpellingWrecker
from src.core.spell_checker import SpellChecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RATES = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
DATA_SIZES = [5_000, 10_000, 20_000, 40_000]


def error_rate_sweep(lang, seeds, test_size, sample_mode="type", min_word_length=3):
    """Evaluates the checker on running text at each error density, from 5% to 100% of the words."""
    checker = SpellChecker()
    checker.load_models(ROOT / "data" / "models")
    _sample = sample_tokens if sample_mode == "token" else sample_unique
    test_words = _sample(load_words(corpus_path(lang, "test")), test_size)

    series = []
    for rate in RATES:
        runs = [evaluate_at_rate_checker(checker, lang, test_words, rate, s,
                                        min_word_length=min_word_length)
                for s in range(seeds)]
        m, sd = mean_std([r["mf1"] for r in runs])
        def mn(k): return float(np.mean([r[k] for r in runs]))
        series.append({"rate": rate, "mean": m, "std": sd,
                    "tp": mn("tp"), "fp": mn("fp"), "fn": mn("fn"), "tn": mn("tn"),
                    "incorrect_precision": mn("precision"), "incorrect_recall": mn("recall"),
                    "incorrect_f1": mn("f1_error"), "correct_f1": mn("f1_correct")})
        logger.info("[%s] rate=%.2f mF1=%.4f +/- %.4f (tp=%.0f fp=%.0f fn=%.0f tn=%.0f)",
                    lang, rate, m, sd, mn("tp"), mn("fp"), mn("fn"), mn("tn"))
    return series, checker, test_words


def rate_word_sample(checker, lang, test_words, rate=0.5, n_sample=200, show=6, seed=0,
                     min_word_length=MIN_WORD_LEN):
    """
    Corrupts a fraction of a small sample at one error density and runs detection on it.
    Prints the first rows and returns the outcome of every word.
    """
    sample = test_words[:n_sample]
    wr = SpellingWrecker(corruption_rate=rate, seed=seed)
    text, gt = wr.corrupt_text(" ".join(sample), min_word_length=min_word_length)
    corrupted_set = set(gt.keys())
    res = checker.check(text, language=lang,
                        include_suggestions=False, min_word_length=min_word_length)
    scored = {e["word"]: e.get("probability") for e in res["errors"]}
    rows = []
    for tok in text.split():
        is_err = tok in corrupted_set
        fl = tok in scored
        rows.append({"rate": rate, "token": tok, "original_if_corrupted": gt.get(tok, ""),
                    "is_error": int(is_err), "flagged": int(fl),
                    "detection_score": ("" if not fl else round(scored[tok], 4)),
                    "outcome": ("tp" if is_err and fl else "fp" if fl
                                else "fn" if is_err else "tn")})

    print(f"\n  Error-density word sample at rate={rate} (first {show} of {len(rows)} tokens):")
    for r in rows[:show]:
        print(f"    {r['token']:<18} is_error={r['is_error']} flagged={r['flagged']} "
            f"score={r['detection_score']} outcome={r['outcome']}")

    return rows


def data_size_sweep(lang, seeds, test_size, sample_mode="type", min_word_length=3):
    """Trains a fresh detection model and threshold at each corpus size and evaluates them."""

    all_train = load_words(corpus_path(lang, "train"))
    val_words = load_words(corpus_path(lang, "validation"))
    _sample = sample_tokens if sample_mode == "token" else sample_unique
    test_words = _sample(load_words(corpus_path(lang, "test")), test_size)

    series = []
    for size in DATA_SIZES:
        if size > len(all_train):
            break
        sample = sample_tokens(all_train, size, seed=42); actual = len(set(sample))
        model = train_detection_model(sample, lang)
        clf = train_lr(model, val_words)
        if clf is None:
            continue
        runs = [evaluate_balanced_model(model, clf, test_words, s,
                                        min_word_length=min_word_length) for s in range(seeds)]
        m, sd = mean_std([r["mf1"] for r in runs])
        def mn(k): return float(np.mean([r[k] for r in runs]))
        series.append({"unique_words": actual, "train_tokens": len(sample),
                    "mean": m, "std": sd,
                    "tp": mn("tp"), "fp": mn("fp"), "fn": mn("fn"), "tn": mn("tn"),
                    "incorrect_f1": mn("f1_error"), "correct_f1": mn("f1_correct")})
        logger.info("[%s] train=%d unique (%d tokens) mF1=%.4f +/- %.4f",
                    lang, actual, len(sample), m, sd)
    return series


def _run_mode(lang, seeds, test_size, out, sample_mode="type", min_word_length=3):

    err, checker, test_words = error_rate_sweep(lang, seeds, test_size,
                                                sample_mode, min_word_length)
    # a word-by-word sample at every rate
    sample_rows = [r for rate in RATES
                   for r in rate_word_sample(checker, lang, test_words, rate=rate, show=3,
                                             min_word_length=min_word_length)]
    dat = data_size_sweep(lang, seeds, test_size, sample_mode, min_word_length)

    for r in err:
        mlflow.log_metrics({
            f"errsweep_rate_{sample_mode}": r["rate"],
            f"errsweep_mF1_{sample_mode}": r["mean"],
            f"errsweep_mF1_std_{sample_mode}": r["std"],
        }, step=int(r["rate"] * 100))

    for d in dat:
        mlflow.log_metrics({
            f"datasweep_train_size_{sample_mode}": d["train_tokens"],
            f"datasweep_mF1_{sample_mode}": d["mean"],
            f"datasweep_mF1_std_{sample_mode}": d["std"],
        }, step=d["train_tokens"])

    out.mkdir(parents=True, exist_ok=True)
    (out / "c10_scaling.json").write_text(json.dumps(
        {"language": lang, "error_rate_sweep": err, "data_size_sweep": dat}, indent=2))
    mlflow.log_artifact(str(out / "c10_scaling.json"), artifact_path=f"{sample_mode}-based")

    # one row per point of each sweep, and the word-by-word sample behind each error density
    for _name, _rows in [("c10_error_rate_sweep.csv", err),
                        ("c10_data_size_sweep.csv", dat),
                        (f"c10_words_by_rate_{lang}.csv", sample_rows)]:
        _p = write_csv(out / _name, _rows)
        if _p:
            mlflow.log_artifact(_p, artifact_path=f"{sample_mode}-based")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar([r["rate"] for r in err], [r["mean"] for r in err],
                yerr=[r["std"] for r in err], fmt="o-", capsize=4, color="#d62728")
    ax.set_xlabel("Evaluation corruption rate")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"C10A: mF1 vs error density ({lang})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out / "c10_error_rate.pdf", dpi=150)
    mlflow.log_artifact(str(out / "c10_error_rate.pdf"), artifact_path=f"{sample_mode}-based")

    if dat:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.errorbar([d["train_tokens"] for d in dat], [d["mean"] for d in dat],
                    yerr=[d["std"] for d in dat], fmt="s-", capsize=4, color="#1f77b4")
        ax2.set_xlabel("Training tokens")
        ax2.set_ylabel("Macro F1")
        ax2.set_title(f"C10B: mF1 vs training size ({lang})")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.0)
        fig2.tight_layout()
        fig2.savefig(out / "c10_data_size.pdf", dpi=150)
        mlflow.log_artifact(str(out / "c10_data_size.pdf"), artifact_path=f"{sample_mode}-based")

    logger.info("Saved results/c10_scaling.{json,pdf}")

    print(f"\n{'='*72}\n  C10 - scaling ({lang})\n{'='*72}")
    print("  (A) Error-density sweep = fraction of words corrupted in one pass.")
    print(f"  {'rate':>6}{'mF1':>9}{'+/-':>8}{'tp':>8}{'fp':>8}{'fn':>8}{'tn':>8}")
    for r in err:
        print(f"  {r['rate']:>6.2f}{r['mean']:>9.4f}{r['std']:>8.4f}"
            f"{r['tp']:>8.0f}{r['fp']:>8.0f}{r['fn']:>8.0f}{r['tn']:>8.0f}")
    print("  (rate 1.0: no correct words remain -> TN=0 -> mF1 drops by construction)")
    print("\n  (B) Data-size sweep = mF1 vs training tokens (fresh model each):")
    print(f"  {'unique':>8}{'tokens':>9}{'mF1':>9}{'+/-':>8}")
    for d in dat:
        print(f"  {d['unique_words']:>8}{d['train_tokens']:>9}{d['mean']:>9.4f}{d['std']:>8.4f}")
    print(f"{'='*72}\n")

    return {"error_rate_sweep": err, "data_size_sweep": dat}


def run(lang, seeds, test_size, modes=("type", "token")):
    """
    Runs both sweeps for one language, once per mode. Short words are scored (min_word_length=1).
    """
    base = ROOT / "results" / "c10"
    mlflow.set_experiment("scalability_analysis")
    with mlflow.start_run():
        mlflow.log_params({"experiment": "c10_scaling", "language": lang, "seeds": seeds,
                        "test_size": test_size, "modes": ",".join(modes)})
        for mode in modes:
            _run_mode(lang, seeds, test_size, base / f"{mode}-based",
                    sample_mode=mode, min_word_length=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zulu")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--test-size", type=int, default=4000)
    a = ap.parse_args()
    run(a.lang, a.seeds, a.test_size)
