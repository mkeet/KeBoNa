"""
exp_rq3_external_eval.py - runs the spell checker on the fixed 500-word set, so that its
results can be set beside those of KenLM and of the SADiLaR spelling checker on the same words.
The words and their corrupted forms come from scripts/make_rq3_eval_set.py.

Outputs (results/rq3/):
    rq3_external_spellchecker_summary.csv  tp, fp, fn, tn and macro-F1 per corruption type
    rq3_external_spellchecker_words.csv    every word, clean and corrupted, with its trigrams,
        their log-probabilities, its score, the boundary tau, the flag and the outcome
"""

import argparse
from contextlib import nullcontext
import csv
import logging
from pathlib import Path
import sys

import mlflow

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from exp_common import (confusion_to_mf1, write_csv, MIN_WORD_LEN,
                        lr_boundary, trigram_breakdown, trigram_calc)
from src.core.spell_checker import SpellChecker
from analysis.kenlm_detect import run as run_kenlm
from analysis.plot_rq3_comparison import run as run_plot

logging.basicConfig(level=logging.WARNING)
EVAL_DIR = ROOT / "results" / "rq3"
TYPES = ["random", "qwerty"]


def load_groundtruth():
    """Reads the 500-word evaluation set."""
    p = EVAL_DIR / "rq3_500_groundtruth.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run scripts/make_rq3_eval_set.py first")
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score(det, clf, gt, ctype):
    """
    Scores the checker on the fixed set for one corruption type. Every word gives a clean row,
    which is a false positive or a true negative, and a corrupted row, which is a true positive
    or a false negative. Returns the counts (tp, fp, fn, tn) and the rows.
    """
    b = lr_boundary(clf); tau = "" if b is None else round(b["tau"], 6)

    def flag(w):
        if len(w) < MIN_WORD_LEN:
            return False, None
        s = det.score_word(w)
        return clf.predict_error(s), s

    def tg(w):
        tris, lps, _ = trigram_breakdown(det, w)
        return "|".join(tris), "|".join(f"{x:.3f}" for x in lps)

    tp = fp = fn = tn = 0
    rows = []
    for r in gt:
        # clean word
        w = r["original"]
        fl, s = flag(w)
        oc = "fp" if fl else "tn"
        fp += oc == "fp"; tn += oc == "tn"
        t, l = tg(w)
        rows.append({"corruption_type": ctype, "instance": "clean", "original": w,
                    "corrupted": "", "is_error": 0, "trigrams": t, "trigram_logprobs": l,
                    "trigram_calc": trigram_calc(det, w),
                    "detection_score": ("" if s is None else round(s, 4)), "tau": tau,
                    "flagged": int(fl), "outcome": oc})
        # corrupted form
        c = r[f"{ctype}_rate1.0_corrupted"]; label = int(r[f"{ctype}_rate1.0_is_error"])
        fl, s = flag(c)
        oc = "tp" if (label and fl) else "fp" if fl else "fn" if label else "tn"
        tp += oc == "tp"; fp += oc == "fp"; fn += oc == "fn"; tn += oc == "tn"
        t, l = tg(c)
        rows.append({"corruption_type": ctype, "instance": "corrupted", "original": w,
                    "corrupted": c, "is_error": label, "trigrams": t, "trigram_logprobs": l,
                    "trigram_calc": trigram_calc(det, c),
                    "detection_score": ("" if s is None else round(s, 4)), "tau": tau,
                    "flagged": int(fl), "outcome": oc})
    return (tp, fp, fn, tn), rows


def run(lang):
    """Evaluates the models of one language on the fixed set and writes the two CSVs."""
    checker = SpellChecker()
    checker.load_models(ROOT / "data" / "models")
    det = checker.detection_models[lang]
    clf = checker.threshold_classifiers[lang]
    gt = load_groundtruth()

    active = mlflow.active_run() is not None
    if not active:
        mlflow.set_experiment("RQ3_Baseline_Comparison")
    with (nullcontext() if active else mlflow.start_run()):
        summary_rows, all_word_rows = [], []
        print(f"\n{'='*96}\n  RQ3 external comparison - the spell checker on the fixed 500-word "
                f"set ({lang})\n{'='*96}")
        print(f"  {'config':<22}{'tp':>5}{'fp':>5}{'fn':>5}{'tn':>5}"
                f"{'incP':>8}{'incR':>8}{'incF1':>8}{'corF1':>8}{'mF1':>8}")
        print(f"  {'-'*92}")

        for ctype in TYPES:
            (tp, fp, fn, tn), wrows = score(det, clf, gt, ctype)
            mf1, p_e, r_e, f1_e, f1_c = confusion_to_mf1(tp, fp, fn, tn)
            summary_rows.append({
                "system": "spellchecker", "corruption_type": ctype, "rate": 1.0,
                "n_eval_words": len(wrows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "incorrect_precision": round(p_e, 6), "incorrect_recall": round(r_e, 6),
                "incorrect_f1": round(f1_e, 6), "correct_f1": round(f1_c, 6), "mf1": round(mf1, 6)})
            all_word_rows.extend(wrows)
            mlflow.log_metrics({f"spellchecker_{lang}_mf1_{ctype}": mf1,
                                f"spellchecker_{lang}_fp_{ctype}": fp})
            print(f"  {ctype+'_rate1.0':<22}{tp:>5}{fp:>5}{fn:>5}{tn:>5}"
                    f"{p_e:>8.3f}{r_e:>8.3f}{f1_e:>8.3f}{f1_c:>8.3f}{mf1:>8.3f}")
        print(f"{'='*96}\n")

        sp = write_csv(EVAL_DIR / f"rq3_external_spellchecker_summary_{lang}.csv", summary_rows)
        wp = write_csv(EVAL_DIR / f"rq3_external_spellchecker_words_{lang}.csv", all_word_rows)
        for p in (sp, wp):
            if p:
                mlflow.log_artifact(p)
        print(f"  Wrote rq3_external_spellchecker_summary_{lang}.csv and _words_{lang}.csv to {EVAL_DIR}")
        return summary_rows


def run_suite():
    """
    Runs the evaluation for both isiZulu corpora, then the KenLM detector and the comparison
    charts.
    """
    mlflow.set_experiment("RQ3_Baseline_Comparison")
    with mlflow.start_run():
        run("zulu")
        run("zu")
        try:
            run_kenlm("zulu", 5)
        except Exception as e:
            print("KenLM skipped:", e)
        run_plot()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zulu")
    a = ap.parse_args()
    run(a.lang)
