"""
kenlm_detect.py - a KenLM detector, compared with the checker on the fixed 500-word set (RQ3).

The pipeline is the same as the checker's, with the trigram score replaced by a
character-level KenLM model with modified Kneser-Ney smoothing.

    1. a character-level KenLM model of order N is trained on the same train.txt, with each
       word written as its characters separated by spaces
    2. a word is scored by its KenLM log10 probability divided by its length
    3. a one-dimensional logistic regression is fitted on the validation words for the
       boundary, as ThresholdClassifier does
    4. the detector is evaluated on the fixed 500-word set, and the CSVs have the same format
       as those of exp_rq3_external_eval.py

The KenLM binaries (lmplz) and the kenlm Python module are required.
"""

import argparse
from contextlib import nullcontext
import csv
import logging
import os
from pathlib import Path
import subprocess
import sys

import kenlm
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from exp_common import (load_words, corpus_path, confusion_to_mf1, write_csv, MIN_WORD_LEN)
from src.evaluation.spelling_wrecker import SpellingWrecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KENLM_BIN = Path(os.environ.get("KENLM_BIN", ROOT.parent / "kenlm" / "build" / "bin"))
EVAL_DIR = ROOT / "results" / "rq3"
MODEL_DIR = EVAL_DIR / "kenlm_models"
TYPES = ["random", "qwerty"]
RATES = [1.0]


def build_char_lm(lang, order):
    """
    Trains a character-level KenLM model of the given order on train.txt and returns the path
    of its .arpa file. The model is trained once and reused on later runs.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    arpa = MODEL_DIR / f"{lang}_char_o{order}.arpa"

    if arpa.exists():
        logger.info("[%s] reusing cached char-LM %s", lang, arpa.name)
        return arpa
    words = load_words(corpus_path(lang, "train"))
    chars_file = MODEL_DIR / f"{lang}_chars.txt"
    chars_file.write_text("\n".join(" ".join(w) for w in words) + "\n", encoding="utf-8")
    logger.info("[%s] training KenLM order=%d on %d words ...", lang, order, len(words))

    with open(chars_file, "rb") as fin, open(arpa, "wb") as fout:
        subprocess.run([str(KENLM_BIN / "lmplz"), "-o", str(order), "--discount_fallback"],
                    stdin=fin, stdout=fout, stderr=subprocess.DEVNULL, check=True)
    return arpa


class KenLMDetector:
    """Scores a word with a character-level KenLM model."""
    def __init__(self, arpa):
        self.m = kenlm.Model(str(arpa))

    def score_word(self, w):
        # mean log10 probability per character
        if len(w) < 1:
            return -20.0
        return self.m.score(" ".join(w), bos=True, eos=True) / len(w)


def train_lr(detector, val_words, seed=42):
    """
    Fits the regression on the KenLM scores of the validation words and of the same words after
    corruption. Returns the regression and the boundary tau = -intercept / coef.
    """
    wr = SpellingWrecker(corruption_rate=1.0, seed=seed)
    pos = [detector.score_word(w) for w in val_words if len(w) >= MIN_WORD_LEN]
    neg = []
    corrupted_text, gt = wr.corrupt_text(" ".join(val_words), min_word_length=MIN_WORD_LEN)
    for w in corrupted_text.split():
        if w in gt and len(w) >= MIN_WORD_LEN:
            neg.append(detector.score_word(w))
    X = np.array([[s] for s in pos] + [[s] for s in neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(X, y)
    tau = -float(clf.intercept_[0]) / float(clf.coef_[0][0])
    logger.info("KenLM LR boundary tau=%.4f (pos=%d neg=%d)", tau, len(pos), len(neg))
    return clf, tau


def flag(detector, clf, w):
    """Returns whether the detector flags a word, and its score."""
    if len(w) < MIN_WORD_LEN:
        return False, None
    s = detector.score_word(w)
    # the same decision as clf.predict
    return ((clf.coef_[0][0] * s + clf.intercept_[0]) <= 0.0), s


def load_groundtruth():
    """Reads the 500-word evaluation set."""
    p = EVAL_DIR / "rq3_500_groundtruth.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run scripts/make_rq3_eval_set.py first")
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pairs_for(gt, ctype, rate):
    """
    Returns every word of one corruption type with its error label. At a rate of 1.0 these are
    the clean words, labelled 0, followed by their corrupted forms.
    """
    if rate == 1.0:
        rows = [(r["original"], 0) for r in gt]
        rows += [(r[f"{ctype}_rate1.0_corrupted"], int(r[f"{ctype}_rate1.0_is_error"])) for r in gt]
        return rows
    return [(r[f"{ctype}_rate{rate}_corrupted"], int(r[f"{ctype}_rate{rate}_is_error"])) for r in gt]


def run(lang, order):
    """Trains the KenLM detector for one language and evaluates it on the fixed set."""
    arpa = build_char_lm(lang, order)
    detector = KenLMDetector(arpa)
    clf, tau = train_lr(detector, load_words(corpus_path(lang, "validation")))

    gt = load_groundtruth()
    active = mlflow.active_run() is not None
    if not active:
        mlflow.set_experiment("RQ3_Baseline_Comparison")
    with (nullcontext() if active else mlflow.start_run()):
        mlflow.log_metric(f"kenlm_o{order}_lr_tau", tau)
        summary_rows, word_rows = [], []
        print(f"\n{'='*96}\n  RQ3 external comparison - KenLM (char {order}-gram + LR) on the "
            f"500-word set ({lang})\n  LR boundary tau={tau:.4f}\n{'='*96}")
        print(f"  {'config':<22}{'tp':>5}{'fp':>5}{'fn':>5}{'tn':>5}"
            f"{'incP':>8}{'incR':>8}{'incF1':>8}{'corF1':>8}{'mF1':>8}")
        print(f"  {'-'*92}")
        for ctype in TYPES:
            tp = fp = fn = tn = 0
            for r in gt:
                for instance, w, label in [("clean", r["original"], 0),
                                            ("corrupted", r[f"{ctype}_rate1.0_corrupted"],
                                            int(r[f"{ctype}_rate1.0_is_error"]))]:
                    fl, s = flag(detector, clf, w)
                    oc = "tp" if label and fl else "fp" if fl else "fn" if label else "tn"
                    tp += oc == "tp"; fp += oc == "fp"; fn += oc == "fn"; tn += oc == "tn"
                    word_rows.append({"corruption_type": ctype, "instance": instance,
                                    "original": r["original"],
                                    "corrupted": "" if instance == "clean" else w,
                                    "is_error": label,
                                    "kenlm_score": ("" if s is None else round(s, 4)),
                                    "tau": round(tau, 4), "flagged": int(fl), "outcome": oc})

            mf1, p_e, r_e, f1_e, f1_c = confusion_to_mf1(tp, fp, fn, tn)
            summary_rows.append({"system": "kenlm", "corruption_type": ctype, "rate": 1.0,
                                "order": order, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                "incorrect_precision": round(p_e, 6),
                                "incorrect_recall": round(r_e, 6),
                                "incorrect_f1": round(f1_e, 6),
                                "correct_f1": round(f1_c, 6), "mf1": round(mf1, 6)})
            mlflow.log_metrics({f"kenlm_o{order}_mf1_{ctype}": mf1,
                                f"kenlm_o{order}_fp_{ctype}": fp})
            print(f"  {ctype+'_rate1.0':<22}{tp:>5}{fp:>5}{fn:>5}{tn:>5}"
                f"{p_e:>8.3f}{r_e:>8.3f}{f1_e:>8.3f}{f1_c:>8.3f}{mf1:>8.3f}")
        print(f"{'='*96}\n")
        sp = write_csv(EVAL_DIR / "rq3_external_kenlm_summary.csv", summary_rows)
        wp = write_csv(EVAL_DIR / "rq3_external_kenlm_words.csv", word_rows)
        for p in (sp, wp):
            if p:
                mlflow.log_artifact(p)
        print(f"  Wrote rq3_external_kenlm_summary.csv + rq3_external_kenlm_words.csv to {EVAL_DIR}")
        return summary_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zulu")
    ap.add_argument("--order", type=int, default=5, help="char n-gram order (default 5)")
    a = ap.parse_args()
    run(a.lang, a.order)
