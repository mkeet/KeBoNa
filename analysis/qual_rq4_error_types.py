"""
qual_rq4_error_types.py - correction accuracy for each of the four error types.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import (ROOT, ERROR_TYPES, load_words, corpus_path, model_path,
                        mean_std, write_csv, write_figure, disp, MIN_WORD_LEN)
from src.models.trigram_model import TrigramModel
from src.models.threshold_classifier import ThresholdClassifier
from src.evaluation.spelling_wrecker import SpellingWrecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RQ4 = ROOT / "results" / "rq4"
MAX_K = 5


def evaluate_type(detection, clf, word_model, test_words, error_type, seed):
    """
    Returns CA@1, CA@5 and MRR over the errors of one type that detection flags, with the
    number of those errors.
    """
    wrecker = SpellingWrecker(corruption_rate=1.0, seed=seed, error_types=[error_type])
    ca1 = ca5 = n = 0
    rr = 0.0
    for w in test_words:
        c, gt = wrecker.corrupt_text(w, min_word_length=MIN_WORD_LEN)
        c = c.strip()
        if not (c != w and c in gt and len(c) >= MIN_WORD_LEN):
            continue
        score = detection.score_word(c)
        if not (clf.predict_error(score) if clf else score < -8.0):
            continue  # an undetected error is never offered for correction
        n += 1
        sugg = [s[0] for s in word_model.suggest_corrections(c, max_suggestions=MAX_K, tie_breaker_model=detection)]
        if w in sugg:
            k = sugg.index(w) + 1
            rr += 1.0 / k; ca5 += 1; ca1 += (k == 1)
    return (ca1 / n if n else 0.0, ca5 / n if n else 0.0, rr / n if n else 0.0, n)


def run(lang="zu", seeds=30, quick=4000):
    """Evaluates each error type over the seeds and writes the table and the chart."""
    detection = TrigramModel.load(model_path(lang, "trigram"))
    clf_p = model_path(lang, "classifier")
    clf = ThresholdClassifier.load(clf_p) if clf_p.exists() else None
    word_model = TrigramModel.load(model_path(lang, "word"))
    test_words = load_words(corpus_path(lang, "test"))
    if quick:
        test_words = test_words[:quick]
    logger.info("[%s] correction by error type on %d words, %d seeds", lang, len(test_words), seeds)

    mlflow.set_experiment("RQ4_Correction_Qualitative")
    with mlflow.start_run():
        mlflow.log_params({"experiment": "qual_rq4_error_types", "language": lang,
                           "seeds": seeds, "quick": quick, "max_k": MAX_K})
        rows = []
        for et in ERROR_TYPES:
            per = [evaluate_type(detection, clf, word_model, test_words, et, s) for s in range(seeds)]
            ca1, s1 = mean_std([p[0] for p in per])
            ca5, s5 = mean_std([p[1] for p in per])
            mrr, sm = mean_std([p[2] for p in per])
            rows.append({"error_type": et, "n_mean": float(np.mean([p[3] for p in per])),
                         "ca1": ca1, "ca1_std": s1, "ca5": ca5, "ca5_std": s5,
                         "mrr": mrr, "mrr_std": sm})
            mlflow.log_metrics({f"ca1_{et}": ca1, f"ca5_{et}": ca5, f"mrr_{et}": mrr})
            logger.info("  %-14s CA@1=%.3f+/-%.3f CA@5=%.3f MRR=%.3f (n=%.0f)",
                        et, ca1, s1, ca5, mrr, rows[-1]["n_mean"])

        RQ4.mkdir(parents=True, exist_ok=True)
        csv_p = write_csv(RQ4 / f"qual_rq4_error_types_{lang}.csv", rows)
        (RQ4 / f"qual_rq4_error_types_{lang}.json").write_text(
            json.dumps({"language": lang, "seeds": seeds, "by_error_type": rows}, indent=2))

        order = sorted(rows, key=lambda r: -r["ca1"])
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        x = np.arange(len(order)); w = 0.27
        for i, (k, lbl, col) in enumerate((("ca1", "CA@1", "#1f77b4"),
                                           ("ca5", "CA@5", "#98c379"),
                                           ("mrr", "MRR", "#ff7f0e"))):
            ax.bar(x + (i - 1) * w, [r[k] for r in order], w, label=lbl, color=col,
                   yerr=[r[k + "_std"] for r in order], capsize=3)
        ax.set_xticks(x); ax.set_xticklabels([r["error_type"] for r in order])
        ax.set_ylim(0, 1.0); ax.set_ylabel("Correction accuracy")
        ax.set_title(f"Correction accuracy by error type ({disp(lang)}, {seeds} seeds)",
                     fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3); ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        pdf, png = write_figure(fig, RQ4, f"qual_rq4_error_types_{lang}")
        plt.close(fig)
        for p in (csv_p, RQ4 / f"qual_rq4_error_types_{lang}.json", pdf, png):
            if p and Path(p).exists():
                mlflow.log_artifact(str(p))

    print(f"\n{'='*68}\n  RQ4 correction by error type ({lang}, {seeds} seeds)\n{'='*68}")
    print(f"  {'error type':<16}{'n':>8}{'CA@1':>9}{'CA@5':>9}{'MRR':>9}")
    for r in sorted(rows, key=lambda r: -r["ca1"]):
        print(f"  {r['error_type']:<16}{r['n_mean']:>8.0f}{r['ca1']:>9.3f}{r['ca5']:>9.3f}{r['mrr']:>9.3f}")
    print(f"{'='*68}\n")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zu")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--quick", type=int, default=4000)
    a = ap.parse_args()
    run(a.lang, a.seeds, a.quick)
