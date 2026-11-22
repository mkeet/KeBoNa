"""
exp_rq4_correction.py - correction accuracy (RQ4).

For every corrupted word that the detector flags, the word model returns a ranked list of
suggestions and the rank of the original word is recorded. An error that is not detected is
left out, because it is never offered for correction.

    CA@1   the top suggestion is the original word
    CA@5   the original word is among the top five
    MRR    mean reciprocal rank of the original word

Suggestions are ranked by edit distance, then by frequency, then by trigram score.

Outputs (results/rq4/):
    rq4_correction_{lang}.json             CA@1, CA@5 and MRR as mean and standard deviation
    rq4_correction_{lang}.pdf              bar chart
    rq4_corrections_seed0_{lang}.csv       every correction of seed 0
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import (ROOT, load_words, corpus_path, model_path,
                        mean_std, MIN_WORD_LEN, lr_boundary, write_csv)
from src.models.trigram_model import TrigramModel
from src.models.threshold_classifier import ThresholdClassifier
from src.evaluation.spelling_wrecker import SpellingWrecker
from src.utils.text_utils import TextUtils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_K = 5


def evaluate_seed(detection, clf, word_model, test_words, seed):
    """Returns CA@1, CA@5 and MRR over the detected errors of one seed."""
    wrecker = SpellingWrecker(corruption_rate=1.0, seed=seed)
    pairs = []
    for w in test_words:
        c, gt = wrecker.corrupt_text(w, min_word_length=MIN_WORD_LEN)
        c = c.strip()
        if c != w and c in gt:
            pairs.append((c, w))

    corrupted_to_original = dict(pairs)
    text = " ".join(p[0] for p in pairs)

    flagged = []
    for word in TextUtils.tokenize(TextUtils.normalize(text)):
        if len(word) < MIN_WORD_LEN:
            continue
        score = detection.score_word(word)
        is_err = clf.predict_error(score) if clf else score < -8.0
        if is_err:
            flagged.append(word)

    ca1 = ca5 = tp = 0
    rr = 0.0
    for word in flagged:
        if word not in corrupted_to_original:
            continue  # a false positive is not an error to correct
        tp += 1
        original = corrupted_to_original[word]
        sugg = word_model.suggest_corrections(word, max_suggestions=MAX_K, tie_breaker_model=detection)
        sug_words = [s[0] for s in sugg[:MAX_K]]
        if original in sug_words:
            rank = sug_words.index(original) + 1
            rr += 1.0 / rank
            ca5 += 1
            if rank == 1:
                ca1 += 1

    return (ca1 / tp if tp else 0.0,
            ca5 / tp if tp else 0.0,
            rr / tp if tp else 0.0)


def show_examples(detection, clf, word_model, test_words, seed=0, n=5):
    """
    Prints the first n flagged words in full. Each shows the corrupted word, the original, the
    score against the boundary, the ranked suggestions and the rank of the original.
    """
    b = lr_boundary(clf) if clf is not None else None
    tau = b["tau"] if b else None
    print(f"\n{'-'*78}")
    print(f"  WORKED EXAMPLE - first {n} flagged words (seed={seed})")
    if tau is not None:
        print(f"  Detection: score = avg log P(tri_k | tri_(k-2), tri_(k-1)) over the word's")
        print(f"  character trigrams; flagged as an error when score < tau = {tau:+.4f}.")
    print(f"  Ranking (suggest_corrections): candidates within edit distance <= 2 of the")
    print(f"  flagged word, sorted by (edit distance asc, word frequency desc, LM tiebreak).")
    print(f"  'frequency' is the word model's counts[word] (the word->frequency map in the")
    print(f"  {{lang}}_word.pkl). CA_at_1 = original ranked #1; CA_at_5 = original in top 5;")
    print(f"  MRR contribution = 1/rank.")
    print(f"{'-'*78}")

    wrecker = SpellingWrecker(corruption_rate=1.0, seed=seed)
    pairs = []
    for w in test_words:
        c, gt = wrecker.corrupt_text(w, min_word_length=MIN_WORD_LEN)
        c = c.strip()
        if c != w and c in gt:
            pairs.append((c, w))

    shown = 0
    for corrupted, original in pairs:
        if shown >= n:
            break
        if len(corrupted) < MIN_WORD_LEN:
            continue
        score = detection.score_word(corrupted)
        flagged = clf.predict_error(score) if clf else score < -8.0
        if not flagged:
            continue  # only the words the detector flags
        shown += 1
        sugg = word_model.suggest_corrections(corrupted, max_suggestions=MAX_K, tie_breaker_model=detection)
        sug_words = [s[0] for s in sugg]
        rank = sug_words.index(original) + 1 if original in sug_words else None

        print(f"\n  [{shown}] corrupted = '{corrupted}'   intended original = '{original}'")
        print(f"      detection score = {score:+.4f}  ->  flagged as error "
            f"({score:+.4f} < tau {tau:+.4f})" if tau is not None
            else f"      detection score = {score:+.4f}  -> flagged")
        print(f"      ranked suggestions (rank. word  freq):")
        for i, (sw, fr) in enumerate(sugg, 1):
            mark = "   <== intended original" if sw == original else ""
            print(f"        {i}. {sw:<24} {int(fr):>8}{mark}")
        if rank is not None:
            print(f"      -> original at rank {rank}: "
                f"CA_at_1 {'HIT (+1)' if rank == 1 else 'miss (+0)'}, "
                f"CA_at_5 HIT (+1), MRR += 1/{rank} = {1.0/rank:.3f}")
        else:
            print(f"      -> original NOT in top {MAX_K}: "
                f"CA_at_1 +0, CA_at_5 +0, MRR += 0")
    if shown == 0:
        print("  (no words were flagged in this sample - nothing to trace)")
    print(f"{'-'*78}")


def correction_rows(detection, clf, word_model, test_words, seed=0):
    """
    Returns one row per flagged word of one seed, with the suggestions and the rank of the
    original, so that CA@k and MRR can be recomputed from the CSV.
    """
    wrecker = SpellingWrecker(corruption_rate=1.0, seed=seed)
    rows = []
    for w in test_words:
        c, gt = wrecker.corrupt_text(w, min_word_length=MIN_WORD_LEN); c = c.strip()
        if not (c != w and c in gt and len(c) >= MIN_WORD_LEN):
            continue
        score = detection.score_word(c)
        flagged = clf.predict_error(score) if clf else score < -8.0
        if not flagged:
            continue
        sugg = word_model.suggest_corrections(c, max_suggestions=MAX_K, tie_breaker_model=detection)
        sw = [s[0] for s in sugg]
        rank = sw.index(w) + 1 if w in sw else 0
        rows.append({"corrupted": c, "intended_original": w,
                    "detection_score": round(score, 4),
                    "suggestions": "|".join(sw), "rank_of_original": rank,
                    "ca_at_1": int(rank == 1), "ca_at_5": int(1 <= rank <= MAX_K),
                    "reciprocal_rank": round(1.0 / rank, 4) if rank else 0.0})
    return rows


def run(lang, seeds, quick):

    """
    Runs the experiment for one language. quick limits the test words to a sample of that
    size, and None uses the whole test split.
    """
    mlflow.set_experiment("RQ4_Correction_Accuracy")

    with mlflow.start_run():
        mlflow.log_params({
            "experiment": "rq4_correction",
            "language": lang,
            "seeds": seeds,
            "quick": quick,
        })

        detection = TrigramModel.load(model_path(lang, "trigram"))
        clf_p = model_path(lang, "classifier")
        clf = ThresholdClassifier.load(clf_p) if clf_p.exists() else None
        word_model = TrigramModel.load(model_path(lang, "word"))

        test_words = load_words(corpus_path(lang, "test"))
        if quick:
            test_words = test_words[:quick]
        logger.info("[%s] correction eval on %d test words, %d seeds",
                    lang, len(test_words), seeds)

        show_examples(detection, clf, word_model, test_words, seed=0, n=5)
        _cr = write_csv(ROOT / "results" / "rq4" / f"rq4_corrections_seed0_{lang}.csv",
                        correction_rows(detection, clf, word_model, test_words, seed=0))
        if _cr:
            mlflow.log_artifact(_cr)

        ca1s, ca5s, mrrs = [], [], []
        for s in range(seeds):
            ca1, ca5, mrr = evaluate_seed(detection, clf, word_model, test_words, s)
            ca1s.append(ca1)
            ca5s.append(ca5)
            mrrs.append(mrr)
            logger.info("seed %2d: CA_at_1=%.3f CA_at_5=%.3f MRR=%.3f", s, ca1, ca5, mrr)

        res = {k: dict(zip(("mean", "std"), mean_std(v)))
            for k, v in [("ca1", ca1s), ("ca5", ca5s), ("mrr", mrrs)]}
        mlflow.log_metrics({
            "CA_at_1_mean": res["ca1"]["mean"],
            "CA_at_1_std": res["ca1"]["std"],
            "CA_at_5_mean": res["ca5"]["mean"],
            "CA_at_5_std": res["ca5"]["std"],
            "MRR_mean": res["mrr"]["mean"],
            "MRR_std": res["mrr"]["std"],
        })

        out = ROOT / "results" / "rq4"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"rq4_correction_{lang}.json").write_text(json.dumps(res, indent=2))
        mlflow.log_artifact(str(out / f"rq4_correction_{lang}.json"))

        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["CA_at_1", "CA_at_5", "MRR"]
        means = [res["ca1"]["mean"], res["ca5"]["mean"], res["mrr"]["mean"]]
        stds = [res["ca1"]["std"], res["ca5"]["std"], res["mrr"]["std"]]
        ax.bar(labels, means, yerr=stds, capsize=5, color="#1f77b4", alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title(f"RQ4: Correction accuracy ({lang})")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"rq4_correction_{lang}.pdf", dpi=150)
        mlflow.log_artifact(str(out / f"rq4_correction_{lang}.pdf"))

        logger.info("Saved results/rq4_correction.{json,pdf}")

        print(f"\n{'='*50}\n  RQ4 - {lang} correction accuracy\n{'='*50}")
        print(f"  CA_at_1 = {res['ca1']['mean']:.3f} +/- {res['ca1']['std']:.3f}")
        print(f"  CA_at_5 = {res['ca5']['mean']:.3f} +/- {res['ca5']['std']:.3f}")
        print(f"  MRR  = {res['mrr']['mean']:.3f} +/- {res['mrr']['std']:.3f}\n{'='*50}\n")

        return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zulu")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--quick", type=int, default=None,
                    help="Limit to first N test words (faster, less rigorous)")
    a = ap.parse_args()
    run(a.lang, a.seeds, a.quick)
