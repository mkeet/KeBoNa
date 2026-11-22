"""
exp_c8_fp_fn_analysis.py - why the checker errs, and the ceiling this puts on detection.

Every false negative and false positive is sorted into a category, and example words are
listed for each.

    false negatives   became_valid_word, identical_to_original, short_word, other_missed
    false positives   rare_valid_word, short_word, other_fp

The first two categories of false negative do not depend on the model. Either the corruption
produced a real word, or it produced no change at all. They limit recall whatever the model.

Outputs (results/c8/<mode>-based/):
    c8_fp_fn_{lang}.json            category counts and example words
    c8_categories_{lang}.{pdf,png}  chart of the categories
    c8_words_seed0_{lang}.csv       seed 0 word by word
"""

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import (ROOT, load_words, corpus_path, model_path, MIN_WORD_LEN, lr_boundary,
                        write_csv, audit_rows, write_figure, disp)
from src.models.trigram_model import TrigramModel
from src.models.threshold_classifier import ThresholdClassifier
from src.evaluation.spelling_wrecker import SpellingWrecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run_mode(lang, seeds, n_examples, out_dir, sample_mode="type", corruption_type="random"):

    detection = TrigramModel.load(model_path(lang, "trigram"))
    clf_p = model_path(lang, "classifier")
    clf = ThresholdClassifier.load(clf_p) if clf_p.exists() else None
    boundary = lr_boundary(clf) if clf is not None else None

    if boundary is not None:
        logger.info("[%s] LR boundary tau=%+.4f (intercept=%+.4f coef=%+.4f)",
                    lang, boundary["tau"], boundary["intercept"], boundary["coef"])
        mlflow.log_metrics({f"lr_tau_{sample_mode}": boundary["tau"],
                            f"lr_intercept_{sample_mode}": boundary["intercept"],
                            f"lr_coef_{sample_mode}": boundary["coef"]})

    word_model = TrigramModel.load(model_path(lang, "word"))
    vocab = set(w.lower() for w in word_model.vocab)
    _all = load_words(corpus_path(lang, "test"))
    test_words = list(dict.fromkeys(_all)) if sample_mode == "type" else _all

    # short words are scored, and MIN_WORD_LEN only labels the short_word categories below
    min_word_length = 1
    def flag(word):
        if len(word) < min_word_length:
            return False
        s = detection.score_word(word)
        return clf.predict_error(s) if clf else s < -8.0

    fn_cats = Counter()
    fp_cats = Counter()
    fn_examples = defaultdict(list)
    fp_examples = defaultdict(list)

    # confusion counts summed over the seeds
    tp = fp = fn = tn = 0

    for seed in range(seeds):
        wrecker = SpellingWrecker(corruption_rate=1.0, seed=seed, corruption_type=corruption_type)

        for w in test_words:
            c, gt = wrecker.corrupt_text(w, min_word_length=min_word_length)
            c = c.strip()
            if c not in gt:
                if len(w) >= MIN_WORD_LEN:
                    fn_cats["identical_to_original"] += 1
                    if len(fn_examples["identical_to_original"]) < n_examples:
                        fn_examples["identical_to_original"].append(f"{w}->{c}")
                continue
            if flag(c):
                tp += 1
                continue
            fn += 1
            if len(c) < MIN_WORD_LEN:
                fn_cats["short_word"] += 1
                bucket = "short_word"
            elif c.lower() in vocab:
                fn_cats["became_valid_word"] += 1
                bucket = "became_valid_word"
            else:
                fn_cats["other_missed"] += 1
                bucket = "other_missed"
            if len(fn_examples[bucket]) < n_examples:
                fn_examples[bucket].append(f"{gt[c]}->{c}")

        for w in test_words:
            if not flag(w):
                tn += 1
                continue
            fp += 1
            if len(w) < MIN_WORD_LEN:
                fp_cats["short_word"] += 1
                bucket = "short_word"
            else:
                fp_cats["rare_valid_word"] += 1
                bucket = "rare_valid_word"
            if len(fp_examples[bucket]) < n_examples:
                fp_examples[bucket].append(w)

    for k, v in fn_cats.items():
        mlflow.log_metric(f"fn_{k}_{sample_mode}", v)
    for k, v in fp_cats.items():
        mlflow.log_metric(f"fp_{k}_{sample_mode}", v)

    p_e = tp / (tp + fp) if (tp + fp) else 0.0
    r_e = tp / (tp + fn) if (tp + fn) else 0.0
    f1_e = 2 * p_e * r_e / (p_e + r_e) if (p_e + r_e) else 0.0
    p_c = tn / (tn + fn) if (tn + fn) else 0.0
    r_c = tn / (tn + fp) if (tn + fp) else 0.0
    f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) else 0.0
    mf1 = (f1_e + f1_c) / 2.0

    detection_metrics = {
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "incorrect": {"precision": p_e, "recall": r_e, "f1": f1_e},
        "correct":   {"precision": p_c, "recall": r_c, "f1": f1_c},
        "mf1": mf1,
    }
    mlflow.log_metrics({f"mF1_{sample_mode}": mf1, f"incorrect_f1_{sample_mode}": f1_e, f"correct_f1_{sample_mode}": f1_c})

    out = {
        "language": lang,
        "seeds": seeds,
        "lr_boundary": boundary,
        "detection_metrics": detection_metrics,
        "false_negative_categories": dict(fn_cats),
        "false_positive_categories": dict(fp_cats),
        "false_negative_examples": {k: v for k, v in fn_examples.items()},
        "false_positive_examples": {k: v for k, v in fp_examples.items()},
    }
    intrinsic_fn = fn_cats["became_valid_word"] + fn_cats["identical_to_original"]
    total_fn = sum(fn_cats.values())
    out["intrinsic_fn_share_pct"] = (100 * intrinsic_fn / total_fn) if total_fn else 0.0
    mlflow.log_metric(f"intrinsic_fn_share_pct_{sample_mode}", out["intrinsic_fn_share_pct"])

    res_dir = out_dir
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"c8_fp_fn_{lang}.json").write_text(json.dumps(out, indent=2))

    cats = [("Missed: other (improvable)", fn_cats.get("other_missed", 0), "#1f77b4"),
            ("Missed: became a VALID word", fn_cats.get("became_valid_word", 0), "#d62728"),
            ("Missed: corruption was a no-op", fn_cats.get("identical_to_original", 0), "#ff7f0e"),
            ("False alarm: rare valid word", fp_cats.get("rare_valid_word", 0), "#2ca02c"),
            ("False alarm: short word", fp_cats.get("short_word", 0), "#9467bd")]
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ys = range(len(cats))
    ax.barh(list(ys), [c[1] for c in cats], color=[c[2] for c in cats], alpha=0.9)
    top = max(c[1] for c in cats) or 1
    for y, (_, v, _) in zip(ys, cats):
        ax.text(v + top * 0.015, y, f"{v:,}", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(list(ys)); ax.set_yticklabels([c[0] for c in cats], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.34)
    if top >= 1e6:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:g}M"))
    elif top >= 1e3:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:g}k"))
    ax.set_xlabel(f"Count ({disp(lang)}, {sample_mode}-level, {seeds} seeds)", fontsize=11)
    ax.set_title(f"C8: why detection fails - {out['intrinsic_fn_share_pct']:.0f}% "
                f"intrinsically undetectable", fontsize=10, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _pdf, _png = write_figure(fig, res_dir, f"c8_categories_{lang}")
    plt.close(fig)
    mlflow.log_artifact(str(_pdf), artifact_path=f"{sample_mode}-based")
    mlflow.log_artifact(str(_png), artifact_path=f"{sample_mode}-based")

    # seed 0 word by word
    a_rows, _, _ = audit_rows(detection, clf, test_words, seed=0,
                            extra={"language": lang, "corruption_type": corruption_type},
                            min_word_length=min_word_length)
    _ap = write_csv(res_dir / f"c8_words_seed0_{lang}.csv", a_rows)

    if _ap:
        mlflow.log_artifact(_ap, artifact_path=f"{sample_mode}-based")
    mlflow.log_artifact(str(res_dir / f"c8_fp_fn_{lang}.json"), artifact_path=f"{sample_mode}-based")

    logger.info("Saved results/c8_fp_fn_%s.json", lang)

    print(f"\n{'='*68}\n  C8 - FP/FN ceiling analysis ({lang})\n{'='*68}")
    if boundary is not None:
        print(f"  LR boundary: tau={boundary['tau']:+.4f}  "
            f"(intercept={boundary['intercept']:+.4f}  coef={boundary['coef']:+.4f})")
    print(f"\n  Detection metrics (micro-average over {seeds} seeds):")
    print(f"    {'':<10}{'Prec':>9}{'Rec':>9}{'F1':>9}")
    print(f"    {'Incorrect':<10}{p_e:>9.4f}{r_e:>9.4f}{f1_e:>9.4f}")
    print(f"    {'Correct':<10}{p_c:>9.4f}{r_c:>9.4f}{f1_c:>9.4f}")
    print(f"    {'Avg F1':<10}{mf1:>27.4f}")
    print(f"    confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}\n")
    print("  FALSE NEGATIVES (missed errors):")
    for k, v in fn_cats.most_common():
        print(f"    {k:<24}{v:>8}")
    print("  FALSE POSITIVES (wrongly flagged correct words):")
    for k, v in fp_cats.most_common():
        print(f"    {k:<24}{v:>8}")
    print(f"\n  Intrinsically-undetectable FNs (valid-word + identical): "
        f"{out['intrinsic_fn_share_pct']:.1f}% of all FNs")
    print(f"  -> This bounds achievable recall; the rest is model-improvable.")
    print("\n  Example FNs that became valid words:")
    for ex in fn_examples.get("became_valid_word", [])[:10]:
        print(f"    {ex}")
    print("\n  Example FPs (rare valid words flagged):")
    for ex in fp_examples.get("rare_valid_word", [])[:10]:
        print(f"    {ex}")
    print(f"{'='*68}\n")

    return out


def run(lang, seeds, n_examples, corruption_type="random", modes=("type", "token")):
    """Runs the analysis for one language, once per mode."""
    base = ROOT / "results" / "c8"
    mlflow.set_experiment("fp_fn_analysis")
    with mlflow.start_run():
        mlflow.log_params({"experiment": "c8_fp_fn_analysis", "language": lang, "seeds": seeds,
                        "n_examples": n_examples, "corruption_type": corruption_type,
                        "modes": ",".join(modes)})
        for mode in modes:
            _run_mode(lang, seeds, n_examples, base / f"{mode}-based",
                    sample_mode=mode, corruption_type=corruption_type)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zulu")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--examples", type=int, default=25)
    ap.add_argument("--corruption-type", default="random",
                    choices=["random", "qwerty"])
    a = ap.parse_args()
    run(a.lang, a.seeds, a.examples, a.corruption_type)
