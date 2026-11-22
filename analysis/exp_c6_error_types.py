"""
exp_c6_error_types.py - detection for each of the four error types.

The test words are corrupted with one error type at a time, and precision, recall and F1 are
given for incorrect and for correct words, with macro-F1.

The experiment is run under two corruption types, which are compared side by side.
    random   substituted and inserted characters are drawn uniformly from a-z
    qwerty   they are drawn from the keys next to the affected character

Deletion and transposition depend only on position and are the same under both.

Outputs (results/c6/<mode>-based/): c6_error_types_{lang}.json, the summary and per-word CSVs,
one chart per corruption type and the comparison charts
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
from exp_common import (ROOT, ERROR_TYPES, load_words, corpus_path, model_path, sample_unique,
                        sample_tokens, MIN_WORD_LEN, lr_boundary, write_csv, trigram_breakdown,
                        trigram_calc, disp, write_figure)
from src.models.trigram_model import TrigramModel
from src.models.threshold_classifier import ThresholdClassifier
from src.evaluation.spelling_wrecker import SpellingWrecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _metrics(tp, fp, fn, tn):
    """
    Returns precision, recall and F1 for incorrect and for correct words from the confusion counts.
    """
    error_recall    = tp / (tp + fn) if (tp + fn) else 0.0
    error_precision = tp / (tp + fp) if (tp + fp) else 0.0
    error_f1        = (2 * error_precision * error_recall /
                    (error_precision + error_recall)
                    if (error_precision + error_recall) else 0.0)
    lang_recall     = tn / (tn + fp) if (tn + fp) else 0.0
    lang_precision  = tn / (tn + fn) if (tn + fn) else 0.0
    lang_f1         = (2 * lang_precision * lang_recall /
                    (lang_precision + lang_recall)
                    if (lang_precision + lang_recall) else 0.0)
    mf1             = (error_f1 + lang_f1) / 2.0
    return {
        "error_recall":    error_recall,
        "error_precision": error_precision,
        "error_f1":        error_f1,
        "lang_recall":     lang_recall,
        "lang_precision":  lang_precision,
        "lang_f1":         lang_f1,
        "mf1":             mf1,
    }


def evaluate_type(detection, clf, test_words, error_type, seed, corruption_type,
                  min_word_length=MIN_WORD_LEN):
    """Evaluates one error type for one seed and returns the confusion counts and the metrics."""
    wrecker = SpellingWrecker(corruption_rate=1.0, seed=seed, error_types=[error_type],
                              corruption_type=corruption_type)
    generated, identical, corrupted_words = 0, 0, []
    for w in test_words:
        c, gt = wrecker.corrupt_text(w, min_word_length=min_word_length)
        c = c.strip()
        if c in gt:
            generated += 1; corrupted_words.append(c)
        else:
            identical += 1

    def flag(word):
        if len(word) < min_word_length: return False
        s = detection.score_word(word)
        return clf.predict_error(s) if clf else s < -8.0

    tp = sum(1 for w in corrupted_words if flag(w))
    fn = len(corrupted_words) - tp
    fp = sum(1 for w in test_words if flag(w))
    tn = len(test_words) - fp

    m = _metrics(tp, fp, fn, tn)
    m["tp"] = tp; m["fp"] = fp; m["fn"] = fn; m["tn"] = tn
    m["generated"]    = generated
    m["identical"]    = identical
    m["identical_pct"]= (identical / (generated + identical) * 100
                        if (generated + identical) else 0.0)
    return m


def _aggregate_mode(detection, clf, test_words, seeds, corruption_type, lang, min_word_length=MIN_WORD_LEN):
    """Runs the four error types under one corruption type and returns the mean metrics of each."""

    summary = {}
    for et in ERROR_TYPES:
        seed_res = [evaluate_type(detection, clf, test_words, et, s, corruption_type,
                    min_word_length)
                    for s in range(seeds)]

        metric_keys = ["error_recall","error_precision","error_f1",
                    "lang_recall","lang_precision","lang_f1","mf1",
                    "identical_pct","generated","identical"]

        agg = {k: float(np.mean([r[k] for r in seed_res])) for k in metric_keys}

        for k in ["error_recall","error_precision","error_f1","mf1"]:
            agg[f"{k}_std"] = float(np.std([r[k] for r in seed_res]))

        for k in ["tp","fp","fn","tn"]:
            agg[k] = sum(r[k] for r in seed_res)
        summary[et] = agg

        logger.info("[%s/%s] %-14s err_rec=%.3f err_prec=%.3f err_f1=%.3f "
                    "lang_rec=%.3f mF1=%.3f wasted=%.1f%%",
                    lang, corruption_type, et, agg["error_recall"], agg["error_precision"],
                    agg["error_f1"], agg["lang_recall"], agg["mf1"],
                    agg["identical_pct"])
    return summary


def _print_mode_table(summary, lang, corruption_type, seeds):
    """Prints the per-class metrics of one corruption type."""
    note = ("QWERTY-adjacent substitution/insertion" if corruption_type == "qwerty"
            else "uniform-random substitution/insertion")
    print(f"\n{'='*88}")
    print(f"  Per-error-type metrics  ({lang}, {seeds} seeds)  |  corruption_type={corruption_type}")
    print(f"  Character source: {note}")
    print(f"{'='*88}")
    print(f"  {'':<14}{'Incorrect words':^27}{'Correct words':^27}")
    print(f"  {'type':<14}{'Prec':>9}{'Rec':>9}{'F1':>9}"
        f"{'Prec':>9}{'Rec':>9}{'F1':>9}{'AvgF1':>9}{'wasted%':>9}")
    print(f"  {'-'*86}")
    for t in ERROR_TYPES:
        s = summary[t]
        print(f"  {t:<14}"
            f"{s['error_precision']:>9.3f}{s['error_recall']:>9.3f}{s['error_f1']:>9.3f}"
            f"{s['lang_precision']:>9.3f}{s['lang_recall']:>9.3f}{s['lang_f1']:>9.3f}"
            f"{s['mf1']:>9.3f}{s['identical_pct']:>9.1f}")


def _print_comparison(all_summaries, modes, lang):
    """Prints the two corruption types side by side, with the difference between them."""
    a, b = modes[0], modes[1]
    print(f"\n{'='*92}")
    print(f"  C6 - corruption-type comparison ({lang}): does '{b}' beat '{a}'? "
        f"(delta = {b} - {a})")
    print(f"  Hypothesis: keyboard-adjacent (qwerty) errors are more phonotactically "
        f"'normal',\n  so they should be EASIER for the trigram detector to flag "
        f"-> higher incorrect recall/F1 and mF1.")
    print(f"{'='*92}")

    hdr = (f"  {'type':<14}"
        f"{'incRec_'+a:>15}{'incRec_'+b:>15}{'d':>9}"
        f"{'incF1_'+a:>15}{'incF1_'+b:>15}{'d':>9}"
        f"{'mF1_'+a:>13}{'mF1_'+b:>13}{'d':>9}")
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")

    def row(label, sa, sb):
        d_rec = sb["error_recall"] - sa["error_recall"]
        d_f1  = sb["error_f1"]     - sa["error_f1"]
        d_mf1 = sb["mf1"]          - sa["mf1"]
        print(f"  {label:<14}"
            f"{sa['error_recall']:>15.3f}{sb['error_recall']:>15.3f}{d_rec:>+9.3f}"
            f"{sa['error_f1']:>15.3f}{sb['error_f1']:>15.3f}{d_f1:>+9.3f}"
            f"{sa['mf1']:>13.3f}{sb['mf1']:>13.3f}{d_mf1:>+9.3f}")

    for t in ERROR_TYPES:
        row(t, all_summaries[a][t], all_summaries[b][t])
    # overall is the mean over the four error types
    mean_a = {k: float(np.mean([all_summaries[a][t][k] for t in ERROR_TYPES]))
                for k in ["error_recall","error_f1","mf1"]}
    mean_b = {k: float(np.mean([all_summaries[b][t][k] for t in ERROR_TYPES]))
                for k in ["error_recall","error_f1","mf1"]}
    print(f"  {'-'*(len(hdr)-2)}")

    row("OVERALL(mean)", mean_a, mean_b)
    verdict = ("supports" if mean_b["mf1"] > mean_a["mf1"] else "does NOT support")
    print(f"\n  Verdict: overall mF1 {mean_b['mf1']:.3f} ({b}) vs {mean_a['mf1']:.3f} ({a}) "
        f"-> the data {verdict} the hypothesis (delta mF1 = {mean_b['mf1']-mean_a['mf1']:+.3f}).")
    print(f"{'='*92}\n")


def _plot_mode(summary, lang, corruption_type, seeds, out):
    """Draws recall and precision per error type for one corruption type, with macro-F1 overlaid."""
    x = np.arange(len(ERROR_TYPES)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*w, [summary[t]["error_recall"]   for t in ERROR_TYPES], w,
        label="Incorrect-word recall (detection rate)", color="#d62728", alpha=0.85)
    ax.bar(x - 0.5*w, [summary[t]["error_precision"] for t in ERROR_TYPES], w,
        label="Incorrect-word precision", color="#ff7f0e", alpha=0.85)
    ax.bar(x + 0.5*w, [summary[t]["lang_recall"]     for t in ERROR_TYPES], w,
        label="Correct-word recall", color="#1f77b4", alpha=0.85)
    ax.bar(x + 1.5*w, [summary[t]["lang_precision"]  for t in ERROR_TYPES], w,
        label="Correct-word precision", color="#2ca02c", alpha=0.85)

    mf1s = [summary[t]["mf1"] for t in ERROR_TYPES]
    ax.plot(x, mf1s, "k^--", linewidth=1.5, markersize=8, label="Macro F1 (mF1)", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in ERROR_TYPES], fontsize=11)
    ax.set_xlabel("Error type", fontsize=12)
    ax.set_ylabel("Score (precision / recall / F1)", fontsize=12)
    for xi, mf1 in zip(x, mf1s):
        ax.text(xi, mf1 + 0.015, f"{mf1:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, 1.09)
    ax.set_title(f"C6: Per-error-type metrics - {disp(lang)}  (error injection: {corruption_type}, "
                f"{seeds} seeds)", fontsize=11)

    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    pdf, _ = write_figure(fig, out, f"c6_error_types_{corruption_type}_{lang}", bbox_inches="tight")
    plt.close(fig)
    return pdf


# each metric compared between the corruption types has a key, a y-axis title and a file stem
_CMP_PANELS = [
    ("error_recall", "Incorrect-word recall (detection rate)", "c6_mode_comparison_incorrect_recall"),
    ("error_f1",     "Incorrect-word F1",                       "c6_mode_comparison_incorrect_f1"),
    ("mf1",          "Macro F1 (mF1)",                          "c6_mode_comparison_mf1"),
]


def _draw_cmp_panel(ax, all_summaries, modes, key, ytitle, legend_outside):
    """Draws one metric for the two corruption types onto an axis."""
    a, b = modes[0], modes[1]
    x = np.arange(len(ERROR_TYPES)); w = 0.38
    va = [all_summaries[a][t][key] for t in ERROR_TYPES]
    vb = [all_summaries[b][t][key] for t in ERROR_TYPES]
    ax.bar(x - w/2, va, w, label=a, color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, vb, w, label=b, color="#ff7f0e", alpha=0.85)
    for xi, v in zip(x - w/2, va):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    for xi, v in zip(x + w/2, vb):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in ERROR_TYPES], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Error type", fontsize=11)
    ax.set_ylabel(ytitle, fontsize=11)
    ax.set_title(ytitle, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    if legend_outside:
        ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                    title="Error injection")
    else:
        ax.legend(fontsize=9, loc="lower right", title="Error injection")


def _plot_comparison(all_summaries, modes, lang, seeds, out):
    """
    Draws the comparison of the two corruption types on three metrics, as one combined figure
    and as a separate figure per metric. Returns the path of the combined figure.
    """
    a, b = modes[0], modes[1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, (key, ytitle, _stem) in zip(axes, _CMP_PANELS):
        _draw_cmp_panel(ax, all_summaries, modes, key, ytitle, legend_outside=True)
    fig.suptitle(f"C6: error-injection comparison - {disp(lang)} ({seeds} seeds): {a} vs {b}",
                fontsize=13, fontweight="bold")
    fig.tight_layout()
    combined, _ = write_figure(fig, out, f"c6_mode_comparison_{lang}", bbox_inches="tight")
    plt.close(fig)

    for key, ytitle, stem in _CMP_PANELS:
        f1, ax1 = plt.subplots(figsize=(8, 5))
        _draw_cmp_panel(ax1, all_summaries, modes, key, ytitle, legend_outside=True)
        ax1.set_title(f"{ytitle} - {disp(lang)} ({a} vs {b})", fontsize=11)
        f1.tight_layout()
        write_figure(f1, out, f"{stem}_{lang}", bbox_inches="tight")
        plt.close(f1)

    return combined


def _flag(detection, clf, word, min_word_length=MIN_WORD_LEN):
    """
    Returns whether the word is flagged, and its score. A word shorter than min_word_length is
    never flagged.
    """
    if len(word) < min_word_length:
        return False, None
    s = detection.score_word(word)
    return (clf.predict_error(s) if clf else s < -8.0), s


def _word_rows(detection, clf, test_words, corruption_types, seed, tau, min_word_length=MIN_WORD_LEN):
    """
    Returns the word-by-word rows of one seed. Every clean word is a false positive or a true
    negative, and every corrupted word, per corruption type and error type, is a true positive
    or a false negative. The confusion counts can be recomputed from these rows.
    """
    rows = []
    tau_s = "" if tau is None else round(tau, 6)
    def _tg(word):
        tris, logps, _ = trigram_breakdown(detection, word)
        return "|".join(tris), "|".join(f"{lp:.3f}" for lp in logps)
    # clean words do not depend on the corruption type or the error type
    for w in test_words:
        flagged, s = _flag(detection, clf, w, min_word_length)
        tg, lps = _tg(w)
        rows.append({"corruption_type": "-", "error_type": "(clean)", "instance": "clean",
                    "original": w, "corrupted": "", "generated": "",
                    "trigrams": tg, "trigram_logprobs": lps,
                    "trigram_calc": trigram_calc(detection, w),
                    "detection_score": ("" if s is None else round(s, 4)),
                    "tau": tau_s, "flagged": int(flagged),
                    "outcome": ("fp" if flagged else "tn")})
    for cm in corruption_types:
        for et in ERROR_TYPES:
            wr = SpellingWrecker(corruption_rate=1.0, seed=seed, error_types=[et], corruption_type=cm)
            for w in test_words:
                c, gt = wr.corrupt_text(w, min_word_length=min_word_length)
                c = c.strip()

                if c not in gt:  # the corruption reproduced the word
                    rows.append({"corruption_type": cm, "error_type": et,
                                "instance": "corrupted", "original": w,
                                "corrupted": c, "generated": 0,
                                "trigrams": "", "trigram_logprobs": "",
                                "trigram_calc": "",
                                "detection_score": "",
                                "tau": tau_s, "flagged": "", "outcome": "wasted_noop"})
                    continue
                flagged, s = _flag(detection, clf, c, min_word_length)
                tg, lps = _tg(c)
                rows.append({"corruption_type": cm, "error_type": et, "instance": "corrupted",
                            "original": w, "corrupted": c, "generated": 1,
                            "trigrams": tg, "trigram_logprobs": lps,
                            "trigram_calc": trigram_calc(detection, c),
                            "detection_score": ("" if s is None else round(s, 4)),
                            "tau": tau_s, "flagged": int(flagged),
                            "outcome": ("tp" if flagged else "fn")})
    return rows


def _run_mode(lang, seeds, out, sample_mode="type", min_word_length=3, corruption_types=("random", "qwerty"), n_words=4000):

    corruption_types = list(corruption_types)

    for m in corruption_types:
        if m not in ("random", "qwerty"):
            raise ValueError(f"char mode must be 'random' or 'qwerty', got {m!r}")

    _lm = mlflow.log_metrics
    logm = lambda d, **kw: _lm({f"{k}_{sample_mode}": v for k, v in d.items()}, **kw)

    detection = TrigramModel.load(model_path(lang, "trigram"))
    clf_p = model_path(lang, "classifier")
    clf   = ThresholdClassifier.load(clf_p) if clf_p.exists() else None
    b = lr_boundary(clf)
    if b is not None:
        logger.info("[%s] LR boundary tau=%+.4f (intercept=%+.4f coef=%+.4f)",
                    lang, b["tau"], b["intercept"], b["coef"])
        logm({"lr_tau": b["tau"], "lr_intercept": b["intercept"],
                            "lr_coef": b["coef"]})


    # the same test words are used for every corruption type
    _sample = sample_tokens if sample_mode == "token" else sample_unique
    test_words = _sample(load_words(corpus_path(lang, "test")), n_words, seed=42)
    logger.info("[%s] %d test words (sampled, seed=42), %d seeds, modes=%s",
                lang, len(test_words), seeds, corruption_types)

    all_summaries = {}
    for corruption_type in corruption_types:
        summary = _aggregate_mode(detection, clf, test_words, seeds, corruption_type, lang,
                        min_word_length)
        all_summaries[corruption_type] = summary

        for et in ERROR_TYPES:
            agg = summary[et]
            logm({
                f"{corruption_type}_error_recall_{et}":    agg["error_recall"],
                f"{corruption_type}_error_precision_{et}": agg["error_precision"],
                f"{corruption_type}_error_f1_{et}":        agg["error_f1"],
                f"{corruption_type}_lang_recall_{et}":     agg["lang_recall"],
                f"{corruption_type}_lang_precision_{et}":  agg["lang_precision"],
                f"{corruption_type}_lang_f1_{et}":         agg["lang_f1"],
                f"{corruption_type}_mF1_{et}":             agg["mf1"],
                f"{corruption_type}_identical_pct_{et}":   agg["identical_pct"],
            })

    out.mkdir(parents=True, exist_ok=True)
    (out / f"c6_error_types_{lang}.json").write_text(json.dumps(
        {"language": lang, "seeds": seeds, "corruption_types": corruption_types,
        "char_source": {
            "random": "substitution + insertion drawn uniformly from ascii_lowercase",
            "qwerty": "substitution + insertion drawn from QWERTY-adjacent keys"},
        "per_mode": all_summaries}, indent=2))
    mlflow.log_artifact(str(out / f"c6_error_types_{lang}.json"), artifact_path=f"{sample_mode}-based")

    # one row per corruption type and error type
    summary_rows = []
    for cm in corruption_types:
        for et in ERROR_TYPES:
            s = all_summaries[cm][et]
            summary_rows.append({
                "corruption_type": cm, "error_type": et, "seeds": seeds,
                "lr_tau": ("" if b is None else round(b["tau"], 6)),
                "lr_intercept": ("" if b is None else round(b["intercept"], 6)),
                "lr_coef": ("" if b is None else round(b["coef"], 6)),
                "tp": s["tp"], "fp": s["fp"], "fn": s["fn"], "tn": s["tn"],
                "incorrect_precision": round(s["error_precision"], 6),
                "incorrect_recall":    round(s["error_recall"], 6),
                "incorrect_f1":        round(s["error_f1"], 6),
                "correct_precision":   round(s["lang_precision"], 6),
                "correct_recall":      round(s["lang_recall"], 6),
                "correct_f1":          round(s["lang_f1"], 6),
                "mf1":                 round(s["mf1"], 6),
                "wasted_pct":          round(s["identical_pct"], 4)})
    csv1 = write_csv(out / f"c6_error_types_summary_{lang}.csv", summary_rows)
    if csv1: mlflow.log_artifact(csv1, artifact_path=f"{sample_mode}-based")

    # every test word of seed 0 and what happened to it
    tau = b["tau"] if b is not None else None
    word_rows = _word_rows(detection, clf, test_words, corruption_types, seed=0, tau=tau,
                            min_word_length=min_word_length)
    csv2 = write_csv(out / f"c6_words_seed0_{lang}.csv", word_rows)
    if csv2: mlflow.log_artifact(csv2, artifact_path=f"{sample_mode}-based")

    for corruption_type in corruption_types:
        p = _plot_mode(all_summaries[corruption_type], lang, corruption_type, seeds, out)
        mlflow.log_artifact(str(p), artifact_path=f"{sample_mode}-based")
    if len(corruption_types) >= 2:
        pc = _plot_comparison(all_summaries, corruption_types, lang, seeds, out)
        mlflow.log_artifact(str(pc), artifact_path=f"{sample_mode}-based")


    for corruption_type in corruption_types:
        _print_mode_table(all_summaries[corruption_type], lang, corruption_type, seeds)
    if len(corruption_types) >= 2:
        _print_comparison(all_summaries, corruption_types, lang)
        a, b2 = corruption_types[0], corruption_types[1]
        mean_mf1_a = float(np.mean([all_summaries[a][t]["mf1"] for t in ERROR_TYPES]))
        mean_mf1_b = float(np.mean([all_summaries[b2][t]["mf1"] for t in ERROR_TYPES]))
        logm({f"overall_mF1_{a}": mean_mf1_a,
                            f"overall_mF1_{b2}": mean_mf1_b,
                            "overall_mF1_delta_qwerty_minus_random":
                                mean_mf1_b - mean_mf1_a})
    return all_summaries


def run(lang, seeds, corruption_types=("random", "qwerty"), n_words=4000, modes=("type", "token")):
    """
    Runs the experiment for one language, once per mode. Short words are scored (min_word_length=1).
    """
    base = ROOT / "results" / "c6"
    mlflow.set_experiment("error_type_analysis")
    with mlflow.start_run():
        mlflow.log_params({"language": lang, "seeds": seeds, "corruption_types": list(corruption_types),
                            "n_words": n_words, "modes": ",".join(modes)})
        for mode in modes:
            _run_mode(lang, seeds, base / f"{mode}-based", sample_mode=mode, min_word_length=1,
                        corruption_types=corruption_types, n_words=n_words)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang",      default="zulu")
    ap.add_argument("--seeds",     type=int, default=30)
    ap.add_argument("--corruption-types", nargs="+", default=["random", "qwerty"],
                    choices=["random", "qwerty"],
                    help="character source(s) to run: random (uniform) and/or qwerty "
                        "(keyboard-adjacent); both by default")
    ap.add_argument("--n-words", type=int, default=4000,
                    help="number of unique test words to sample (seed 42)")
    a = ap.parse_args()
    run(a.lang, a.seeds, a.corruption_types, a.n_words)
