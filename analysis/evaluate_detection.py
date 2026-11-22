"""
evaluate_detection.py - detection macro-F1 per language, over N seeds with a 95% confidence
interval.

Each language is evaluated on a sample of its test split under the balanced protocol, once on
word types and once on tokens.

Outputs (per mode, in <out>/<mode>-based/): evaluation_detection.json, .pdf, .png,
evaluation_detection_words_seed0.csv, evaluation_detection_confusion.csv
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_common import audit_rows, class_metrics, disp, lr_boundary, write_csv, write_figure
from src.core.spell_checker import SpellChecker
from src.evaluation.spelling_wrecker import SpellingWrecker

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MIN_WORD_LEN = 3

LANG_LABELS = {
    "zulu":    "isiZulu\n(SADiLaR)",  "xhosa":   "isiXhosa\n(SADiLaR)",
    "ndebele": "isiNdebele\n(SADiLaR)","swati":  "siSwati\n(SADiLaR)",
    "zu":      "isiZulu\n(NCHLT)",     "xh":      "isiXhosa\n(NCHLT)",
    "nd":      "isiNdebele\n(NCHLT)", "sw":       "siSwati\n(NCHLT)",
}

SADILAR_ORDER = ["zulu","xhosa","ndebele","swati"]
NCHLT_ORDER   = ["zu","xh","nd","sw"]


def _corpus_tag(langs):
    """
    Returns SADiLaR or NCHLT when every language of the run comes from that one source, and
    None for a mixed run. A single-source run labels its chart with bare language names, and
    a mixed run keeps the source in each label so that zulu and zu stay apart.
    """
    s = set(langs)
    if s and s <= set(SADILAR_ORDER):
        return "SADiLaR"
    if s and s <= set(NCHLT_ORDER):
        return "NCHLT"
    return None

LANG_COLOURS = {
    "zulu": "#1f77b4", "xhosa": "#d62728",
    "ndebele": "#2ca02c", "swati": "#9467bd",
    "zu":  "#aec7e8", "xh":  "#ff9896",
    "nd":  "#98df8a", "sw":  "#c5b0d5",
}


def _eval_seed(checker, lang, test_words, seed, min_word_length=MIN_WORD_LEN):
    """
    Runs one seed of the balanced protocol and counts every sampled word once. Type and token
    evaluation differ only in the sample passed in.
    """
    wrecker = SpellingWrecker(corruption_rate=1.0, seed=seed)
    corrupted = [wrecker.corrupt_text(w, min_word_length=min_word_length)[0].strip()
                for w in test_words]

    res_c = checker.check(" ".join(corrupted), language=lang,
                    include_suggestions=False, min_word_length=min_word_length)
    flagged_c = {e["word"].lower() for e in res_c["errors"]
                if len(e["word"]) >= min_word_length}
    res_clean = checker.check(" ".join(test_words), language=lang,
                    include_suggestions=False, min_word_length=min_word_length)
    flagged_clean = {e["word"].lower() for e in res_clean["errors"]
                    if len(e["word"]) >= min_word_length}

    corr_l = [c.lower() for c in corrupted]; test_l = [w.lower() for w in test_words]
    tp = sum(1 for c in corr_l if c in flagged_c); fn = len(corr_l) - tp
    fp = sum(1 for w in test_l if w in flagged_clean); tn = len(test_l) - fp
    m = class_metrics(tp, fp, fn, tn)

    return {"mf1": m["mf1"], "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": m["incorrect_precision"], "recall": m["incorrect_recall"],
            "f1_error": m["incorrect_f1"], "precision_correct": m["correct_precision"],
            "recall_correct": m["correct_recall"], "f1_correct": m["correct_f1"]}


def _ci95(values):
    """Returns the half-width of the 95% confidence interval of the mean, from Student's t."""
    n = len(values)
    if n < 2: return 0.0
    return float(stats.t.ppf(0.975, df=n-1) * np.std(values, ddof=1) / np.sqrt(n))


def _run_mode(lang_list, n_seeds, n_words, output_dir, sample_mode="type",
              min_word_length=MIN_WORD_LEN):
    checker = SpellChecker()
    checker.load_models(ROOT/"data"/"models")

    tag = _corpus_tag(lang_list)
    def _lab(l):
        return disp(l) if tag else LANG_LABELS.get(l, l)

    print(f"\n  LR decision boundary (tau = -intercept / coef):")
    for lang in lang_list:
        clf = checker.threshold_classifiers.get(lang)
        b = lr_boundary(clf) if clf is not None else None
        if b is None:
            print(f"    {lang:<10}  (no trained classifier; fallback threshold "
                f"{getattr(clf, 'fallback_threshold', -10.0):+.3f})")
            continue
        print(f"    {lang:<10}  tau={b['tau']:+.4f}   "
            f"intercept={b['intercept']:+.4f}  coef={b['coef']:+.4f}")
        mlflow.log_metrics({f"lr_tau_{lang}_{sample_mode}": b["tau"],
                            f"lr_intercept_{lang}_{sample_mode}": b["intercept"],
                            f"lr_coef_{lang}_{sample_mode}": b["coef"]})

    results = {}
    per_seed_mf1 = {}
    word_audit_rows = []
    confusion_rows = []
    for lang in lang_list:
        tp_path = ROOT/"data"/"corpora"/lang/"test.txt"
        if not tp_path.exists() or lang not in checker.detection_models:
            logger.warning("[%s] skipping", lang); continue
        all_words = [l.strip() for l in open(tp_path, encoding="utf-8") if l.strip()]
        pool = all_words if sample_mode == "token" else list(dict.fromkeys(all_words))
        test_words = (random.Random(42).sample(pool, n_words)
                    if n_words and len(pool) > n_words else pool)
        print(f"  [{lang}] {len(test_words)} words ({sample_mode}), {n_seeds} seeds...", flush=True)
        seed_res = [_eval_seed(checker, lang, test_words, s, min_word_length=min_word_length)
                    for s in range(n_seeds)]
        mf1s = [r["mf1"] for r in seed_res]
        # seed 0 word by word, and the confusion row computed from those same words
        w_rows, (wtp, wfp, wfn, wtn), wtau = audit_rows(
            checker.detection_models[lang], checker.threshold_classifiers.get(lang),
            test_words, 0, extra={"language": lang}, min_word_length=min_word_length)
        word_audit_rows.extend(w_rows)
        wm = class_metrics(wtp, wfp, wfn, wtn)
        _lb = lr_boundary(checker.threshold_classifiers.get(lang)) or {}
        confusion_rows.append({
            "language": lang, "n_words": len(test_words), "seed": 0,
            "lr_tau": ("" if wtau is None else round(wtau, 6)),
            "lr_intercept": round(_lb.get("intercept", float("nan")), 6),
            "lr_coef": round(_lb.get("coef", float("nan")), 6),
            "tp": wtp, "fp": wfp, "fn": wfn, "tn": wtn,
            "incorrect_precision": round(wm["incorrect_precision"], 6),
            "incorrect_recall": round(wm["incorrect_recall"], 6),
            "incorrect_f1": round(wm["incorrect_f1"], 6),
            "correct_precision": round(wm["correct_precision"], 6),
            "correct_recall": round(wm["correct_recall"], 6),
            "correct_f1": round(wm["correct_f1"], 6), "mf1_seed0": round(wm["mf1"], 6),
            "mf1_30seed_mean": round(float(np.mean(mf1s)), 6)})
        results[lang] = {
            "label":        _lab(lang),
            "n_test_words": len(test_words),
            "n_seeds":      n_seeds,
            "mf1":   {"mean": float(np.mean(mf1s)), "std": float(np.std(mf1s,ddof=1)),
                    "ci95": _ci95(mf1s), "min": float(np.min(mf1s)),
                    "max": float(np.max(mf1s))},
            "precision": {"mean": float(np.mean([r["precision"] for r in seed_res]))},
            "recall":    {"mean": float(np.mean([r["recall"]    for r in seed_res]))},
            "incorrect": {
                "precision": float(np.mean([r["precision"]  for r in seed_res])),
                "recall":    float(np.mean([r["recall"]     for r in seed_res])),
                "f1":        float(np.mean([r["f1_error"]   for r in seed_res]))},
            "correct": {
                "precision": float(np.mean([r["precision_correct"] for r in seed_res])),
                "recall":    float(np.mean([r["recall_correct"]    for r in seed_res])),
                "f1":        float(np.mean([r["f1_correct"]        for r in seed_res]))},
            "last_seed_confusion": {k: seed_res[-1][k] for k in ["tp","fp","fn","tn"]},
            "per_seed_mf1": mf1s,
        }
        per_seed_mf1[lang] = mf1s
        mlflow.log_metrics({f"mF1_mean_{lang}_{sample_mode}": results[lang]["mf1"]["mean"],
                            f"mF1_ci95_{lang}_{sample_mode}": results[lang]["mf1"]["ci95"]})

    # paired t-test between every two languages, over the seeds
    sig_matrix = {}
    keys = list(per_seed_mf1.keys())
    for i, la in enumerate(keys):
        for lb in keys[i+1:]:
            a, b = per_seed_mf1[la], per_seed_mf1[lb]
            if len(a) == len(b) >= 2:
                t_stat, p_val = stats.ttest_rel(a, b)
                sig_matrix[f"{la}_vs_{lb}"] = {
                    "t": round(float(t_stat),4), "p": round(float(p_val),4),
                    "significant_p05": bool(p_val < 0.05)}

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation_detection.json"
    json_path.write_text(json.dumps(
        {"protocol": {"corruption_rate":1.0,"min_word_length":min_word_length,
                    "sample_mode":sample_mode, "corpus": tag or "SADiLaR + NCHLT",
                    "n_seeds":n_seeds,"n_words_per_lang":n_words,"seed":42},
        "per_language": results, "significance": sig_matrix}, indent=2))
    mlflow.log_artifact(str(json_path), artifact_path=f"{sample_mode}-based")

    for name, rows in (("evaluation_detection_words_seed0.csv", word_audit_rows),
                       ("evaluation_detection_confusion.csv", confusion_rows)):
        csv_path = write_csv(output_dir / name, rows)
        if csv_path:
            mlflow.log_artifact(csv_path, artifact_path=f"{sample_mode}-based")

    # bar chart, SADiLaR on the left and NCHLT on the right
    ordered = ([l for l in SADILAR_ORDER if l in results] +
            [l for l in NCHLT_ORDER   if l in results])
    means  = [results[l]["mf1"]["mean"] for l in ordered]
    ci95s  = [results[l]["mf1"]["ci95"] for l in ordered]
    colours= [LANG_COLOURS.get(l,"#999") for l in ordered]
    xlabels= [_lab(l) for l in ordered]

    y_min = 0.0
    y_max = 1.0

    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    bars = ax.bar(range(len(ordered)), means, yerr=ci95s, capsize=5,
                color=colours, alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, val, ci in zip(bars, means, ci95s):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+ci+0.003,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    n_sad = sum(1 for l in SADILAR_ORDER if l in results)
    n_nch = sum(1 for l in NCHLT_ORDER   if l in results)
    if n_sad and n_nch:
        sep = n_sad - 0.5
        ax.axvline(sep, color="#333", linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title(f"Spell-checker detection mF1 ({n_seeds} seeds, "
                f"{n_words or 'all'} words/lang, 95% CI error bars)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    pdf_path, _png = write_figure(fig, output_dir, "evaluation_detection", bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(pdf_path), artifact_path=f"{sample_mode}-based")

    print(f"\n{'='*82}")
    print(f"  Spell Checker Detection  |  {n_seeds} seeds  |  "
        f"{'all' if not n_words else n_words} words/lang")
    print(f"{'='*82}")
    print(f"  {'language':<24}{'mF1':>8}{'+/-std':>7}{'95%CI':>8}"
        f"{'Prec':>8}{'Rec':>8}{'n':>7}")
    print(f"  {'-'*71}")
    for lang in ordered:
        r  = results[lang]
        print(f"  {r['label'].replace(chr(10),' '):<24}"
            f"{r['mf1']['mean']:>8.4f}{r['mf1']['std']:>7.4f}"
            f"{r['mf1']['ci95']:>8.4f}"
            f"{r['precision']['mean']:>8.4f}{r['recall']['mean']:>8.4f}"
            f"{r['n_test_words']:>7}")

    print(f"\n  Per-class detection metrics (mean over {n_seeds} seeds):")
    print(f"  {'':<24}{'Incorrect words':^24}{'Correct words':^24}{'':>9}")
    print(f"  {'language':<24}{'Prec':>8}{'Rec':>8}{'F1':>8}"
        f"{'Prec':>8}{'Rec':>8}{'F1':>8}{'AvgF1':>9}")
    print(f"  {'-'*81}")
    for lang in ordered:
        r = results[lang]
        inc, cor = r["incorrect"], r["correct"]
        print(f"  {r['label'].replace(chr(10),' '):<24}"
            f"{inc['precision']:>8.4f}{inc['recall']:>8.4f}{inc['f1']:>8.4f}"
            f"{cor['precision']:>8.4f}{cor['recall']:>8.4f}{cor['f1']:>8.4f}"
            f"{r['mf1']['mean']:>9.4f}")
    if sig_matrix:
        print(f"\n  Paired t-tests:")
        for pair, v in sig_matrix.items():
            s = "  [ok]" if v["significant_p05"] else ""
            print(f"    {pair:<32}  t={v['t']:>7.3f}  p={v['p']:>6.4f}{s}")
    print(f"\n  Confusion matrix (last seed):")
    for lang in ordered:
        cm = results[lang]["last_seed_confusion"]
        print(f"    {lang:<10}  TP={cm['tp']}  FP={cm['fp']}"
            f"  FN={cm['fn']}  TN={cm['tn']}")
    print(f"{'='*82}\n")


def run(lang_list, n_seeds, n_words, output_dir, modes=("type", "token")):
    """
    Evaluates the languages once per mode and writes each mode to its own subfolder. Short
    words are scored in both modes (min_word_length=1).
    """
    base = Path(output_dir)
    mlflow.set_experiment("eval_detection")
    with mlflow.start_run():
        mlflow.log_params({"n_seeds": n_seeds, "n_words": n_words or "all",
            "languages": lang_list, "modes": ",".join(modes)})
        for mode in modes:
            _run_mode(lang_list, n_seeds, n_words, base / f"{mode}-based",
                sample_mode=mode, min_word_length=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang",  nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--words", type=int, default=20000,
                    help="cap on test words per language; a smaller test split is used whole")
    ap.add_argument("--out",   type=Path, default=ROOT/"results"/"eval_detection")
    a = ap.parse_args()
    if a.lang:
        lang_list = a.lang
    else:
        c = SpellChecker()
        c.load_models(ROOT/"data"/"models")
        lang_list = sorted(c.detection_models.keys())
    run(lang_list, a.seeds, a.words, a.out)
