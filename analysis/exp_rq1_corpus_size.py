"""
exp_rq1_corpus_size.py - detection macro-F1 against the size of the training corpus (RQ1).

A fresh detection model and threshold are trained at each corpus size and evaluated on a test
sample of the same size for every language, so the curves of different languages can be compared.

Outputs (results/rq1/<mode>-based/): rq1_corpus_size.json, .csv, .pdf and one chart per corpus
source
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import (ROOT, NGUNI, LANG_LABELS, LANG_COLOURS, load_words, corpus_path, sample_unique,
                        sample_tokens, train_detection_model, train_lr, evaluate_balanced_model,
                        mean_std, write_csv)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIZES = [2_000, 5_000, 10_000, 15_000, 20_000, 30_000, 40_000,
        60_000, 100_000, 150_000, 200_000, 300_000]


def saturation(x, a, b, c):
    """The curve fitted to each language, which rises towards the ceiling a."""
    return a - b * np.exp(-c * x)


def _run_mode(langs, seeds, test_size, out, sample_mode="type", min_word_length=3):
    results = {}
    for lang in langs:
        tr = corpus_path(lang, "train")
        va = corpus_path(lang, "validation")
        te = corpus_path(lang, "test")
        if not (tr.exists() and va.exists() and te.exists()):
            logger.warning("[%s] missing corpora, skipping", lang); continue
        all_train = load_words(tr)
        val_words = load_words(va)
        _sample = sample_tokens if sample_mode == "token" else sample_unique
        test_words = _sample(load_words(te), test_size, seed=42)
        logger.info("[%s] train pool %d tokens (%d unique) | test %d words",
                    lang, len(all_train), len(set(all_train)), len(test_words))
        rows = []
        for size in SIZES:
            if size > len(all_train):
                logger.info("[%s] stops at %d tokens: training split has %d",
                            lang, rows[-1]["train_tokens"] if rows else 0, len(all_train))
                break
            sample = sample_tokens(all_train, size, seed=42); actual = len(set(sample))
            model = train_detection_model(sample, lang)
            clf   = train_lr(model, val_words)
            if clf is None:
                logger.warning("[%s] size=%d LR failed", lang, size); continue
            runs = [evaluate_balanced_model(model, clf, test_words, s,
                                            min_word_length=min_word_length)
                    for s in range(seeds)]
            m, sd = mean_std([r["mf1"] for r in runs])
            def mn(k): return float(np.mean([r[k] for r in runs]))
            logger.info("[%s] %d unique (%d tokens) -> mF1=%.4f +/- %.4f",
                        lang, actual, len(sample), m, sd)
            rows.append({"unique_words": actual, "train_tokens": len(sample),
                        "mean": m, "std": sd,
                        "tp": mn("tp"), "fp": mn("fp"), "fn": mn("fn"), "tn": mn("tn"),
                        "incorrect_f1": mn("f1_error"), "correct_f1": mn("f1_correct")})
        results[lang] = rows

    out.mkdir(parents=True, exist_ok=True)
    (out / "rq1_corpus_size.json").write_text(json.dumps(results, indent=2))
    mlflow.log_artifact(str(out / "rq1_corpus_size.json"), artifact_path=f"{sample_mode}-based")

    csv_rows = []
    for lang, rows in results.items():
        for r in rows:
            csv_rows.append({"language": lang, "unique_words": r["unique_words"],
                            "train_tokens": r["train_tokens"],
                            "tokens_per_unique": round(r["train_tokens"] / r["unique_words"], 3),
                            "tp": round(r["tp"], 1), "fp": round(r["fp"], 1),
                            "fn": round(r["fn"], 1), "tn": round(r["tn"], 1),
                            "incorrect_f1": round(r["incorrect_f1"], 6),
                            "correct_f1": round(r["correct_f1"], 6),
                            "mf1_mean": round(r["mean"], 6), "mf1_std": round(r["std"], 6)})
    _c = write_csv(out / "rq1_corpus_size.csv", csv_rows)
    if _c:
        mlflow.log_artifact(_c, artifact_path=f"{sample_mode}-based")

    y_min = 0.0
    y_max = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    for lang, rows in results.items():
        if not rows:
            continue
        xs  = np.array([r["train_tokens"] for r in rows])
        ys  = np.array([r["mean"] for r in rows])
        es  = np.array([r["std"]  for r in rows])
        col = LANG_COLOURS.get(lang, "#999")
        lbl = LANG_LABELS.get(lang, lang)

        ax.errorbar(xs, ys, yerr=es, fmt="o", color=col, capsize=3,
                    markersize=6, alpha=0.85)
        if len(xs) >= 4:
            try:
                p, _ = curve_fit(saturation, xs, ys,
                                p0=[ys.max(), ys.max()-ys.min(), 1e-4],
                                maxfev=10000)
                gx = np.linspace(xs.min(), xs.max(), 300)
                ax.plot(gx, saturation(gx, *p), color=col,
                        linewidth=2, label=lbl)
            except Exception:
                ax.plot(xs, ys, color=col, linewidth=2, label=lbl)
        else:
            ax.plot(xs, ys, color=col, linewidth=2, label=lbl)

    ax.set_xlabel("Training tokens", fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("RQ1: Detection accuracy vs corpus size\n"
                f"({test_size} equal-size test words/lang, {seeds} seeds)", fontsize=13)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "rq1_corpus_size.pdf", dpi=150)
    mlflow.log_artifact(str(out / "rq1_corpus_size.pdf"), artifact_path=f"{sample_mode}-based")
    logger.info("Saved results/rq1_corpus_size.{json,pdf}")

    # one chart per corpus source, so the curves stay legible
    FAMILIES = {"nchlt": (["zu", "xh", "nd", "sw"], "NCHLT corpora"),
                "sadilar": (["zulu", "xhosa", "ndebele", "swati"], "SADiLaR morphological corpora")}
    for tag, (fam, label) in FAMILIES.items():
        present = [l for l in fam if results.get(l)]
        if not present:
            continue
        f2, a2 = plt.subplots(figsize=(9.5, 5.4))
        for lang in present:
            rows = results[lang]
            xs = np.array([r["train_tokens"] for r in rows])
            ys = np.array([r["mean"] for r in rows])
            es = np.array([r["std"] for r in rows])
            a2.errorbar(xs, ys, yerr=es, fmt="o-", color=LANG_COLOURS.get(lang, "#999"),
                        capsize=3, markersize=5, linewidth=2, alpha=0.9,
                        label=LANG_LABELS.get(lang, lang))
        a2.axvspan(10000, 15000, color="#999", alpha=0.15)
        a2.set_xlabel("Training tokens", fontsize=12)
        a2.set_ylabel("Macro F1", fontsize=12)
        a2.set_ylim(0, 1.0)
        a2.set_title(f"RQ1: detection accuracy vs corpus size - {label}\n"
                    f"({sample_mode}-level, {test_size} test words per language, {seeds} seeds)",
                    fontsize=12, fontweight="bold")
        a2.legend(loc="lower right", fontsize=10)
        a2.grid(True, alpha=0.3)
        f2.tight_layout()
        p2 = out / f"rq1_corpus_size_{tag}.pdf"
        f2.savefig(p2, dpi=150); f2.savefig(p2.with_suffix(".png"), dpi=150)
        plt.close(f2)
        mlflow.log_artifact(str(p2), artifact_path=f"{sample_mode}-based")

    print(f"\n{'='*70}\n  RQ1 - mF1 vs training tokens (test {test_size} words/lang)\n{'='*70}")
    print("  " + f"{'lang':<22}" + "".join(f"{s//1000}k".rjust(8) for s in SIZES))
    for lang, rows in results.items():
        by = {r["train_tokens"]: r["mean"] for r in rows}
        cells = []
        for s in SIZES:
            v = next((val for k, val in by.items() if abs(k-s) < 600), None)
            cells.append(f"{v:.3f}".rjust(8) if v else "-".rjust(8))
        print("  " + f"{LANG_LABELS.get(lang,lang):<22}" + "".join(cells))
    print(f"{'='*70}\n")


def run(langs, seeds, test_size, modes=("type", "token")):
    """Runs the experiment once per mode. Short words are scored (min_word_length=1)."""
    base = ROOT / "results" / "rq1"
    mlflow.set_experiment("RQ1_Corpus_Size")
    with mlflow.start_run():
        mlflow.log_params({"experiment": "rq1_corpus_size", "languages": langs, "seeds": seeds,
                        "test_size": test_size, "modes": ",".join(modes)})
        for mode in modes:
            _run_mode(langs, seeds, test_size, base / f"{mode}-based",
                    sample_mode=mode, min_word_length=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang",      nargs="+", default=NGUNI)
    ap.add_argument("--seeds",     type=int,  default=30)
    ap.add_argument("--test-size", type=int,  default=4000)
    a = ap.parse_args()
    run(a.lang, a.seeds, a.test_size)
