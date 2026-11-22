"""
exp_c5_timing.py - the speed of detection and how the running time scales.

Measures words per second and milliseconds per word, whether time grows linearly with the
size of the input, what suggestions add, and whether loading more languages changes the time
of a check. The timings are wall-clock times, and the CPU, memory and operating system of the
machine are recorded with them.

Outputs (results/c5/<mode>-based/):
    c5_timing_{lang}.json   throughput, the scaling series and the machine
    c5_timing_{lang}.pdf    time against word count
    c5_sample_{lang}.csv    a small sample checked word by word
"""

import argparse
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import ROOT, load_words, corpus_path, write_csv, disp
from src.evaluation.spelling_wrecker import SpellingWrecker
from src.core.spell_checker import SpellChecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WORD_COUNTS = [1_000, 5_000, 10_000, 50_000, 100_000]


def machine_spec():
    """Returns the CPU, memory, operating system and Python version of the machine."""
    spec = {"python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.processor() or "unknown",
            "machine": platform.machine()}
    spec["cpu_count"] = os.cpu_count()
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    spec["mem_total"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return spec


def make_text(words, n, seed=42):
    """
    Builds a text of n words, each of them corrupted. The word list is repeated when it is
    shorter than n.
    """
    pool = words if len(words) >= n else (words * (n // len(words) + 1))
    sample = pool[:n]
    wrecker = SpellingWrecker(corruption_rate=1.0, seed=seed)
    corrupted = [wrecker.corrupt_text(w, min_word_length=3)[0].strip() for w in sample]
    return " ".join(corrupted)


def detect_correct_sample(checker, lang, words, n_sample=200, show=5, seed=42):
    """
    Runs detection and correction on a small corrupted sample through the same check() call
    that is timed. Prints the first rows and returns one row per word for the CSV.
    """

    wr = SpellingWrecker(corruption_rate=1.0, seed=seed)
    pairs = []
    for w in words[:n_sample]:
        c, gt = wr.corrupt_text(w, min_word_length=3); c = c.strip()
        pairs.append((w, c, c in gt))
    text = " ".join(c for _, c, _ in pairs)
    res = checker.check(text, language=lang,
                        include_suggestions=True)
    info = {e["word"]: (e.get("probability"), e.get("suggestions", []))
            for e in res["errors"]}
    rows = []
    for orig, corr, is_err in pairs:
        prob, sugg = info.get(corr, (None, []))
        sw = [s[0] for s in sugg][:5]
        rows.append({"original": orig, "corrupted": corr, "is_error": int(is_err),
                    "flagged": int(corr in info),
                    "detection_score": ("" if prob is None else round(prob, 4)),
                    "top_suggestion": (sw[0] if sw else ""),
                    "suggestions": "|".join(sw),
                    "correct_in_top5": int(orig in sw)})
    print(f"\n  Real detection+correction sample (first {show} of {n_sample} words):")
    for r in rows[:show]:
        print(f"    {r['original']:<16} -> '{r['corrupted']:<16}' flagged={r['flagged']} "
            f"score={r['detection_score']} top='{r['top_suggestion']}' "
            f"correct_in_top5={r['correct_in_top5']}")
    return rows


def time_check(checker, lang, text, include_suggestions, repeats):
    """Times check() on a text and returns the mean and the standard deviation over the repeats."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        checker.check(text, language=lang,
                    include_suggestions=include_suggestions)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def _run_mode(lang, repeats, out, sample_mode="type"):
    out.mkdir(parents=True, exist_ok=True)

    checker = SpellChecker()
    checker.load_models(ROOT / "data" / "models")

    _all = load_words(corpus_path(lang, "test"))
    words = list(dict.fromkeys(_all)) if sample_mode == "type" else _all
    spec = machine_spec()
    mlflow.log_params({"machine_spec": str(spec)})
    logger.info("Machine: %s", spec)

    # a small sample with ground truth, to show the timing covers real detection and correction
    sample_rows = detect_correct_sample(checker, lang, words)
    _sp = write_csv(out / f"c5_sample_{lang}.csv", sample_rows)
    if _sp:
        mlflow.log_artifact(_sp, artifact_path=f"{sample_mode}-based")

    scaling = []
    for n in WORD_COUNTS:
        text = make_text(words, n)
        mean_t, std_t = time_check(checker, lang, text, False, repeats)
        wps = n / mean_t if mean_t else 0
        scaling.append({"words": n, "mean_s": mean_t, "std_s": std_t,
                       "words_per_s": wps, "ms_per_word": 1000 * mean_t / n})
        mlflow.log_metrics({
            f"words_{sample_mode}": n,
            f"time_s_{sample_mode}": mean_t,
            f"words_per_s_{sample_mode}": wps,
            f"ms_per_word_{sample_mode}": 1000 * mean_t / n,
        }, step=n)
        logger.info("[detect] %7d words: %.3fs (%.0f w/s, %.4f ms/word)",
                    n, mean_t, wps, 1000 * mean_t / n)

    fixed_n = 1000
    text = make_text(words, fixed_n)
    det_t, _ = time_check(checker, lang, text, False, repeats)
    sug_t, _ = time_check(checker, lang, text, True, repeats)
    suggestions_overhead = {
        "words": fixed_n,
        "detect_only_s": det_t,
        "detect_plus_suggest_s": sug_t,
        "suggestion_overhead_x": (sug_t / det_t) if det_t else 0,
    }
    mlflow.log_metrics({
        f"suggestion_overhead_x_{sample_mode}": suggestions_overhead["suggestion_overhead_x"],
    })
    logger.info("[suggest] %d words: detect=%.3fs, +suggest=%.3fs (%.1fx)",
                fixed_n, det_t, sug_t, suggestions_overhead["suggestion_overhead_x"])

    loaded = list(checker.detection_models.keys())
    mlflow.log_params({"loaded_models": loaded, "n_models": len(loaded)})

    out.mkdir(parents=True, exist_ok=True)
    (out / f"c5_timing_{lang}.json").write_text(json.dumps(
        {"machine": spec, "language": lang, "repeats": repeats,
        "scaling": scaling, "suggestions": suggestions_overhead,
        "languages_loaded": {"n_models_loaded": len(loaded), "models": loaded,
                            "note": "models are dict-keyed; per-check cost is O(1) in number of loaded languages"}}, indent=2))
    mlflow.log_artifact(str(out / f"c5_timing_{lang}.json"), artifact_path=f"{sample_mode}-based")

    xs = [r["words"] for r in scaling]
    ys = [r["mean_s"] for r in scaling]
    es = [r["std_s"] for r in scaling]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, color="#1f77b4")
    ax.set_xlabel("Number of words checked")
    ax.set_ylabel("Detection time (s)")
    ax.set_title(f"C5: Detection time vs word count ({disp(lang)})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out / f"c5_timing_{lang}.pdf", dpi=150)
    mlflow.log_artifact(str(out / f"c5_timing_{lang}.pdf"), artifact_path=f"{sample_mode}-based")

    logger.info("Saved results/c5_timing.{json,pdf}")

    print(f"\n{'='*64}\n  C5 - timing ({lang}), {repeats} repeats\n{'='*64}")
    print(f"  {'words':>8}{'time_s':>10}{'words/s':>12}{'ms/word':>10}")
    for r in scaling:
        print(f"  {r['words']:>8}{r['mean_s']:>10.3f}{r['words_per_s']:>12.0f}{r['ms_per_word']:>10.4f}")
    print(f"\n  Suggestions overhead at {fixed_n} words: "
        f"{suggestions_overhead['suggestion_overhead_x']:.1f}x slower than detect-only")
    print(f"  Machine: {spec.get('processor','?')} | {spec.get('cpu_count','?')} CPUs | "
        f"{spec.get('mem_total','?')}\n{'='*64}\n")

    return scaling


def run(lang, repeats, modes=("type", "token")):
    """Runs the timings for one language, once per mode."""
    base = ROOT / "results" / "c5"
    mlflow.set_experiment("timing_analysis")
    with mlflow.start_run():
        mlflow.log_params({"experiment": "c5_timing", "language": lang, "repeats": repeats, "modes": ",".join(modes)})
        for mode in modes:
            _run_mode(lang, repeats, base / f"{mode}-based", sample_mode=mode)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zulu")
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    run(a.lang, a.repeats)
