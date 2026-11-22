"""
compare_min_word_len.py - the effect of skipping words of fewer than three characters.

The effect is measured on languages written conjunctively and on the Sotho-Tswana languages,
which are written disjunctively and have many short words. Each language is run four ways
against its trained model.

    sampling   types    distinct words, each counted once
               tokens   running text, counted per occurrence
    filter     on       words of fewer than three characters are skipped
               off      every word is scored, and a short word is flagged

Sampling types hides the effect. In Setswana about 32% of tokens have fewer than three
characters but only about 0.6% of types do.

    python scripts/compare_min_word_len.py --langs zu xh nd sw sotho tswana --seeds 30

Outputs (results/min_word_len/): min_word_len.csv, min_word_len.pdf
"""

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import random
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from src.models.trigram_model import TrigramModel, START_TOKEN
from src.utils.trigram_tokenizer import CharTrigramTokenizer
from src.utils.text_utils import TextUtils
from src.evaluation.spelling_wrecker import SpellingWrecker
from exp_common import (load_words, corpus_path, model_path, sample_unique,
                        confusion_to_mf1, mean_std)

_TOK = CharTrigramTokenizer(n=3)
FLOOR = -20.0
DISJUNCTIVE = {"sotho", "tswana", "lozi", "silozi"}


def _ctx_totals(model):
    """Context denominators, summed once per model."""
    if not hasattr(model, "_ct_cache"):
        model._ct_cache = {c: sum(v.values()) for c, v in model.ngrams.items()}
    return model._ct_cache


def score_word(model, word):
    """
    The score used by the checker, which is the mean log-probability of each trigram given the
    two before it, with -20 for an unseen transition. A word with no trigram scores -20.
    """
    tris = _TOK.segment_word(word)
    if not tris:
        return FLOOR
    ct = _ctx_totals(model); padded = [START_TOKEN, START_TOKEN] + tris; tot = 0.0
    for i in range(2, len(padded)):
        tok = padded[i]; ctx = tuple(padded[i - 2:i]); d = ct.get(ctx, 0)
        p = (model.ngrams[ctx].get(tok, 0) / d) if d > 0 else 0.0
        tot += math.log(p) if p > 0 else FLOOR
    return tot / len(tris)


def _make_flag(model, val_words):
    """
    Fits the threshold on validation words of at least three characters and on their corrupted
    forms. Returns a function that says whether a word is flagged, or None when either class
    is empty.
    """
    pos, neg = [], []
    for w in val_words:
        if len(w) >= 3:
            s = score_word(model, w)
            if s > -19.0: pos.append(s)
    wr = SpellingWrecker(corruption_rate=1.0, seed=42)
    ct, gt = wr.corrupt_text(" ".join(val_words)); cset = set(gt.keys())
    for w in TextUtils.tokenize(TextUtils.normalize(ct)):
        if w in cset and len(w) >= 3:
            s = score_word(model, w)
            if s > -19.0: neg.append(s)
    if not pos or not neg:
        return None
    X = np.array([[s] for s in pos] + [[s] for s in neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X, y)
    coef = float(clf.coef_[0][0]); intc = float(clf.intercept_[0])
    cache = {}

    def flag(w):
        s = cache.get(w)
        if s is None:
            s = score_word(model, w); cache[w] = s
        return (coef * s + intc) <= 0.0  # the same decision as sklearn predict
    flag.tau = (-intc / coef) if coef else float("nan")
    return flag


def _evaluate(flag, population, seeds, min_len):
    """
    Evaluates under the balanced protocol, counting every occurrence in a word list that may
    hold duplicates. Words shorter than min_len are skipped.
    """
    if not population:
        return None
    fp = sum(1 for w in population if flag(w)); tn = len(population) - fp
    f1s = []
    for s in range(seeds):
        wr = SpellingWrecker(corruption_rate=1.0, seed=s)
        corrupted = [wr.corrupt_text(w, min_word_length=min_len)[0].strip() for w in population]
        tp = sum(1 for c in corrupted if flag(c)); fn = len(corrupted) - tp
        mf1, *_ = confusion_to_mf1(tp, fp, fn, tn); f1s.append(mf1)
    m, sd = mean_std(f1s)
    return {"mf1": m, "std": sd, "fp": fp, "tn": tn, "n": len(population)}


def run(langs, seeds, test_size, val_size):
    """Runs the four settings for each language and writes the CSV and the chart."""
    out = ROOT / "results" / "min_word_len"; out.mkdir(parents=True, exist_ok=True)
    rows = []; audit = []
    for lang in langs:
        model = TrigramModel.load(model_path(lang, "trigram"))
        val = sample_unique(load_words(corpus_path(lang, "validation")), val_size, seed=42)
        raw = load_words(corpus_path(lang, "test"))
        uniq = sample_unique(raw, test_size, seed=42)
        toks = random.Random(42).sample(raw, min(test_size, len(raw)))
        pct_short_types = 100.0 * sum(1 for w in uniq if len(w) < 3) / max(1, len(uniq))
        pct_short_toks = 100.0 * sum(1 for w in toks if len(w) < 3) / max(1, len(toks))
        print(f"\n[{lang}] short<3: {pct_short_types:.2f}% of TYPES vs {pct_short_toks:.2f}% of "
              f"TOKENS  (unique={len(set(raw))}, tokens={len(raw)})", flush=True)

        flag = _make_flag(model, val)
        if flag is None:
            print(f"  LR train failed - skipped"); continue

        conds = {
            ("dedup", "filter"):    (uniq, 3),
            ("dedup", "nofilter"):  (uniq, 1),
            ("dups",  "filter"):    (toks, 3),
            ("dups",  "nofilter"):  (toks, 1),
        }
        res = {}
        for (samp, filt), (words, _ml) in conds.items():
            pop = [w for w in words if len(w) >= (3 if filt == "filter" else 1)]
            r = _evaluate(flag, pop, seeds, min_len=(3 if filt == "filter" else 1))
            res[(samp, filt)] = r
            rows.append({"language": lang, "disjunctive": lang in DISJUNCTIVE,
                         "sampling": samp, "filter": filt,
                         "pct_short_types": round(pct_short_types, 3),
                         "pct_short_tokens": round(pct_short_toks, 3),
                         "n_words": r["n"], "fp": r["fp"], "tn": r["tn"],
                         "mf1_mean": round(r["mf1"], 6), "mf1_std": round(r["std"], 6),
                         "seeds": seeds})
            print(f"  {samp:<5} {filt:<9} n={r['n']:<6} mF1={r['mf1']:.4f} +/- {r['std']:.4f}"
                  f"  (FP={r['fp']}, TN={r['tn']})")
        d = res[("dups", "filter")]["mf1"] - res[("dups", "nofilter")]["mf1"]
        print(f"  -> DUPS filter effect (real text): removing filter changes mF1 by {-d:+.4f}")

        # short words are left out with the filter on and are false positives with it off
        tau = getattr(flag, "tau", float("nan"))
        for w, c in Counter(x for x in raw if len(x) < 3).most_common():
            s = score_word(model, w); fl = flag(w)
            audit.append({"language": lang, "word": w, "length": len(w),
                          "occurrences_in_test": c, "detection_score": round(s, 4),
                          "tau_threshold": round(tau, 4),
                          "filter": "on", "scored": "no", "flagged": "n/a",
                          "outcome": "excluded",
                          "note": "not evaluated by this script (evaluate_detection would "
                          "count it as TN by omission)"})
            audit.append({"language": lang, "word": w, "length": len(w),
                          "occurrences_in_test": c, "detection_score": round(s, 4),
                          "tau_threshold": round(tau, 4),
                          "filter": "off", "scored": "yes", "flagged": int(fl),
                          "outcome": ("FP" if fl else "TN"),
                          "note": ("valid word floored below tau -> flagged as error"
                                   if fl else "not flagged")})

    if not rows:
        print("no languages evaluated"); return

    with (out / "min_word_len.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if audit:
        with (out / "short_word_audit.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
            w.writeheader(); w.writerows(audit)
        print(f"  wrote {out}/short_word_audit.csv ({len(audit)} rows)")

    langs_ord = sorted({r["language"] for r in rows},
                       key=lambda L: (any(r["language"] == L and r["disjunctive"] for r in rows), L))
    fig, axes = plt.subplots(1, 2, figsize=(2.0 * len(langs_ord) + 4, 5.5), sharey=True)
    for ax, samp, subtitle in [(axes[0], "dedup", "DEDUP (unique words / types)"),
                               (axes[1], "dups", "DUPS (all occurrences / real text)")]:
        x = np.arange(len(langs_ord)); wbar = 0.38
        def val(L, filt): return next(r["mf1_mean"] for r in rows
                                      if r["language"] == L and r["sampling"] == samp and r["filter"] == filt)
        def err(L, filt): return next(r["mf1_std"] for r in rows
                                      if r["language"] == L and r["sampling"] == samp and r["filter"] == filt)
        fm = [val(L, "filter") for L in langs_ord]; fe = [err(L, "filter") for L in langs_ord]
        um = [val(L, "nofilter") for L in langs_ord]; ue = [err(L, "nofilter") for L in langs_ord]
        ax.bar(x - wbar / 2, fm, wbar, yerr=fe, capsize=3, color="#2ca02c", alpha=0.9,
               label="filter ON (words >=3)")
        ax.bar(x + wbar / 2, um, wbar, yerr=ue, capsize=3, color="#d62728", alpha=0.9,
               label="filter OFF (all words)")
        for i, L in enumerate(langs_ord):
            ax.text(i - wbar / 2, fm[i] + 0.01, f"{fm[i]:.2f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + wbar / 2, um[i] + 0.01, f"{um[i]:.2f}", ha="center", va="bottom", fontsize=7)
        n_agg = sum(1 for L in langs_ord
                    if not any(r["language"] == L and r["disjunctive"] for r in rows))
        if 0 < n_agg < len(langs_ord):
            ax.axvline(n_agg - 0.5, color="#333", linewidth=1.1, linestyle="--", alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(langs_ord, fontsize=10)
        ax.set_title(subtitle, fontsize=11); ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Macro F1 (balanced protocol)", fontsize=11)
    fig.suptitle(f"<3-char filter effect ({len(langs_ord)} langs, {seeds} seeds) - "
                 f"agglutinative | disjunctive", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "min_word_len.pdf", dpi=150); plt.close(fig)

    print(f"\n{'='*86}\n  <3-CHAR FILTER: mF1 by sampling x filter, {seeds} seeds\n{'='*86}")
    print(f"  {'lang':<9}{'%short_tok':>11}{'dedup ON':>11}{'dedup OFF':>11}"
          f"{'dups ON':>11}{'dups OFF':>11}")
    for L in langs_ord:
        g = lambda s, f: next(r["mf1_mean"] for r in rows
                              if r["language"] == L and r["sampling"] == s and r["filter"] == f)
        pst = next(r["pct_short_tokens"] for r in rows if r["language"] == L)
        print(f"  {L:<9}{pst:>10.2f}%{g('dedup','filter'):>11.4f}{g('dedup','nofilter'):>11.4f}"
              f"{g('dups','filter'):>11.4f}{g('dups','nofilter'):>11.4f}")
    print(f"\n  wrote {out}/min_word_len.csv + .pdf\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["zu", "xh", "nd", "sw", "sotho", "tswana"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=6000)
    a = ap.parse_args()
    run(a.langs, a.seeds, a.test_size, a.val_size)
