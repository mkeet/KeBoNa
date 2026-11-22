"""
compare_scoring_methods.py - compares seven ways of scoring a word for error detection.

Each score is computed here from the counts of the trained model, is given its own
logistic-regression threshold and is evaluated under the same balanced protocol over N seeds.

    trigram_conditional   mean log P(t_k | t_k-2, t_k-1), the score the checker uses
    bigram_conditional    mean log P(t_k | t_k-1)
    unigram_freq          mean log P(t_k), with -20 for an unseen trigram
    freq_mean_laplace     mean log P(t_k), with Laplace smoothing
    min_pool              the log-probability of the least likely trigram
    variance_penalty      mean minus twice the standard deviation of the trigram log-frequencies
    interp_backoff        0.7 * trigram + 0.2 * bigram + 0.1 * frequency

    python scripts/compare_scoring_methods.py --langs zulu xhosa --seeds 30

Outputs (results/scoring_methods/): scoring_methods.csv, scoring_methods.pdf
"""

import argparse
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
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


def _bigrams(model):
    """
    Returns the count of each trigram given the one before it, summed out of the trigram table
    of the model. The first START pad of every word is followed by the second, once per word.
    """
    if not hasattr(model, "_bigrams"):
        bi = defaultdict(Counter)
        for (_, prev), nxt in model.ngrams.items():
            bi[(prev,)].update(nxt)
        bi[(START_TOKEN,)][START_TOKEN] += sum(model.ngrams.get((START_TOKEN, START_TOKEN), {}).values())
        model._bigrams = bi
    return model._bigrams


def _ctx_totals(model):
    """Context denominators, summed once per model."""
    if not hasattr(model, "_ct_cache"):
        model._ct_cache = {c: sum(v.values()) for c, v in model.ngrams.items()}
        model._bt_cache = {c: sum(v.values()) for c, v in _bigrams(model).items()}
    return model._ct_cache, model._bt_cache


def s_trigram_conditional(model, word):
    tris = _TOK.segment_word(word)
    if not tris: return FLOOR
    ct, _ = _ctx_totals(model); padded = [START_TOKEN, START_TOKEN] + tris; tot = 0.0
    for i in range(2, len(padded)):
        tok = padded[i]; ctx = tuple(padded[i - 2:i]); d = ct.get(ctx, 0)
        p = (model.ngrams[ctx].get(tok, 0) / d) if d > 0 else 0.0
        tot += math.log(p) if p > 0 else FLOOR
    return tot / len(tris)


def s_bigram_conditional(model, word):
    tris = _TOK.segment_word(word)
    if not tris: return FLOOR
    _, bt = _ctx_totals(model); padded = [START_TOKEN] + tris; tot = 0.0
    for i in range(1, len(padded)):
        tok = padded[i]; ctx = (padded[i - 1],); d = bt.get(ctx, 0)
        p = (_bigrams(model)[ctx].get(tok, 0) / d) if d > 0 else 0.0
        tot += math.log(p) if p > 0 else FLOOR
    return tot / len(tris)


def s_unigram_freq(model, word):
    tris = _TOK.segment_word(word)
    if not tris: return FLOOR
    tot = 0.0
    for t in tris:
        c = model.counts.get(t, 0)
        tot += math.log(c / model.total_tokens) if c > 0 else FLOOR
    return tot / len(tris)


def s_freq_mean_laplace(model, word):
    tris = _TOK.segment_word(word)
    if not tris: return FLOOR
    V = len(model.vocab) + 1; a = 1.0
    return sum(math.log((model.counts.get(t, 0) + a) / (model.total_tokens + a * V))
               for t in tris) / len(tris)


def s_min_pool(model, word):
    tris = _TOK.segment_word(word)
    if not tris: return FLOOR
    V = len(model.vocab) + 1; a = 1e-5; mn = 0.0
    for t in tris:
        lp = math.log((model.counts.get(t, 0) + a) / (model.total_tokens + a * V))
        mn = min(mn, lp)
    return mn


def s_variance_penalty(model, word):
    tris = _TOK.segment_word(word)
    if not tris: return FLOOR
    V = len(model.vocab) + 1; a = 1e-5
    lps = [math.log((model.counts.get(t, 0) + a) / (model.total_tokens + a * V)) for t in tris]
    avg = sum(lps) / len(lps)
    if len(lps) < 2: return avg
    std = (sum((x - avg) ** 2 for x in lps) / len(lps)) ** 0.5
    return avg - 2 * std


def s_interp_backoff(model, word):
    tris = _TOK.segment_word(word)
    if not tris: return FLOOR
    ct, bt = _ctx_totals(model); V = len(model.vocab) + 1; a = 1e-5
    padded = [START_TOKEN, START_TOKEN] + tris; tot = 0.0
    for i in range(2, len(padded)):
        tok = padded[i]; ctx = tuple(padded[i - 2:i]); bctx = (padded[i - 1],)
        dt = ct.get(ctx, 0); ptri = (model.ngrams[ctx].get(tok, 0) / dt) if dt > 0 else 0.0
        db = bt.get(bctx, 0); pbi = (_bigrams(model)[bctx].get(tok, 0) / db) if db > 0 else 0.0
        pf = (model.counts.get(tok, 0) + a) / (model.total_tokens + a * V)
        p = 0.7 * ptri + 0.2 * pbi + 0.1 * pf
        tot += math.log(p) if p > 0 else FLOOR
    return tot / len(tris)


METHODS = [
    ("trigram_conditional", s_trigram_conditional, "CHOSEN"),
    ("bigram_conditional",  s_bigram_conditional,  ""),
    ("unigram_freq",        s_unigram_freq,        "freq MEAN, no Laplace"),
    ("freq_mean_laplace",   s_freq_mean_laplace,   "freq MEAN, Laplace"),
    ("min_pool",            s_min_pool,            "worst trigram"),
    ("variance_penalty",    s_variance_penalty,    "dspread"),
    ("interp_backoff",      s_interp_backoff,      "0.7/0.2/0.1"),
]


def _evaluate(model, score_fn, val_words, test_words, seeds):
    """
    Fits a threshold for one scoring method on the validation words and evaluates it over the
    seeds. Returns the mean and standard deviation of macro-F1 with the per-class F1, the
    boundary and the counts on clean words, or None when the fit fails. The clean words are
    scored once, because their outcome does not change with the seed.
    """
    pos, neg = [], []
    for w in val_words:
        if len(w) >= 3:
            s = score_fn(model, w)
            if s > -19.0: pos.append(s)
    wr = SpellingWrecker(corruption_rate=1.0, seed=42)
    ct, gt = wr.corrupt_text(" ".join(val_words)); cset = set(gt.keys())
    for w in TextUtils.tokenize(TextUtils.normalize(ct)):
        if w in cset and len(w) >= 3:
            s = score_fn(model, w)
            if s > -19.0: neg.append(s)
    if not pos or not neg:
        return None
    X = np.array([[s] for s in pos] + [[s] for s in neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X, y)
    coef = float(clf.coef_[0][0]); intc = float(clf.intercept_[0])

    def flag(w):
        return len(w) >= 3 and (coef * score_fn(model, w) + intc) <= 0.0  # the same decision as sklearn predict

    fp = sum(1 for w in test_words if flag(w)); tn = len(test_words) - fp
    f1s, f1e_l, f1c_l = [], [], []
    for s in range(seeds):
        wr2 = SpellingWrecker(corruption_rate=1.0, seed=s)
        corrupted = [wr2.corrupt_text(w, min_word_length=3)[0].strip() for w in test_words]
        tp = sum(1 for c in corrupted if flag(c)); fn = len(corrupted) - tp
        mf1, _pe, _re, f1e, f1c = confusion_to_mf1(tp, fp, fn, tn)
        f1s.append(mf1); f1e_l.append(f1e); f1c_l.append(f1c)
    m, sd = mean_std(f1s)
    return {"mf1": m, "std": sd, "f1_err": float(np.mean(f1e_l)),
            "f1_cor": float(np.mean(f1c_l)),
            "tau": (-intc / coef if coef else float("nan")), "fp": fp, "tn": tn}


def run(langs, seeds, test_size, val_size, label=""):
    """
    Evaluates every method on each language and writes the CSV and the chart. label picks the
    output subfolder, so that runs on different corpora do not overwrite each other.
    """
    out = ROOT / "results" / ("scoring_methods" + (f"_{label}" if label else ""))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    per_method = {name: [] for name, _, _ in METHODS}
    for lang in langs:
        model = TrigramModel.load(model_path(lang, "trigram"))
        # the regression needs only a few thousand validation words
        val = sample_unique(load_words(corpus_path(lang, "validation")), val_size, seed=42)
        test = sample_unique(load_words(corpus_path(lang, "test")), test_size, seed=42)
        print(f"\n[{lang}] {len(test)} test words, {len(val)} val words, {seeds} seeds")
        for name, fn, note in METHODS:
            r = _evaluate(model, fn, val, test, seeds)
            if r is None:
                print(f"  {name:<22} (LR train failed)"); continue
            per_method[name].append(r["mf1"])
            rows.append({"language": lang, "method": name, "note": note,
                         "mf1_mean": round(r["mf1"], 6), "mf1_std": round(r["std"], 6),
                         "f1_incorrect": round(r["f1_err"], 6), "f1_correct": round(r["f1_cor"], 6),
                         "tau": round(r["tau"], 4), "fp": r["fp"], "tn": r["tn"], "seeds": seeds})
            print(f"  {name:<22} mF1={r['mf1']:.4f} +/- {r['std']:.4f}  "
                  f"(incF1={r['f1_err']:.3f} corF1={r['f1_cor']:.3f}) {note}")

    with (out / "scoring_methods.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # mean macro-F1 per method over the languages
    summ = [(name, float(np.mean(per_method[name])), float(np.std(per_method[name])), note)
            for name, _, note in METHODS if per_method[name]]
    summ.sort(key=lambda x: -x[1])
    print(f"\n{'='*66}\n  SCORING METHODS - mean mF1 across {len(langs)} language(s), {seeds} seeds"
          f"\n{'='*66}")
    for name, m, sd, note in summ:
        star = " <== CHOSEN" if name == "trigram_conditional" else ""
        print(f"  {name:<22} {m:.4f} +/- {sd:.4f} {note}{star}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = [s[0] for s in summ]; means = [s[1] for s in summ]; errs = [s[2] for s in summ]
    colours = ["#2ca02c" if n == "trigram_conditional" else "#1f77b4" for n in names]
    ax.bar(range(len(names)), means, yerr=errs, capsize=4, color=colours, alpha=0.9)
    for i, (m, e) in enumerate(zip(means, errs)):
        ax.text(i, m + e + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Macro F1 (balanced protocol)", fontsize=11)
    ax.set_title(f"Scoring-method comparison ({len(langs)} langs, {seeds} seeds) - "
                 f"green = chosen trigram conditional", fontsize=11)
    ax.set_ylim(0, min(1.0, max(means) + max(errs) + 0.06)); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "scoring_methods.pdf", dpi=150); plt.close(fig)
    print(f"\n  wrote {out}/scoring_methods.csv + .pdf\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["zulu", "xhosa"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=6000,
                    help="validation words sampled to train each method's LR")
    ap.add_argument("--label", default="",
                    help="output subfolder suffix, e.g. 'nchlt' -> results/scoring_methods_nchlt/")
    a = ap.parse_args()
    run(a.langs, a.seeds, a.test_size, a.val_size, a.label)
