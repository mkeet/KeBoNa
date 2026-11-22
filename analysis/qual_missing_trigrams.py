"""
qual_missing_trigrams.py - which unseen trigram transitions lie behind the detection errors.

An unseen transition contributes the penalty of -20 to a score. This script reads the
word-by-word CSV of seed 0 that exp_c8_fp_fn_analysis.py writes, and nothing is run again.

Outputs (results/c8/<mode>-based/):
    qual_missing_trigrams_{lang}.csv          one row per penalised trigram, counted by outcome
    qual_missing_trigrams_summary_{lang}.csv  per outcome, the instances and the share that
                                              have a penalised step
"""

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import write_csv
from exp_common import load_words, corpus_path, model_path, sample_tokens, _trigram_steps
from src.models.trigram_model import TrigramModel
from src.models.threshold_classifier import ThresholdClassifier

csv.field_size_limit(sys.maxsize)
OUTCOMES = ["tp", "fp", "fn", "tn"]


def floored_steps(calc):
    """
    Returns the context, the trigram and whether the context was unseen, for every penalised step of a word.
    """
    out = []
    for step in calc.split(" ; "):
        if not step.endswith("(floor)"):
            continue
        head, frac = step.split("=", 1)
        ctx, tri = head.rsplit(">", 1)
        denom = frac.split("=", 1)[0].split("/")[1]
        out.append((ctx, tri, denom == "0"))
    return out


def run(lang="zu", mode="token", top=25):
    """Counts the penalised trigrams of one language by outcome and writes the two CSVs."""
    d = ROOT / "results" / "c8" / f"{mode}-based"
    src = d / f"c8_words_seed0_{lang}.csv"
    by_tri = {o: Counter() for o in OUTCOMES}
    words = {o: {} for o in OUTCOMES}
    n = Counter(); n_floor = Counter(); n_ctx = Counter()
    with open(src, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            oc = r["outcome"]
            if oc not in by_tri:
                continue
            n[oc] += 1
            steps = floored_steps(r["trigram_calc"])
            if steps:
                n_floor[oc] += 1
            if any(u for _, _, u in steps):
                n_ctx[oc] += 1
            w = r["corrupted"] or r["original"]
            for _, tri, _ in steps:
                by_tri[oc][tri] += 1
                words[oc].setdefault(tri, w)

    summary = [{"outcome": o, "instances": n[o], "with_floored_step": n_floor[o],
                "share_floored": round(n_floor[o] / n[o], 4) if n[o] else 0.0,
                "with_unseen_context": n_ctx[o]} for o in OUTCOMES]
    tris = set().union(*by_tri.values())
    rows = sorted(({"trigram": t, **{f"in_{o}": by_tri[o][t] for o in OUTCOMES},
                    "example_fp": words["fp"].get(t, ""), "example_tp": words["tp"].get(t, "")}
                   for t in tris), key=lambda r: -r["in_fp"])
    write_csv(d / f"qual_missing_trigrams_summary_{lang}.csv", summary)
    write_csv(d / f"qual_missing_trigrams_{lang}.csv", rows)

    print(f"\n  Missing trigrams ({lang}, {mode}-level, seed 0)")
    for s in summary:
        print(f"  {s['outcome']}: {s['instances']:>9,} instances, {s['share_floored']:.1%} with a floored step")
    print(f"\n  {'trigram':<10}{'in fp':>8}{'in tp':>9}  example fp")
    for r in rows[:top]:
        print(f"  {r['trigram']:<10}{r['in_fp']:>8,}{r['in_tp']:>9,}  {r['example_fp']}")
    return summary, rows


def run_pair(source, target, n=4000, top=15):
    """
    Applies the model of a source language to a sample of the test tokens of a target language,
    to show why valid words of the target are flagged.
    """
    det = TrigramModel.load(model_path(source, "trigram"))
    clf = ThresholdClassifier.load(model_path(source, "classifier"))
    words = sample_tokens(load_words(corpus_path(target, "test")), n, seed=42)
    flagged = short = with_floor = 0
    tri_count, tri_word = Counter(), {}
    for w in words:
        tris, steps = _trigram_steps(det, w)
        score = sum(s["logp"] for s in steps) / len(tris) if tris else -20.0
        if not clf.predict_error(score):
            continue
        flagged += 1
        if not tris:
            short += 1
            continue
        fl = [s["token"] for s in steps if s["prob"] == 0.0]
        if fl:
            with_floor += 1
        for t in fl:
            tri_count[t] += 1
            tri_word.setdefault(t, w)
    summary = {"source": source, "target": target, "tokens": len(words), "flagged": flagged,
               "flagged_short": short, "flagged_with_floored_step": with_floor,
               "flagged_other": flagged - short - with_floor}
    rows = [{"trigram": t, "count": c, "example": tri_word[t]} for t, c in tri_count.most_common()]
    d = ROOT / "results" / "rq2" / "token-based"
    write_csv(d / f"qual_missing_trigrams_summary_{source}_{target}.csv", [summary])
    write_csv(d / f"qual_missing_trigrams_{source}_{target}.csv", rows)

    print(f"\n  {source} model on {target} ({len(words)} test tokens, seed 42): {flagged} valid tokens flagged")
    print(f"    under 3 characters: {short}   with an unseen transition: {with_floor}   "
          f"neither: {flagged - short - with_floor}")
    for r in rows[:top]:
        print(f"    {r['trigram']:<8}{r['count']:>6}  {r['example']}")
    return summary, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zu")
    ap.add_argument("--mode", default="token", choices=["token", "type"])
    ap.add_argument("--pair", nargs=2, metavar=("SOURCE", "TARGET"), default=None)
    a = ap.parse_args()
    if a.pair:
        run_pair(*a.pair)
    else:
        run(a.lang, a.mode)
