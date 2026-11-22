"""
qual_rq4_corrections.py - sorts the corrections of seed 0 into the ways they succeed or fail.

Reads results/rq4/rq4_corrections_seed0_{lang}.csv, which exp_rq4_correction.py writes.
"""

import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import mlflow
from rapidfuzz.distance import OSA as Levenshtein

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import write_figure, disp
from src.models.trigram_model import TrigramModel

RQ4_DIR = ROOT / "results" / "rq4"
PER_CATEGORY = 8
MAX_EDIT = 2

CATEGORIES = [
    ("success_rank1",      "Correct: intended ranked 1",        "#2ca02c"),
    ("near_miss",          "Near miss: intended ranked 2-5",    "#98c379"),
    ("miss_outranked",     "Missed: in reach, outranked",       "#1f77b4"),
    ("miss_too_far",       "Missed: beyond edit distance 2",    "#ff7f0e"),
    ("miss_not_in_vocab",  "Missed: not in training vocabulary","#d62728"),
]


def _rank(r):
    """Returns the rank of the original word among the suggestions, or 0 when it is absent."""
    try:
        return int(r["rank_of_original"])
    except Exception:
        return 0


def classify(rows, vocab):
    """Assigns every detected error to one category, and separates the near misses."""
    for r in rows:
        k = _rank(r)
        want = r["intended_original"]
        if k == 1:
            r["_cat"] = "success_rank1"
        elif 2 <= k <= 5:
            r["_cat"] = "near_miss"
        elif want.lower() not in vocab:
            r["_cat"] = "miss_not_in_vocab"
        elif Levenshtein.distance(r["corrupted"].lower(), want.lower()) > MAX_EDIT:
            r["_cat"] = "miss_too_far"
        else:
            r["_cat"] = "miss_outranked"
    return rows


def near_miss_split(rows, freq):
    """Splits the near misses by whether the winning suggestion was closer or only more frequent."""
    closer = tied = 0
    examples = []
    for r in (x for x in rows if x["_cat"] == "near_miss"):
        sugg = [s for s in r["suggestions"].split("|") if s]
        if not sugg:
            continue
        w, want, top = r["corrupted"].lower(), r["intended_original"], sugg[0]
        d_top = Levenshtein.distance(w, top.lower())
        d_want = Levenshtein.distance(w, want.lower())
        if d_top < d_want:
            closer += 1
        elif d_top == d_want:
            tied += 1
            if len(examples) < PER_CATEGORY and freq.get(top, 0) > freq.get(want, 0):
                examples.append({"corrupted": r["corrupted"], "intended": want,
                                 "intended_freq": freq.get(want, 0),
                                 "beaten_by": top, "winner_freq": freq.get(top, 0)})
    return closer, tied, examples


def build(lang):
    """
    Classifies the corrections of one language. Returns the classified rows, a few examples
    per category, the summary and the words that outranked the original.
    """
    f = RQ4_DIR / f"rq4_corrections_seed0_{lang}.csv"
    if not f.exists():
        cands = sorted(RQ4_DIR.glob("rq4_corrections_seed0_*.csv"))
        if not cands:
            raise FileNotFoundError(f"no RQ4 correction CSV in {RQ4_DIR}")
        f = cands[0]
    rows = list(csv.DictReader(open(f, encoding="utf-8")))

    wm = TrigramModel.load(ROOT / "data" / "models" / f"{lang}_word.pkl")
    vocab = {w.lower() for w in wm.vocab}
    rows = classify(rows, vocab)
    closer, tied, competitors = near_miss_split(rows, wm.counts)

    counts = {k: sum(1 for r in rows if r["_cat"] == k) for k, _, _ in CATEGORIES}
    picked = []
    for key, _, _ in CATEGORIES:
        for r in [x for x in rows if x["_cat"] == key][:PER_CATEGORY]:
            picked.append({"category": key, "corrupted": r["corrupted"],
                           "intended": r["intended_original"],
                           "rank_of_intended": r["rank_of_original"],
                           "detection_score": r.get("detection_score", ""),
                           "suggestions": r.get("suggestions", "")})
    summary = {"language": lang, "source_csv": f.name, "total": len(rows),
               "counts": counts,
               "near_miss_winner_closer": closer, "near_miss_winner_tied_on_frequency": tied,
               "unfixable_pct": 100 * counts["miss_not_in_vocab"] / len(rows) if rows else 0.0}
    return rows, picked, summary, competitors


def chart(lang, summary, out_dir):
    """Draws the categories as a horizontal bar chart."""
    labels = [lbl for _, lbl, _ in CATEGORIES]
    vals = [summary["counts"][k] for k, _, _ in CATEGORIES]
    colours = [c for _, _, c in CATEGORIES]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.barh(range(len(vals)), vals, color=colours, alpha=0.9)
    top = max(vals) or 1
    for y, v in enumerate(vals):
        ax.text(v + top * 0.015, y, f"{v:,} ({100*v/summary['total']:.1f}%)",
                va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.36)
    if top >= 1e3:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:g}k"))
    ax.set_xlabel(f"Detected errors ({disp(lang)}, seed 0, {summary['total']:,} total)", fontsize=10)
    ax.set_title(f"Why correction fails\nonly {summary['unfixable_pct']:.1f}% is impossible "
                 f"(word not in training vocabulary)", fontsize=10, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    pdf, _ = write_figure(fig, out_dir, f"qual_rq4_failures_{lang}")
    plt.close(fig)
    return pdf


def run(lang="zu"):

    """Classifies the corrections of one language and writes the summary, the CSV and the chart."""
    _, picked, summary, competitors = build(lang)
    RQ4_DIR.mkdir(parents=True, exist_ok=True)

    active = mlflow.active_run() is not None
    if not active:
        mlflow.set_experiment("RQ4_Correction_Qualitative")
    with (nullcontext() if active else mlflow.start_run()):
        mlflow.log_params({"experiment": "qual_rq4_corrections", "language": lang,
                           "source_csv": summary["source_csv"], "max_edit": MAX_EDIT})
        written = []
        with open(RQ4_DIR / f"qual_correction_examples_{lang}.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(picked[0].keys()))
            w.writeheader(); w.writerows(picked)
        written.append(RQ4_DIR / f"qual_correction_examples_{lang}.csv")
        if competitors:
            with open(RQ4_DIR / f"qual_rq4_competitors_{lang}.csv", "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(competitors[0].keys()))
                w.writeheader(); w.writerows(competitors)
            written.append(RQ4_DIR / f"qual_rq4_competitors_{lang}.csv")
        (RQ4_DIR / f"qual_rq4_failures_{lang}.json").write_text(json.dumps(summary, indent=2))
        written.append(RQ4_DIR / f"qual_rq4_failures_{lang}.json")
        pdf = chart(lang, summary, RQ4_DIR)
        written += [pdf, pdf.with_suffix(".png")]

        tot = summary["total"]
        mlflow.log_metrics({f"pct_{k}": 100 * v / tot for k, v in summary["counts"].items()})
        mlflow.log_metrics({f"n_{k}": v for k, v in summary["counts"].items()})
        mlflow.log_metric("unfixable_pct", summary["unfixable_pct"])
        mlflow.log_metrics({"near_miss_winner_closer": summary["near_miss_winner_closer"],
                            "near_miss_winner_tied_on_frequency":
                                summary["near_miss_winner_tied_on_frequency"]})
        for p in written:
            if p.exists():
                mlflow.log_artifact(str(p))

    print(f"\n{'='*82}\n  RQ4 correction outcomes ({lang}, {summary['source_csv']}, {tot:,} errors)\n{'='*82}")
    for key, label, _ in CATEGORIES:
        n = summary["counts"][key]
        print(f"  {label:<38}{n:>9,}  {100*n/tot:>6.1f}%")
    c, t = summary["near_miss_winner_closer"], summary["near_miss_winner_tied_on_frequency"]
    if c + t:
        print(f"\n  Near misses - winner was closer in edit distance: {c:>8,} ({100*c/(c+t):.1f}%)")
        print(f"                winner tied, won on corpus frequency: {t:>8,} ({100*t/(c+t):.1f}%)")
    if competitors:
        print(f"\n  {'corrupted':<20}{'intended':<20}{'freq':>7}   {'beaten by':<18}{'freq':>7}")
        for r in competitors:
            print(f"  {r['corrupted'][:19]:<20}{r['intended'][:19]:<20}{r['intended_freq']:>7}"
                  f"   {r['beaten_by'][:17]:<18}{r['winner_freq']:>7}")
    print(f"{'='*82}\n  wrote results/rq4/qual_rq4_failures_{lang}.*\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zu")
    run(ap.parse_args().lang)
