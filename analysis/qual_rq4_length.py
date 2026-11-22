"""
qual_rq4_length.py - correction accuracy against the length of the intended word, for seed 0.

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
import mlflow

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import write_figure, disp

RQ4_DIR = ROOT / "results" / "rq4"
MIN_BUCKET = 500  # smallest sample per length that is plotted, lowered for small runs
MAX_LEN = 16


def _rank(r):
    """Returns the rank of the original word among the suggestions, or 0 when it is absent."""
    try:
        return int(r["rank_of_original"])
    except Exception:
        return 0


def build(lang):
    """
    Groups the corrections of one language by word length. Returns CA@1, CA@5 and MRR per
    length, the name of the CSV read and the number of corrections.
    """
    f = RQ4_DIR / f"rq4_corrections_seed0_{lang}.csv"
    if not f.exists():
        cands = sorted(RQ4_DIR.glob("rq4_corrections_seed0_*.csv"))
        if not cands:
            raise FileNotFoundError(f"no RQ4 correction CSV in {RQ4_DIR}")
        f = cands[0]
    rows = list(csv.DictReader(open(f, encoding="utf-8")))

    buckets = {}
    for r in rows:
        L = min(len(r["intended_original"]), MAX_LEN)
        k = _rank(r)
        b = buckets.setdefault(L, {"n": 0, "ca1": 0, "ca5": 0, "rr": 0.0})
        b["n"] += 1
        b["ca1"] += (k == 1)
        b["ca5"] += (1 <= k <= 5)
        b["rr"] += (1.0 / k) if k else 0.0

    floor = max(25, min(MIN_BUCKET, len(rows) // 100))
    series = [{"length": L, "n": b["n"],
               "ca1": b["ca1"] / b["n"], "ca5": b["ca5"] / b["n"], "mrr": b["rr"] / b["n"]}
              for L, b in sorted(buckets.items()) if b["n"] >= floor]
    if not series:
        raise ValueError(f"{f.name}: no length bucket reaches {floor} corrections")
    return series, f.name, len(rows)


def chart(lang, series, total, out_dir):
    """Draws accuracy against word length."""
    xs = [s["length"] for s in series]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for key, label, colour in (("ca5", "CA@5", "#98c379"),
                               ("mrr", "MRR", "#ff7f0e"),
                               ("ca1", "CA@1", "#1f77b4")):
        ax.plot(xs, [s[key] for s in series], marker="o", ms=4, lw=2,
                color=colour, label=label)
    ax.set_xlabel(f"Length of the intended word (characters, {MAX_LEN}+ pooled)", fontsize=10)
    ax.set_ylabel("Correction accuracy", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    lo, hi = series[0]["ca1"], max(s["ca1"] for s in series)
    ax.set_title(f"Longer words are easier to correct - CA@1 rises {lo:.2f} to {hi:.2f}\n"
                 f"{disp(lang)}, seed 0, {total:,} detected errors",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    pdf, _ = write_figure(fig, out_dir, f"qual_rq4_length_{lang}")
    plt.close(fig)
    return pdf


def run(lang="zu"):

    """
    Computes the accuracy per length for one language and writes the JSON, the CSV and the chart.
    """
    series, src, total = build(lang)
    RQ4_DIR.mkdir(parents=True, exist_ok=True)
    out = {"language": lang, "source_csv": src, "total": total, "by_length": series}

    active = mlflow.active_run() is not None
    if not active:
        mlflow.set_experiment("RQ4_Correction_Qualitative")
    with (nullcontext() if active else mlflow.start_run()):
        mlflow.log_params({"experiment": "qual_rq4_length", "language": lang,
                           "source_csv": src, "max_len": MAX_LEN})
        with open(RQ4_DIR / f"qual_rq4_length_{lang}.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(series[0].keys()))
            w.writeheader(); w.writerows(series)
        (RQ4_DIR / f"qual_rq4_length_{lang}.json").write_text(json.dumps(out, indent=2))
        pdf = chart(lang, series, total, RQ4_DIR)

        for s in series:
            mlflow.log_metrics({"len_ca1": s["ca1"], "len_ca5": s["ca5"], "len_mrr": s["mrr"],
                                "len_n": s["n"]}, step=s["length"])
        mlflow.log_metrics({"ca1_shortest": series[0]["ca1"],
                            "ca1_best": max(s["ca1"] for s in series)})
        for p in (RQ4_DIR / f"qual_rq4_length_{lang}.csv",
                  RQ4_DIR / f"qual_rq4_length_{lang}.json", pdf, pdf.with_suffix(".png")):
            if p.exists():
                mlflow.log_artifact(str(p))

    print(f"\n{'='*64}\n  RQ4 correction by word length ({lang}, {src}, {total:,} errors)\n{'='*64}")
    print(f"  {'len':>4}{'n':>10}{'CA@1':>9}{'CA@5':>9}{'MRR':>9}")
    for s in series:
        print(f"  {s['length']:>4}{s['n']:>10,}{s['ca1']:>9.3f}{s['ca5']:>9.3f}{s['mrr']:>9.3f}")
    print(f"{'='*64}\n  wrote results/rq4/qual_rq4_length_{lang}.*\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zu")
    run(ap.parse_args().lang)
