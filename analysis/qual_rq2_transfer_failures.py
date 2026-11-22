"""
qual_rq2_transfer_failures.py - the transfer failures of RQ2, word by word.

Turns the word-by-word CSVs of RQ2 into a list of valid words of the target language that the
model of the source language wrongly flags. Those CSVs cover one source language against every
target. Only saved CSVs are read, and no model is loaded.

Outputs (results/rq2/), per mode:
    qual_transfer_failures_{mode}.csv         one row per source and target
    qual_transfer_failures_{mode}_words.csv   every wrongly flagged word
"""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from src.utils.guthrie_distance import guthrie_graph_distance, guthrie_graph_components, GUTHRIE_CODES

DISP = {"zu": "isiZulu", "xh": "isiXhosa", "nd": "isiNdebele", "sw": "siSwati",
        "zulu": "isiZulu", "xhosa": "isiXhosa", "ndebele": "isiNdebele", "swati": "siSwati",
        "sotho": "Sesotho", "tswana": "Setswana", "shona": "ciShona", "lozi": "siLozi",
        "swahili": "Kiswahili"}
BARCOL = {"xh": "#ff7f0e", "xhosa": "#ff7f0e", "sw": "#e377c2", "swati": "#e377c2",
            "nd": "#bcbd22", "ndebele": "#bcbd22", "sotho": "#8c564b", "tswana": "#9467bd",
            "shona": "#d62728", "lozi": "#17becf", "swahili": "#2ca02c"}


def plot_failures(mode, summary_rows, source):
    """Draws the false-alarm rate on valid target words, ordered by Guthrie distance."""
    rows = [r for r in summary_rows if r["source"] == source and r["guthrie_graph_distance"] is not None]
    if not rows:
        return None
    rows.sort(key=lambda r: r["guthrie_graph_distance"])
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for i, r in enumerate(rows):
        t = r["target"]
        ax.bar(i, r["false_alarm_rate"], color=BARCOL.get(t, "#4477aa"), alpha=0.92, edgecolor="white")
        ax.text(i, r["false_alarm_rate"] + 0.015, f"{r['false_alarm_rate']:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{DISP.get(r['target'], r['target'])} ({GUTHRIE_CODES.get(r['target'], '?')})\n"
                        f"(d={r['guthrie_graph_distance']})" for r in rows], fontsize=10)
    ax.set_xlabel("Target language (Guthrie code)", fontsize=12)
    ax.set_ylabel("Fraction of valid target words flagged", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"The transfer penalty word-by-word: valid target words rejected by the "
                f"{DISP.get(source, source)} ({GUTHRIE_CODES.get(source, '?')}) model\n"
                f"({mode}-level; d = zone-adjacency distance)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = RQ2_DIR / f"qual_transfer_failures_{mode}.pdf"
    fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return out

RQ2_DIR = ROOT / "results" / "rq2"
# the word-by-word files of every version, rq2_words_seed0.csv and its _2, _3 and _4 siblings
SUFFIXES = ["", "_2", "_3", "_4"]
N_EXAMPLES = 8


def _native_looking(w):
    """
    Returns True for a word in lower case, of letters and hyphens only, with four or more
    characters. This is applied only to the examples quoted, to keep acronyms and stray English
    out of them. The false-alarm counts are not filtered.
    """
    return (len(w) >= 4 and w == w.lower()
            and all(c.isalpha() or c == "-" for c in w))


def _dist(source, target):
    """
    Returns the Guthrie distance between two languages and the zone borders between them, or
    None for both when either language has no code.
    """
    try:
        d = guthrie_graph_distance(source, target)
        hops, gd, ld = guthrie_graph_components(source, target)
        return d, hops
    except Exception:
        return None, None


def build(mode):
    """
    Collects the wrongly flagged words of every pair and returns the summary rows and the word rows.
    """
    per_pair = defaultdict(list)
    per_pair_total = defaultdict(int)
    for suf in SUFFIXES:
        f = RQ2_DIR / f"{mode}-based" / f"rq2_words_seed0{suf}.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["instance"] != "clean" or r["source"] == r["target"]:
                continue
            key = (r["source"], r["target"])
            per_pair_total[key] += 1
            if r["outcome"] == "fp":
                per_pair[key].append(r)

    summary_rows, word_rows = [], []
    # a pair can appear in several versions, so the one with the most instances is kept
    best_total = {}
    for key, tot in per_pair_total.items():
        if tot > best_total.get(key, -1):
            best_total[key] = tot

    seen = set()
    for (source, target), fps in sorted(per_pair.items(),
                                        key=lambda kv: (kv[0][0], _dist(*kv[0])[0] or 0)):
        if (source, target) in seen:
            continue
        seen.add((source, target))
        total = best_total[(source, target)]
        d, hops = _dist(source, target)
        # deduplicate, prefer native-looking words, sort by distance below tau
        uniq = {}
        for r in fps:
            w = r["original"]
            if w not in uniq:
                uniq[w] = r
        def _score(r):
            try: return float(r["detection_score"])
            except Exception: return 0.0
        ordered = sorted(uniq.values(), key=_score)
        examples = [r for r in ordered if _native_looking(r["original"])] or ordered
        ex_words = [r["original"] for r in examples[:N_EXAMPLES]]

        summary_rows.append({
            "source": source, "target": target,
            "guthrie_graph_distance": d, "zone_hops": hops,
            "target_words_tested": total,
            "valid_words_wrongly_flagged": len(fps),
            "false_alarm_rate": round(len(fps) / total, 4) if total else 0.0,
            "example_words": " ".join(ex_words),
        })
        for r in examples[: max(N_EXAMPLES, 20)]:
            word_rows.append({
                "source": source, "target": target,
                "guthrie_graph_distance": d, "zone_hops": hops,
                "word": r["original"], "detection_score": r["detection_score"],
                "tau": r["tau"], "trigram_logprobs": r.get("trigram_logprobs", ""),
            })

    summary_rows.sort(key=lambda s: (s["source"], s["guthrie_graph_distance"] or 0))
    return summary_rows, word_rows


def write_and_print(mode, summary_rows, word_rows):
    """Writes the two CSVs and prints the summary."""
    if not summary_rows:
        print(f"[{mode}] no rq2 audit CSVs found - run exp_rq2_cross_language.py first.")
        return
    for name, rows in [(f"qual_transfer_failures_{mode}.csv", summary_rows),
                        (f"qual_transfer_failures_{mode}_words.csv", word_rows)]:
        p = RQ2_DIR / name
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"\n{'='*92}\n  RQ2 transfer failures ({mode}-level): valid target words the SOURCE model "
            f"wrongly flags\n{'='*92}")
    print(f"  {'src->target':<18}{'dist':>5}{'hops':>5}{'tested':>8}{'flagged':>9}{'rate':>7}   examples")
    print(f"  {'-'*88}")
    for s in summary_rows:
        print(f"  {s['source']+'->'+s['target']:<18}{str(s['guthrie_graph_distance']):>5}"
                f"{str(s['zone_hops']):>5}{s['target_words_tested']:>8}"
                f"{s['valid_words_wrongly_flagged']:>9}{s['false_alarm_rate']:>7.3f}   "
                f"{' '.join(s['example_words'].split()[:5])}")
    print(f"{'='*92}")
    src0 = summary_rows[0]["source"]
    fig_path = plot_failures(mode, summary_rows, src0)
    print(f"  wrote results/rq2/qual_transfer_failures_{mode}.csv "
            f"(+ _words.csv, {len(word_rows)} example instances"
            f"{', + .pdf/.png chart' if fig_path else ''})\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["type", "token"])
    a = ap.parse_args()
    for mode in a.modes:
        s, w = build(mode)
        write_and_print(mode, s, w)
