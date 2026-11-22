"""
rq2_weight_sensitivity.py - how much the result of RQ2 depends on the weights of the distance.

The distance between two languages is 10 * zone borders + 3 * group difference + 1 * language
difference. The weights 10, 3 and 1 are a choice, so the Spearman correlation between distance
and transfer ratio is recomputed here for every combination of weights on a grid. The
correlation of the weightless ordering, which ranks pairs by zone borders, then group, then
language, is given beside it.

Reads results/rq2/<mode>-based/rq2_transfer_summary<suffix>.csv, which
exp_rq2_cross_language.py writes. No model is loaded.

Outputs (results/rq2/<mode>-based/):
    rq2_weight_sensitivity<suffix>.csv          the correlation for every combination of weights
    rq2_weight_sensitivity<suffix>.json         the chosen weights against the range over the grid
    rq2_weight_sensitivity<suffix>.{pdf,png}    histogram of the correlations
"""

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import write_csv, write_figure
from src.utils.guthrie_distance import GROUP_WEIGHT, LANG_WEIGHT, ZONE_WEIGHT

RQ2 = ROOT / "results" / "rq2"
GRID = [0, 1, 2, 3, 5, 10, 20, 50, 100]
CHOSEN = (ZONE_WEIGHT, GROUP_WEIGHT, LANG_WEIGHT)


def load_pairs(mode, suffix):
    """Reads every pair of different languages with its three distance parts and its transfer ratio."""
    path = RQ2 / f"{mode}-based" / f"rq2_transfer_summary{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run exp_rq2_cross_language.py first")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["source"] != r["target"]]
    parts = np.array([[float(r["zone_hops"]), float(r["guthrie_group_diff"]),
                       float(r["guthrie_lang_diff"])] for r in rows])
    ratio = np.array([float(r["transfer_ratio"]) for r in rows])
    lexrank = np.array([float(r["guthrie_graph_lexrank"]) for r in rows])
    return parts, ratio, lexrank


def correlation(parts, ratio, weights):
    """Returns Spearman's rho and its p-value between the weighted distance and the transfer ratio."""
    rho, p = spearmanr(parts @ np.array(weights, dtype=float), ratio)
    return float(rho), float(p)


def run(mode="token", suffix="_5"):
    """
    Recomputes the correlation for every combination of weights on the grid and writes the
    table, the summary and the histogram.

    Args:
        mode: "token" or "type", the evaluation mode of the saved run.
        suffix: The version suffix of the saved run. "_5" is the set of 21 languages.

    Returns:
        The summary as a dict.
    """
    parts, ratio, lexrank = load_pairs(mode, suffix)
    rows = []
    for w in itertools.product(GRID, repeat=3):
        if not any(w):
            continue
        rho, p = correlation(parts, ratio, w)
        rows.append({"w_zone": w[0], "w_group": w[1], "w_lang": w[2],
                     "ordered": int(w[0] >= w[1] >= w[2] > 0),
                     "spearman_rho": round(rho, 6), "p_value": p})

    chosen_rho, chosen_p = correlation(parts, ratio, CHOSEN)
    lex_rho, lex_p = spearmanr(lexrank, ratio)

    def span(sel):
        r = np.array([x["spearman_rho"] for x in sel])
        return {"combinations": len(sel), "rho_min": float(r.min()), "rho_median": float(np.median(r)),
                "rho_max": float(r.max()),
                "share_negative_and_p_below_0.05": float(np.mean(
                    [x["spearman_rho"] < 0 and x["p_value"] < 0.05 for x in sel]))}

    summary = {
        "mode": mode, "version_suffix": suffix, "pairs": int(len(ratio)), "grid": GRID,
        "chosen_weights": {"zone": CHOSEN[0], "group": CHOSEN[1], "lang": CHOSEN[2],
                           "rho": chosen_rho, "p": chosen_p},
        "weightless_ordering": {"rho": float(lex_rho), "p": float(lex_p)},
        "one_part_only": {name: dict(zip(("rho", "p"), correlation(parts, ratio, w)))
                          for name, w in (("zone", (1, 0, 0)), ("group", (0, 1, 0)), ("lang", (0, 0, 1)))},
        "all_combinations": span(rows),
        "zone_ge_group_ge_lang_gt_0": span([x for x in rows if x["ordered"]]),
    }

    out = RQ2 / f"{mode}-based"
    write_csv(out / f"rq2_weight_sensitivity{suffix}.csv", rows)
    (out / f"rq2_weight_sensitivity{suffix}.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([x["spearman_rho"] for x in rows], bins=30, color="#9aa5b1", label="all combinations")
    ax.hist([x["spearman_rho"] for x in rows if x["ordered"]], bins=30, color="#4e79a7",
            label="zone ≥ group ≥ language > 0")
    ax.axvline(chosen_rho, color="#d62728", linewidth=1.5, label="10 / 3 / 1")
    ax.axvline(float(lex_rho), color="#2ca02c", linewidth=1.5, linestyle="--", label="weightless ordering")
    ax.set_xlabel("Spearman's rho between distance and transfer ratio")
    ax.set_ylabel("Combinations of weights")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    write_figure(fig, out, f"rq2_weight_sensitivity{suffix}")
    plt.close(fig)

    a, o = summary["all_combinations"], summary["zone_ge_group_ge_lang_gt_0"]
    print(f"\n[{mode}{suffix}] {len(ratio)} pairs")
    print(f"  10/3/1               rho = {chosen_rho:.4f}  (p = {chosen_p:.2g})")
    print(f"  weightless ordering  rho = {lex_rho:.4f}  (p = {lex_p:.2g})")
    for name, v in summary["one_part_only"].items():
        print(f"  {name:<5} only           rho = {v['rho']:.4f}  (p = {v['p']:.2g})")
    print(f"  all {a['combinations']} combinations      rho from {a['rho_min']:.4f} to {a['rho_max']:.4f}, "
          f"median {a['rho_median']:.4f}")
    print(f"  {o['combinations']} ordered combinations  rho from {o['rho_min']:.4f} to {o['rho_max']:.4f}, "
          f"median {o['rho_median']:.4f}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["token", "type"])
    ap.add_argument("--suffix", default="_5")
    a = ap.parse_args()
    for mode in a.modes:
        run(mode, a.suffix)
