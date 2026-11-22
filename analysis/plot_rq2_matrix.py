"""
plot_rq2_matrix.py - redraws the transfer matrix of RQ2 from the saved results.

Reads rq2_correlation_matrix<suffix>.csv and rq2_transfer<suffix>.json and draws the heatmap
wide enough to stay legible on a slide. Nothing is evaluated again, so the numbers are those
of the saved run. The axis labels carry the Guthrie code of each language, which makes the
zones visible on the axes.

Outputs (results/rq2/<mode>-based/): rq2_matrix_coded<suffix>.{pdf,png}
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import write_figure
from src.utils.guthrie_distance import GUTHRIE_CODES
from exp_rq2_cross_language import _zone_graph_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RQ2 = ROOT / "results" / "rq2"

# maps the display names stored in the CSV back to the keys of GUTHRIE_CODES
_KEY = {"Zulu": "zu", "Xhosa": "xh", "Ndebele": "nd", "Swati": "sw", "Sepedi": "sepedi",
        "Venda": "venda", "Tsonga": "tsonga", "Runyankore": "runyankore", "Herero": "herero",
        "Kongo": "kongo", "Kimbundu": "kimbundu"}


def code_for(name):
    """Returns the Guthrie code of a language from its display name."""
    key = _KEY.get(name, name.lower())
    return GUTHRIE_CODES.get(key)


def load(mode, suffix):
    """Reads the similarity matrix and the Spearman results of one run."""
    d = RQ2 / f"{mode}-based"
    rows = list(csv.DictReader(open(d / f"rq2_correlation_matrix{suffix}.csv", encoding="utf-8")))
    names = [r["language"] for r in rows]
    S = np.array([[float(r[n]) for n in names] for r in rows])
    j = json.load(open(d / f"rq2_transfer{suffix}.json", encoding="utf-8"))
    return names, S, j["spearman_graphweighted_vs_transfer_ratio"], d


def draw(names, S, sp, print_size=False, cells=True):
    """
    Draws the matrix. With print_size the figure is near-square with larger type, for a full
    printed page. With cells=False the numbers in the cells are left out.
    """
    n = len(names)
    labels = [f"{x} ({code_for(x)})" if code_for(x) else x for x in names]
    off = S[~np.eye(n, dtype=bool)]
    vmin = round(float(off.min()), 2)

    if not cells:      fs, cell, tick = (7.6, 6.3), 0, 9
    elif print_size:   fs, cell, tick = (8.0, 7.2), 7.0, 9
    else:              fs, cell, tick = (13.6, 6.6), 7.6, 11
    fig, ax = plt.subplots(figsize=fs)
    # rectangular cells, so the figure fills the width and the cell text stays legible
    im = ax.imshow(S, cmap="RdYlGn", vmin=vmin, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=tick)
    if cells:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{S[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                        color="black", fontsize=cell, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("symmetric transfer ratio (mean of both directions)", fontsize=9)
    ax.set_title("" if not cells else f"RQ2 transfer similarity - {n} languages, Guthrie code in brackets\n"
                 f"Spearman(zone-adjacency distance, similarity) rho={sp['rho']:.3f}, "
                 f"p<0.0001, n={sp['n_pairs']} pairs", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig, vmin


DISP = {"zu": "isiZulu", "xh": "isiXhosa", "nd": "isiNdebele", "sw": "siSwati", "shona": "Shona",
        "sotho": "Sesotho", "tswana": "Setswana", "swahili": "Kiswahili", "lozi": "Silozi",
        "kinyarwanda": "Kinyarwanda", "kirundi": "Kirundi", "luganda": "Luganda", "kikuyu": "Gikuyu",
        "kamba": "Kamba", "chichewa": "Chichewa", "tumbuka": "Tumbuka", "bemba": "Bemba",
        "umbundu": "Umbundu", "luba": "Luba", "yao": "Yao", "lumbu": "Lumbu"}
DISJUNCTIVE = {"sotho", "tswana", "lozi"}


def run_directional(mode="token", suffix="_5", origin="zu", cmap="YlOrBr"):
    """
    Draws the directed matrix, source against target, with the languages ordered by their
    distance from origin.
    """
    d = RQ2 / f"{mode}-based"
    V, D = {}, {}
    for r in csv.DictReader(open(d / f"rq2_transfer_summary{suffix}.csv", encoding="utf-8")):
        V[(r["source"], r["target"])] = float(r["mf1"])
        D[(r["source"], r["target"])] = float(r["guthrie_graph_distance"])
    langs = sorted({s for s, _ in V}, key=lambda l: (D[(origin, l)], GUTHRIE_CODES[l]))
    n = len(langs)
    M = np.array([[V[(s, t)] for t in langs] for s in langs])
    labels = [f"{DISP.get(l, l)} ({GUTHRIE_CODES[l]}) {'D' if l in DISJUNCTIVE else 'A'}, "
              f"d={D[(origin, l)]:.0f}" for l in langs]

    fig, ax = plt.subplots(figsize=(8.0, 7.4))
    im = ax.imshow(M, cmap=cmap, vmin=M.min(), vmax=M.max())
    mid = (M.min() + M.max()) / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.6,
                    color="white" if M[i, j] > mid else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=6.8)
    zones = [GUTHRIE_CODES[l][0] for l in langs]
    for k in range(1, n):
        if zones[k] != zones[k - 1]:
            ax.axhline(k - 0.5, color="#9a9a9a", lw=0.6); ax.axvline(k - 0.5, color="#9a9a9a", lw=0.6)
    ax.set_xlabel("Target language (test set)", fontsize=8.5)
    ax.set_ylabel("Source language (model)", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("detection macro-F1", fontsize=8)
    fig.tight_layout()
    pdf, png = write_figure(fig, d, f"rq2_matrix_directional{suffix}")
    plt.close(fig)
    logger.info("%d languages ordered from %s, mF1 %.3f-%.3f", n, origin, M.min(), M.max())
    logger.info("wrote %s (+ .png)", pdf.name)
    return png


def run(mode="token", suffix="_5", print_size=False, cells=True):
    """Draws the symmetric matrix of one run and saves it as PDF and PNG."""
    names, S, sp, out = load(mode, suffix)
    fig, vmin = draw(names, S, sp, print_size, cells)
    stem = f"rq2_matrix_coded{suffix}" + ("_print" if print_size else "") + ("" if cells else "_nocells")
    pdf, png = write_figure(fig, out, stem)
    plt.close(fig)
    logger.info("%d languages, vmin=%.2f, rho=%.4f", len(names), vmin, sp["rho"])
    logger.info("wrote %s (+ .png)", pdf.name)
    return png


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="token", choices=["token", "type"])
    ap.add_argument("--suffix", default="_5")
    ap.add_argument("--print-size", action="store_true", help="square, larger type, for a printed page")
    ap.add_argument("--zone-graph", action="store_true", help="redraw the zone-adjacency graph instead")
    ap.add_argument("--no-cell-numbers", action="store_true", help="colour only, for an upright column figure")
    ap.add_argument("--directional", action="store_true",
                    help="source x target mF1, both directions, ordered by distance from isiZulu")
    a = ap.parse_args()
    if a.directional:
        run_directional(a.mode, a.suffix)
    elif a.zone_graph:
        names, _, _, _ = load(a.mode, a.suffix)
        zones = {GUTHRIE_CODES[_KEY.get(n, n.lower())][0] for n in names
                 if _KEY.get(n, n.lower()) in GUTHRIE_CODES}
        _zone_graph_figure(RQ2, sorted(zones), print_size=a.print_size)
        logger.info("wrote guthrie_zone_graph%s.pdf (%d zones marked)",
                    "_print" if a.print_size else "", len(zones))
    else:
        run(a.mode, a.suffix, a.print_size, not a.no_cell_numbers)
