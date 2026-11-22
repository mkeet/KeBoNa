"""
plot_rq3_comparison.py - bar charts comparing the tools on the fixed 500-word set (RQ3).

Reads the summaries written by exp_rq3_external_eval.py, kenlm_detect.py and
score_sadilar_run.py and draws one chart per corruption type. Nothing is run again.

Outputs (results/rq3/): rq3_tool_comparison_random.{pdf,png}, rq3_tool_comparison_qwerty.{pdf,png}
"""

from contextlib import nullcontext
import csv
import glob
import logging
import os
from pathlib import Path
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import write_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RQ3 = ROOT / "results" / "rq3"

CTYPES = ["random", "qwerty"]
CTYPE_LABELS = {"random": "Random", "qwerty": "QWERTY-adjacent"}
CTYPE_STEM = {"random": "rq3_tool_comparison_random", "qwerty": "rq3_tool_comparison_qwerty"}

# labels and colours of the checker's bars, keyed by the language suffix of the summary file
SPELL_LABELS = {"zulu": "char-trigram (SADiLaR isiZulu)", "zu": "char-trigram (NCHLT isiZulu)"}
SPELL_COLOURS = {"zulu": "#2b7bba", "zu": "#7fb0d8"}
# the 500 words are sampled from the SADiLaR isiZulu test set, so only that model is plotted
SPELL_PLOT = ("zulu",)
# the other tools, each with its label, colour and summary file
EXT = {
    "kenlm":   ("KenLM", "#c0603a", "rq3_external_kenlm_summary.csv"),
    "sadilar": ("SADiLaR word extension tool", "#5a9e5a", "rq3_external_sadilar_summary.csv"),
}


def _read_mf1(path):
    """Reads the macro-F1 per corruption type from a summary CSV."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ct = row.get("corruption_type")
            if ct in CTYPES and row.get("mf1") not in (None, ""):
                out[ct] = float(row["mf1"])
    return out


def load_systems():
    """
    Returns the label, colour and macro-F1 of each system, the checker first and the other tools after it.
    """
    systems = []
    prefix = "rq3_external_spellchecker_summary_"
    for p in sorted(glob.glob(str(RQ3 / f"{prefix}*.csv"))):
        lang = os.path.basename(p)[len(prefix):-len(".csv")]
        if lang not in SPELL_PLOT:
            logger.info("[spellchecker %s] skipped (not in SPELL_PLOT)", lang)
            continue
        data = _read_mf1(p)
        if data:
            systems.append((SPELL_LABELS.get(lang, f"char-trigram ({lang})"),
                            SPELL_COLOURS.get(lang, "#4477aa"), data))
            logger.info("[spellchecker %s] %s", lang,
                        ", ".join(f"{c}={data.get(c, float('nan')):.3f}" for c in CTYPES))
    for _key, (label, colour, fname) in EXT.items():
        p = RQ3 / fname
        if p.exists():
            data = _read_mf1(p)
            if data:
                systems.append((label, colour, data))
                logger.info("[%s] %s", label,
                            ", ".join(f"{c}={data.get(c, float('nan')):.3f}" for c in CTYPES))
    return systems


def plot_one(systems, ctype):
    """Draws the chart of one corruption type."""
    present = [(l, c, d) for (l, c, d) in systems if ctype in d]
    if not present:
        logger.warning("[%s] no data - skipping", ctype)
        return None
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n = len(present)
    width = 0.8 / n
    xc = 0.0
    for j, (label, colour, data) in enumerate(present):
        x = xc + (j - (n - 1) / 2) * width
        h = data[ctype]
        ax.bar(x, h, width, label=label, color=colour, alpha=0.9, edgecolor="white", linewidth=0.6)
        ax.text(x, h + 0.008, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks([xc])
    ax.set_xticklabels([CTYPE_LABELS.get(ctype, ctype)])
    ax.set_xlabel("Error-injection type", fontsize=12)
    ax.set_ylabel("Macro F1 (balanced detection)", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.5, 0.5)
    ax.set_title(f"RQ3: detection mF1 on isiZulu (fixed 500-word set)\n"
                 f"{CTYPE_LABELS.get(ctype, ctype)} error injection", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Tool")
    fig.tight_layout()
    pdf, _png = write_figure(fig, RQ3, CTYPE_STEM[ctype], bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s (+ .png)", pdf.name)
    return pdf


def run():
    """Draws both charts."""
    systems = load_systems()
    if not systems:
        raise FileNotFoundError(f"No RQ3 summaries found in {RQ3}")
    active = mlflow.active_run() is not None
    if not active:
        mlflow.set_experiment("RQ3_Baseline_Comparison")
    with (nullcontext() if active else mlflow.start_run()):
        made = [plot_one(systems, ct) for ct in CTYPES]
        print("\n  RQ3 per-error-type tool-comparison charts written:")
        for p in made:
            if p:
                mlflow.log_artifact(str(p))
                png = p.with_suffix(".png")
                if png.exists():
                    mlflow.log_artifact(str(png))
                print(f"    {p} (+ .png)")
        print()


if __name__ == "__main__":
    run()
