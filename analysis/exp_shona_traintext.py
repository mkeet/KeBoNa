"""
exp_shona_traintext.py - trains Shona models on the Melokuhle text as typed and as corrected.

The two files hold the same 2,113 words, and the raw one carries 118 real typos. Training on
each shows what typing errors in the training text cost when nothing else differs. The text
is deliberately not passed through CorpusCleaner, because the comparison is between the text
as written and the text as corrected, and cleaning would remove that difference.
"""

import argparse
import json
import logging
from pathlib import Path
import random
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import (train_detection_model, train_lr, evaluate_balanced_model, load_words,
                        corpus_path, sample_unique, write_csv, write_figure, mean_std)
from src.utils.text_utils import TextUtils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAT = ROOT / "shona_material"
OUT = ROOT / "results" / "shona_realtext"
VARIANTS = {"as typed": "melokuhle_shona_raw-MK.txt",
            "corrected": "melokuhle_shona_corrected-MK.txt"}


def words_of(path):
    """Returns the words of a text file."""
    return TextUtils.tokenize(TextUtils.normalize(path.read_text(encoding="utf-8")))


def split_802010(words, seed=42):
    """Splits the words 80/10/10 with the same seeded shuffle as DataSplitter."""
    w = list(words); random.Random(seed).shuffle(w)
    a, b = int(len(w) * 0.8), int(len(w) * 0.9)
    return w[:a], w[a:b], w[b:]


def run(seeds=30, test_size=2000, split_seeds=(1, 2, 3, 4, 5, 7, 8, 42)):
    """
    Trains on both versions of the text and evaluates on held-out Shona words. One split is not
    enough, because at this corpus size the choice of split moves macro-F1 more than the typos
    do, so the comparison is repeated over several splits and the difference is averaged.
    """
    held_out = sample_unique(load_words(corpus_path("shona", "test")), test_size, seed=42)
    per_split, rows = [], []
    for sp in split_seeds:
        got = {}
        for label, fname in VARIANTS.items():
            train, val, _ = split_802010(words_of(MAT / fname), seed=sp)
            model = train_detection_model(train, "shona")
            clf = train_lr(model, val, seed=42)
            r = [evaluate_balanced_model(model, clf, held_out, s, min_word_length=1)
                 for s in range(seeds)]
            m, sd = mean_std([x["mf1"] for x in r])
            got[label] = {"mf1": m, "std": sd, "types": len(set(train)),
                          "trigrams": len(model.vocab),
                          "precision": sum(x["precision"] for x in r) / seeds,
                          "recall": sum(x["recall"] for x in r) / seeds}
        d = got["corrected"]["mf1"] - got["as typed"]["mf1"]
        per_split.append({"split_seed": sp,
                          "mf1_as_typed": round(got["as typed"]["mf1"], 4),
                          "mf1_corrected": round(got["corrected"]["mf1"], 4),
                          "difference": round(d, 4)})
        logger.info("split %-3s as typed %.4f  corrected %.4f  diff %+.4f",
                    sp, got["as typed"]["mf1"], got["corrected"]["mf1"], d)
        if sp == split_seeds[-1]:
            for label in VARIANTS:
                g = got[label]
                rows.append({"variant": label, "train_types": g["types"],
                             "char_trigrams": g["trigrams"],
                             "precision": round(g["precision"], 4),
                             "recall": round(g["recall"], 4),
                             "mf1_mean": round(g["mf1"], 4), "mf1_std": round(g["std"], 4),
                             "seeds": seeds})
    diffs = [p["difference"] for p in per_split]
    summary = {"n_splits": len(diffs), "mean_difference": round(sum(diffs) / len(diffs), 4),
               "sd_difference": round((sum((x - sum(diffs) / len(diffs)) ** 2
                                           for x in diffs) / len(diffs)) ** 0.5, 4),
               "corrected_better_in": sum(1 for x in diffs if x > 0)}
    logger.info("over %d splits: mean diff %+0.4f (sd %.4f); corrected better in %d",
                summary["n_splits"], summary["mean_difference"],
                summary["sd_difference"], summary["corrected_better_in"])
    write_csv(OUT / "shona_traintext_splits.csv", per_split)

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "shona_traintext.csv", rows)
    (OUT / "shona_traintext.json").write_text(json.dumps(
        {"split": "80/10/10, no corpus cleaning by design",
         "test_set": "held-out Shona corpus split", "n_test_words": len(held_out),
         "seeds": seeds, "per_split": per_split, "summary": summary,
         "variants_last_split": rows}, indent=2))

    x = np.arange(len(per_split)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - w/2, [p["mf1_as_typed"] for p in per_split], w,
           label="trained on the text as typed", color="#c0504d")
    ax.bar(x + w/2, [p["mf1_corrected"] for p in per_split], w,
           label="trained on the corrected text", color="#4f81bd")
    ax.set_xticks(x); ax.set_xticklabels([str(p["split_seed"]) for p in per_split])
    ax.set_xlabel("80/10/10 split seed"); ax.set_ylabel("macro-F1 on held-out Shona test set")
    ax.set_ylim(0.45, 0.60)
    ax.axhline(np.mean([p["mf1_as_typed"] for p in per_split]), color="#c0504d",
               ls=":", lw=1)
    ax.axhline(np.mean([p["mf1_corrected"] for p in per_split]), color="#4f81bd",
               ls=":", lw=1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    write_figure(fig, OUT, "shona_traintext")

    try:
        mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
        mlflow.set_experiment("Shona_TrainText_Demonstration")
        with mlflow.start_run(run_name="shona_traintext"):
            mlflow.log_params({"seeds": seeds, "n_test_words": len(held_out),
                               "n_splits": summary["n_splits"]})
            for r in rows:
                tag = r["variant"].replace(" ", "_")
                mlflow.log_metrics({f"mf1_{tag}": r["mf1_mean"], f"precision_{tag}": r["precision"],
                                    f"recall_{tag}": r["recall"],
                                    f"trigrams_{tag}": r["char_trigrams"]})
            mlflow.log_metrics({"mean_difference": summary["mean_difference"],
                                "sd_difference": summary["sd_difference"]})
    except Exception as e:
        logger.warning("mlflow logging skipped: %s", e)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--test-size", type=int, default=2000)
    a = ap.parse_args()
    run(a.seeds, a.test_size)
