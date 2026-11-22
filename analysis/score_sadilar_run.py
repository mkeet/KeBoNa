"""
score_sadilar_run.py - scores the run of the SADiLaR spelling checker on the fixed 500-word set.

The SADiLaR extension for Microsoft Word was run once by hand, on the clean words and on each
corrupted version. Each run file has one line per word, in the order of
results/rq3/rq3_500_groundtruth.csv.

    original_sadilar.txt   the clean words
    random_sadilar.txt     random corruption
    qwerty_sadilar.txt     QWERTY-adjacent corruption

A line ending in 0 is a word the extension flagged and a line ending in 1 is a word it
accepted. A line with no mark takes the opposite of the mark used in that file.

Outputs (results/rq3/):
    rq3_external_sadilar_summary.csv   tp, fp, fn, tn and macro-F1 per corruption type
    rq3_external_sadilar_words.csv     every word, clean and corrupted, with flag and outcome
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EVAL_DIR = ROOT / "results" / "rq3"
TYPES = ["random", "qwerty"]


def read_flags(path):
    """Returns, for each line of a run file, whether the extension flagged the word."""
    lines = [l.rstrip("\r\n").strip() for l in open(path, encoding="utf-8")]
    marks = [l.split()[-1] if len(l.split()) > 1 else None for l in lines]
    default = "1" if "0" in marks else "0"
    return [(m or default) == "0" for m in marks]


def mf1(tp, fp, fn, tn):
    """Returns precision, recall and F1 on incorrect words, F1 on correct words and macro-F1."""
    def f1(p, r):
        return 2 * p * r / (p + r) if p + r else 0.0
    p_e = tp / (tp + fp) if tp + fp else 0.0
    r_e = tp / (tp + fn) if tp + fn else 0.0
    p_c = tn / (tn + fn) if tn + fn else 0.0
    r_c = tn / (tn + fp) if tn + fp else 0.0
    f1_e, f1_c = f1(p_e, r_e), f1(p_c, r_c)
    return p_e, r_e, f1_e, f1_c, (f1_e + f1_c) / 2


def run(eval_dir=EVAL_DIR, out_dir=None):
    """Scores the three run files and writes the two CSVs."""
    eval_dir = Path(eval_dir)
    out_dir = Path(out_dir) if out_dir else eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = list(csv.DictReader(open(eval_dir / "rq3_500_groundtruth.csv", encoding="utf-8")))
    clean = read_flags(eval_dir / "original_sadilar.txt")
    words, summary = [], []
    for ctype in TYPES:
        corr = read_flags(eval_dir / f"{ctype}_sadilar.txt")
        assert len(clean) == len(corr) == len(gt)
        tp = fp = fn = tn = 0
        for g, fc, fx in zip(gt, clean, corr):
            oc = "fp" if fc else "tn"
            fp += fc; tn += not fc
            words.append([ctype, "clean", g["original"], "", 0, int(fc), oc])
            err = int(g[f"{ctype}_rate1.0_is_error"])
            oc = ("tp" if fx else "fn") if err else ("fp" if fx else "tn")
            tp += fx and err; fn += (not fx) and err
            words.append([ctype, "corrupted", g["original"], g[f"{ctype}_rate1.0_corrupted"], err, int(fx), oc])
        p, r, f1_e, f1_c, m = mf1(tp, fp, fn, tn)
        summary.append(["sadilar", ctype, 1.0, 2 * len(gt), tp, fp, fn, tn,
                        round(p, 6), round(r, 6), round(f1_e, 6), round(f1_c, 6), round(m, 6)])
    with open(out_dir / "rq3_external_sadilar_words.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["corruption_type", "instance", "original", "corrupted", "is_error", "flagged", "outcome"])
        w.writerows(words)
    with open(out_dir / "rq3_external_sadilar_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "corruption_type", "rate", "n_eval_words", "tp", "fp", "fn", "tn",
                    "incorrect_precision", "incorrect_recall", "incorrect_f1", "correct_f1", "mf1"])
        w.writerows(summary)
    for row in summary:
        print(f"{row[1]}: tp={row[4]} fp={row[5]} fn={row[6]} tn={row[7]} mF1={row[12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    run(out_dir=ap.parse_args().out_dir)
