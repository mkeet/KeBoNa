"""
plot_llm_comparison.py - macro-F1 of the spell checker and every language model on the same 500
isiZulu words.

    python analysis/plot_llm_comparison.py

Reads the per-word results of the spell checker and the models run by exp_c9_llm_comparison.py
(results/c9/c9_reasoned_words_zu_local.csv and _cloud.csv) and the replies of Claude Opus 5.5 and
GPT-6.2 Astra (results/c9/paid_models/). It first checks that every run used exactly the words and
corruptions of results/c9/paid_models/words.csv, then scores each system as exp_c9_llm_comparison.py
does.

Outputs (results/c9/): llm_comparison.csv, llm_comparison.pdf, llm_comparison.png
"""

import csv
import json
import re
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
C9 = ROOT / "results" / "c9"
PAID = C9 / "paid_models"

# (label, source, column or model tag, colour)
SYSTEMS = [
    ("Spell checker", "local", "spellchecker", "#d62728"),
    ("Claude Opus 5.5", "json", "claude-opus-5.5-medium", "#4e79a7"),
    ("GPT-6.2 Astra", "json", "gpt-6.2-astra-high", "#59a14f"),
    ("GPT-OSS-120B", "cloud", "gpt-oss-120b", "#b07aa1"),
    ("Gemma 4 12B", "local", "gemma4:12b", "#f28e2b"),
    ("gpt-oss:20b", "local", "gpt-oss:20b", "#76b7b2"),
    ("AfriqueQwen 3.5 9B", "local", "AfriqueQwen3.5-9B-50Langs-Instruct-v1-i1-GGUF:Q4_K_M", "#9c755f"),
]


def numbered_words(path):
    """Returns the words listed in a prompt file, in order."""
    return [m.group(1) for m in re.finditer(r"^\d+\. (.+)$", path.read_text(encoding="utf-8"), re.M)]


def check_words(words):
    """Stops with an error unless every run used the words and corruptions of words.csv."""
    original = [r["original"] for r in words]
    corrupted = [r["corrupted"] for r in words]
    for name in ("c9_reasoned_words_zu_local.csv", "c9_reasoned_words_zu_cloud.csv"):
        rows = list(csv.DictReader(open(C9 / name, encoding="utf-8")))
        clean = [r["word"] for r in rows if r["instance"] == "clean"]
        corr = [r["word"] for r in rows if r["instance"] != "clean"]
        labels = [r["is_error"] for r in rows if r["instance"] != "clean"]
        assert clean == original and corr == corrupted, f"{name}: different words"
        assert labels == [r["is_error"] for r in words], f"{name}: different labels"
    assert numbered_words(PAID / "prompt_original.txt") == original, "prompt_original.txt: different words"
    assert numbered_words(PAID / "prompt_corrupted.txt") == corrupted, "prompt_corrupted.txt: different words"
    print("All runs used the same 500 words and the same corruptions.")


def flags_from_csv(name, column):
    """Returns the flags of one system on the original and on the corrupted words."""
    rows = list(csv.DictReader(open(C9 / f"c9_reasoned_words_zu_{name}.csv", encoding="utf-8")))
    clean = [int(r[f"{column}_flag"]) for r in rows if r["instance"] == "clean"]
    corr = [int(r[f"{column}_flag"]) for r in rows if r["instance"] != "clean"]
    return clean, corr


def flags_from_json(tag):
    """Returns the flags of one model from its two JSON replies; a missing key counts as not flagged."""
    out = []
    for part in ("original", "corrupted"):
        items = json.load(open(PAID / f"prompt_{part}.{tag}.json", encoding="utf-8"))
        assert len(items) == 500, f"{tag} {part}: {len(items)} answers"
        flags = [0] * 500
        for i, obj in enumerate(items):
            flags[obj.get("n", i + 1) - 1] = int(bool(obj.get("misspelled")))
        out.append(flags)
    return out


def score(clean, corr, words):
    """Returns TP, FP, FN, TN and macro-F1, counted as in the original experiment."""
    tp = fp = fn = tn = 0
    for f in clean:
        if f: fp += 1
        else: tn += 1
    for f, row in zip(corr, words):
        err = row["is_error"] == "1"
        if err and f: tp += 1
        elif f: fp += 1
        elif err: fn += 1
        else: tn += 1

    def f1(p, r):
        return 2 * p * r / (p + r) if p + r else 0.0
    f1_e = f1(tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0)
    f1_c = f1(tn / (tn + fn) if tn + fn else 0.0, tn / (tn + fp) if tn + fp else 0.0)
    return tp, fp, fn, tn, (f1_e + f1_c) / 2


def main():
    words = list(csv.DictReader(open(PAID / "words.csv", encoding="utf-8")))
    check_words(words)

    results = []
    for label, source, key, colour in SYSTEMS:
        clean, corr = flags_from_json(key) if source == "json" else flags_from_csv(source, key)
        results.append((label, colour) + score(clean, corr, words))
    results.sort(key=lambda r: -r[6])

    with open(C9 / "llm_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "TP (misspelled words flagged)", "FP (correct words flagged)",
                    "FN (misspelled words accepted)", "TN (correct words accepted)", "macro_F1"])
        for label, _, tp, fp, fn, tn, m in results:
            w.writerow([label, tp, fp, fn, tn, round(m, 4)])
            print(f"{label:20s} TP={tp} FP={fp} FN={fn} TN={tn}  macro-F1={m:.3f}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (label, colour, *_, m) in enumerate(results):
        ax.bar(i, m, color=colour)
        ax.text(i, m + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r[0] for r in results], rotation=25, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection on the same 500 isiZulu words and their corrupted copies")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(C9 / "llm_comparison.pdf")
    fig.savefig(C9 / "llm_comparison.png", dpi=200)
    print(f"Written to {C9}: llm_comparison.csv, .pdf, .png")


if __name__ == "__main__":
    main()
