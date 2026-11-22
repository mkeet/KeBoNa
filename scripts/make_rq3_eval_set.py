"""
make_rq3_eval_set.py - builds the fixed evaluation set of RQ3, so that the SADiLaR spelling
checker, KenLM and this checker all run on the same words.

The set holds 500 distinct isiZulu words, sampled with seed 42 from words of at least three
characters after bare English words are removed. Every word is corrupted once, under random
and under qwerty corruption. When rq3_500_originals.txt already exists its words are kept, and
only the corrupted files and the ground truth are written again.

Outputs (results/rq3/):
    rq3_500_originals.txt        the clean words
    rq3_500_<type>_rate<r>.txt   the corrupted words, in the same order
    rq3_500_groundtruth.csv      every word with each corrupted form and whether it is an error

The clean file gives the false positives and true negatives, and a corrupted file gives the
true positives and false negatives.
"""

import csv
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluation.spelling_wrecker import SpellingWrecker
from analysis.exp_common import load_words, corpus_path, MIN_WORD_LEN

LANG = "zulu"
N = 500
SEED = 42
RATES = [1.0]
TYPES = ["random", "qwerty"]

# bare English is removed before sampling, but a prefixed loan such as i-taxi is kept
_ENGLISH_DICTS = ["/usr/share/dict/american-english", "/usr/share/dict/british-english",
                "/usr/share/dict/words"]


def load_english():
    """
    Returns the English words of the system word lists, in lower case, of three letters or more.
    """
    eng = set()
    for f in _ENGLISH_DICTS:
        p = Path(f)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            w = line.strip().lower()
            if len(w) >= 3 and w.isalpha():
                eng.add(w)
    return eng


def is_bare_english(word, eng):
    """Returns True when the whole word is an English dictionary word."""
    lw = word.lower()
    return lw in eng or lw.replace("-", "") in eng


def sample_clean(all_words, n, eng, seed=SEED):
    """Samples n distinct test words after the bare English words are removed."""
    unique = list(dict.fromkeys(all_words))
    clean = [w for w in unique if len(w) >= MIN_WORD_LEN and not is_bare_english(w, eng)]
    if len(clean) < n:
        raise ValueError(f"only {len(clean)} clean words available, need {n}")
    return random.Random(seed).sample(clean, n)


def corrupt_aligned(originals, rate, ctype):
    """
    Corrupts the words one by one with a single generator, so that the output keeps the order
    of the originals. Returns each corrupted form with whether it differs from the original.
    """
    wr = SpellingWrecker(corruption_rate=rate, seed=SEED, corruption_type=ctype)
    out = []
    for w in originals:
        c, gt = wr.corrupt_text(w, min_word_length=MIN_WORD_LEN)
        c = c.strip()
        out.append((c, 1 if c in gt else 0))
    return out


def main():
    """Builds the set and writes the three kinds of file."""
    out_dir = ROOT / "results" / "rq3"
    out_dir.mkdir(parents=True, exist_ok=True)

    originals_path = out_dir / "rq3_500_originals.txt"
    if originals_path.exists():
        originals = load_words(originals_path)
        print(f"originals: {len(originals)} words read from {originals_path.name}")
    else:
        eng = load_english()
        originals = sample_clean(load_words(corpus_path(LANG, "test")), N, eng)
        print(f"english wordlist: {len(eng)} entries loaded; bare-English removed before sampling "
            f"(prefixed loans kept)")
        originals_path.write_text("\n".join(originals) + "\n", encoding="utf-8")
        print(f"originals: {len(originals)} words -> {originals_path.name}")

    variants = {}
    for ctype in TYPES:
        for rate in RATES:
            rows = corrupt_aligned(originals, rate, ctype)
            variants[(ctype, rate)] = rows
            fname = f"rq3_500_{ctype}_rate{rate}.txt"
            (out_dir / fname).write_text(
                "\n".join(c for c, _ in rows) + "\n", encoding="utf-8")
            n_err = sum(e for _, e in rows)
            print(f"  {ctype:<7} rate={rate}: {n_err}/{len(rows)} corrupted -> {fname}")

    gt_path = out_dir / "rq3_500_groundtruth.csv"
    fields = ["idx", "original"]
    for ctype in TYPES:
        for rate in RATES:
            fields += [f"{ctype}_rate{rate}_corrupted", f"{ctype}_rate{rate}_is_error"]
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, og in enumerate(originals):
            row = {"idx": i + 1, "original": og}
            for ctype in TYPES:
                for rate in RATES:
                    c, e = variants[(ctype, rate)][i]
                    row[f"{ctype}_rate{rate}_corrupted"] = c
                    row[f"{ctype}_rate{rate}_is_error"] = e
            w.writerow(row)
    print(f"ground truth -> {gt_path}")
    print(f"\nAll files in: {out_dir}")


if __name__ == "__main__":
    main()
