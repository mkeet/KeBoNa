"""
exp_c7_orthography_udhr.py - the effect of orthography, measured on the UDHR.

The trained models are run on the UDHR text of each language as a test set. The document is
the same in every language, so the content is controlled and a difference in macro-F1 comes
from the way the language is written.

    v1  zulu, xhosa, ndebele, swati            (written conjunctively)
    v2  v1 with sotho and tswana               (adds languages written disjunctively)
    v3  v2 with swahili and lozi

For each language the mean word length, the mean number of trigrams per word and the macro-F1
on the UDHR are given.

Outputs (results/c7/, with suffix _2 and _3):
    c7_orthography.json / .csv   statistics and macro-F1 per language, and the group means
    c7_words_seed0.csv           seed 0 word by word
    c7_orthography.pdf           macro-F1 by language, grouped by orthography
    c7_trigrams.pdf              mean trigrams per word
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import urllib.request
import xml.etree.ElementTree as ET

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import ROOT, evaluate_balanced_checker, audit_rows, mean_std, write_csv
from src.core.spell_checker import SpellChecker
from src.utils.text_utils import TextUtils
from src.utils.trigram_tokenizer import CharTrigramTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Kiswahili is written conjunctively like the Nguni languages, and Silozi is related to Sesotho
AGGLUTINATIVE = ["zulu", "xhosa", "ndebele", "swati", "swahili"]
DISJUNCTIVE   = ["sotho", "tswana", "lozi"]

NGUNI    = ["zulu", "xhosa", "ndebele", "swati"]
V1_LANGS = NGUNI
V2_LANGS = NGUNI + ["sotho", "tswana"]
V3_LANGS = NGUNI + ["sotho", "tswana", "swahili", "lozi"]

C7_NAMES = {"zulu": "isiZulu", "xhosa": "isiXhosa", "ndebele": "isiNdebele", "swati": "siSwati",
            "sotho": "Sesotho", "tswana": "Setswana", "swahili": "Kiswahili", "lozi": "siLozi"}
C7_COLOURS = {"zulu": "#1f77b4", "xhosa": "#d62728", "ndebele": "#2ca02c", "swati": "#9467bd",
                "sotho": "#ff7f0e", "tswana": "#8c564b", "swahili": "#e377c2", "lozi": "#17becf"}

UDHR_CODES = {
    "zulu": "zul", "xhosa": "xho", "ndebele": "nbl", "swati": "ssw",
    "sotho": "sot", "tswana": "tsn", "shona": "sna",
    "swahili": "swh", "lozi": "loz",
    "sepedi": "nso", "setswana": "tsn", "sesotho": "sot", "english": "eng",
}
UDHR_BASE = "https://raw.githubusercontent.com/eric-muller/udhr/main/data/udhr/udhr_{code}.xml"


def fetch_udhr(lang, dest):
    """Downloads the UDHR XML of a language to dest. Returns False when it is not available."""
    code = UDHR_CODES.get(lang)
    if not code:
        logger.warning("[%s] no UDHR code mapping", lang); return False
    url = UDHR_BASE.format(code=code)
    try:
        logger.info("[%s] fetching %s", lang, url)
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (SpellChecker-Research)"})
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode("utf-8", errors="replace")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        logger.info("[%s] fetched %d chars", lang, len(text))
        return True
    except Exception as e:
        logger.warning("[%s] fetch failed (%s): %s", lang, url, e); return False


def _para_text(xml_str):
    """Returns the text of every paragraph of a UDHR XML document."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        logger.warning("XML parse error: %s", e); return ""
    return "\n".join("".join(el.itertext()) for el in root.iter()
                    if el.tag.split("}")[-1] == "para")


def load_udhr_words(lang):
    """
    Returns the words of the UDHR in a language, downloading the text on first use, or None
    when the text is not available.
    """
    path = ROOT / "data" / "udhr" / f"{lang}.xml"
    if path.exists() and path.stat().st_size < 100:
        path.unlink(); logger.info("[%s] deleted empty UDHR cache, will re-fetch", lang)
    if not path.exists():
        if not fetch_udhr(lang, path):
            return None
    body = _para_text(path.read_text(encoding="utf-8"))
    words = []
    for line in body.splitlines():
        line = line.strip()
        if not line or TextUtils.is_metadata_line(line):
            continue
        for tok in TextUtils.tokenize(TextUtils.normalize(line)):
            if tok and tok[0].isalpha() and not any(c.isdigit() for c in tok):
                words.append(tok)
    return words


def structure_stats(words):
    """Returns the mean word length and the mean number of trigrams per word."""
    tok     = CharTrigramTokenizer(n=3)
    lengths = [len(w) for w in words]
    tris    = [len(tok.segment_word(w)) for w in words]
    return {"n_words": len(words),
            "avg_word_len": float(np.mean(lengths)) if lengths else 0.0,
            "avg_trigrams_per_word": float(np.mean(tris)) if tris else 0.0}


def _eval_language(checker, lang, seeds, sample_mode="type", min_word_length=3):
    """Applies the model and classifier of a language to its UDHR text, over the seeds."""
    if lang not in checker.detection_models:
        logger.warning("[%s] no trained model loaded, skipping", lang); return None
    words = load_udhr_words(lang)
    if not words or len(words) < 50:
        logger.warning("[%s] insufficient UDHR words (%s), skipping",
                    lang, len(words) if words else 0); return None
    test = (list(dict.fromkeys(words)) if sample_mode == "type" else list(words))
    per = [evaluate_balanced_checker(checker, lang, test, s,
                                    count_mode=sample_mode, min_word_length=min_word_length)
                                    for s in range(seeds)]

    m, sd = mean_std([r["mf1"] for r in per])
    mn = lambda k: float(np.mean([r[k] for r in per]))
    return {"mf1_mean": m, "mf1_std": sd, "n_test_unique": len(test),
            "tp": mn("tp"), "fp": mn("fp"), "fn": mn("fn"), "tn": mn("tn"),
            "incorrect_f1": mn("f1_error"), "correct_f1": mn("f1_correct"),
            **structure_stats(words)}


def _run_version(checker, langs, seeds, suffix, title, out, sample_mode="type", min_word_length=3):
    """Evaluates one language set and writes its JSON, CSVs and charts."""
    results, audit = {}, []
    for lang in langs:
        r = _eval_language(checker, lang, seeds, sample_mode, min_word_length)
        if r is None:
            continue
        r["orthography"] = "agglutinative" if lang in AGGLUTINATIVE else "disjunctive"
        results[lang] = r
        logger.info("[%s] %-6s %-13s len=%.1f tri/w=%.1f mF1=%.3f+/-%.3f (n=%d)",
                    title.split()[0], lang, r["orthography"], r["avg_word_len"],
                    r["avg_trigrams_per_word"], r["mf1_mean"], r["mf1_std"],
                    r["n_test_unique"])
        # seed 0 word by word
        test = list(dict.fromkeys(load_udhr_words(lang)))
        rows, _, _ = audit_rows(checker.detection_models[lang],
                                checker.threshold_classifiers.get(lang),
                                test, seed=0, extra={"language": lang,
                                "orthography": r["orthography"]})
        audit.extend(rows)

    def grp(g):
        v = [r["mf1_mean"] for r in results.values() if r["orthography"] == g]
        return float(np.mean(v)) if v else None
    group_summary = {"agglutinative_mean_mf1": grp("agglutinative"),
                    "disjunctive_mean_mf1":   grp("disjunctive")}

    (out / f"c7_orthography{suffix}.json").write_text(json.dumps(
        {"version": ("v2" if suffix else "v1"), "per_language": results,
        "group_summary": group_summary}, indent=2))
    csv_rows = [{"version": ("v2" if suffix else "v1"), "language": l,
                "orthography": r["orthography"], "n_udhr_words": r["n_words"],
                "n_test_unique": r["n_test_unique"],
                "avg_word_len": round(r["avg_word_len"], 3),
                "avg_trigrams_per_word": round(r["avg_trigrams_per_word"], 3),
                "tp": round(r["tp"], 1), "fp": round(r["fp"], 1),
                "fn": round(r["fn"], 1), "tn": round(r["tn"], 1),
                "incorrect_f1": round(r["incorrect_f1"], 6),
                "correct_f1": round(r["correct_f1"], 6),
                "mf1_mean": round(r["mf1_mean"], 6), "mf1_std": round(r["mf1_std"], 6)}
                for l, r in results.items()]
    write_csv(out / f"c7_orthography{suffix}.csv", csv_rows)
    write_csv(out / f"c7_words_seed0{suffix}.csv", audit)

    if results:
        agg  = [l for l in langs if l in results and results[l]["orthography"] == "agglutinative"]
        disj = [l for l in langs if l in results and results[l]["orthography"] == "disjunctive"]
        ordered = agg + disj
        x = np.arange(len(ordered))
        xlabels = [C7_NAMES.get(l, l) for l in ordered]
        sep = len(agg) - 0.5

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, lang in enumerate(ordered):
            r = results[lang]; col = C7_COLOURS.get(lang, "#999")
            ax.bar(i, r["mf1_mean"], yerr=r["mf1_std"], capsize=4, color=col, alpha=0.88)
            ax.text(i, r["mf1_mean"] + r["mf1_std"] + 0.008, f"{r['mf1_mean']:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        if agg and disj:
            ax.axvline(sep, color="#333", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=20, ha="right")
        ax.set_ylabel("Macro F1 (real model on UDHR test)", fontsize=11)
        ax.set_title("mF1 by language (UDHR test set)"
                    + ("  —  agglutinative | disjunctive" if (agg and disj) else ""),
                    fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05); ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(out / f"c7_orthography{suffix}.pdf", dpi=150)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for i, lang in enumerate(ordered):
            r = results[lang]; col = C7_COLOURS.get(lang, "#999")
            ax2.bar(i, r["avg_trigrams_per_word"], color=col, alpha=0.88)
            ax2.text(i, r["avg_trigrams_per_word"] + 0.05,
                    f"{r['avg_trigrams_per_word']:.1f}", ha="center", va="bottom", fontsize=9)
        if agg and disj:
            ax2.axvline(sep, color="#333", linewidth=1.2, linestyle="--", alpha=0.6)
        ax2.set_xticks(x); ax2.set_xticklabels(xlabels, rotation=20, ha="right")
        ax2.set_ylabel("Mean character trigrams per word", fontsize=11)
        ax2.set_title("Sub-word signal by language (UDHR test set)"
                    + ("  —  agglutinative | disjunctive" if (agg and disj) else ""),
                    fontsize=12, fontweight="bold")
        ax2.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout(); fig2.savefig(out / f"c7_trigrams{suffix}.pdf", dpi=150)
        plt.close(fig2)

    print(f"\n{'='*74}\n  C7 {title} - real models on UDHR test set\n{'='*74}")
    print(f"  {'lang':<10}{'ortho':<15}{'words':>7}{'wlen':>6}{'tri/w':>7}"
            f"{'incF1':>8}{'corF1':>8}{'mF1':>8}{'+/-':>7}")
    ordered = [l for l in langs if l in results]

    for l in ordered:
        r = results[l]
        print(f"  {l:<10}{r['orthography']:<15}{r['n_test_unique']:>7}"
            f"{r['avg_word_len']:>6.1f}{r['avg_trigrams_per_word']:>7.1f}"
            f"{r['incorrect_f1']:>8.3f}{r['correct_f1']:>8.3f}"
            f"{r['mf1_mean']:>8.3f}{r['mf1_std']:>7.3f}")
    print(f"\n  agglutinative mean mF1 = {group_summary['agglutinative_mean_mf1']}")
    print(f"  disjunctive   mean mF1 = {group_summary['disjunctive_mean_mf1']}")
    print(f"{'='*74}\n")

    tag = ("_v1" if not suffix else "_v2") + f"_{sample_mode}"
    mlflow.log_metrics({f"agglutinative_mf1{tag}": group_summary["agglutinative_mean_mf1"] or 0.0,
                        f"disjunctive_mf1{tag}": group_summary["disjunctive_mean_mf1"] or 0.0})
    return results


def _run_mode(seeds, out, sample_mode="type", min_word_length=3, v1_langs=None, v2_langs=None, v3_langs=None):

    checker = SpellChecker()
    checker.load_models(ROOT / "data" / "models")
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=== C7 agglutinative ===")
    _run_version(checker, v1_langs or V1_LANGS, seeds, "", "agglutinative", out,
                sample_mode, min_word_length)

    logger.info("=== C7 incl. disjunctive ===")
    _run_version(checker, v2_langs or V2_LANGS, seeds, "_2", "incl. disjunctive", out,
                sample_mode, min_word_length)

    logger.info("=== C7 full metric set ===")
    _run_version(checker, v3_langs or V3_LANGS, seeds, "_3", "full metric set", out,
                sample_mode, min_word_length)

    for f in sorted(out.glob("c7_*")):
        mlflow.log_artifact(str(f))


def run(seeds, v1_langs=None, v2_langs=None, v3_langs=None):
    """
    Runs the three versions on the tokens of the UDHR. Short words are scored (min_word_length=1).
    """
    base = ROOT / "results" / "c7"
    mlflow.set_experiment("udhr_analysis")
    with mlflow.start_run():
        mlflow.log_params({"experiment": "c7_udhr_realmodels", "seeds": seeds, "mode": "token"})
        _run_mode(seeds, base, sample_mode="token", min_word_length=1,
                v1_langs=v1_langs, v2_langs=v2_langs, v3_langs=v3_langs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--v1-langs", nargs="+", default=None)
    ap.add_argument("--v2-langs", nargs="+", default=None)
    ap.add_argument("--v3-langs", nargs="+", default=None)
    a = ap.parse_args()
    run(a.seeds, a.v1_langs, a.v2_langs, a.v3_langs)
