"""
exp_udhr_transfer.py - cross-language transfer measured on the UDHR, a parallel text.

The UDHR is the same document in every language, so the test words match in content and a
difference in transfer reflects the languages and not the corpora. The model of the source
language is applied unchanged.

Three versions all write to results/udhr_transfer/<mode>-based/.
  v1  one source (zulu by default) on the Nguni languages and Shona, on the full text and on
      a sample of equal size per language
  v2  every pair of the six NCHLT languages, so the corpus source is held constant   (*_2.*)
  v3  every pair of eight languages, with NCHLT corpora where they exist            (*_3.*)

The transfer ratio of a pair is the macro-F1 of the source model on the target divided by its
macro-F1 on its own language. It is correlated with the Guthrie distance over the pairs of
different languages. The UDHR has only a few thousand words per language, so the results are
indicative.
"""

import argparse
import json
import logging
from pathlib import Path
import random
import sys
import urllib.request
import xml.etree.ElementTree as ET

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import (ROOT, LANG_LABELS, LANG_COLOURS, evaluate_balanced_checker, mean_std,
                        write_csv, disp)
from src.core.spell_checker import SpellChecker
from src.utils.text_utils import TextUtils
from src.utils.guthrie_distance import (guthrie_full_distance, guthrie_distance,
                                        guthrie_graph_distance, guthrie_graph_lexrank)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# UDHR ISO 639-3 codes, where an NCHLT code shares the text of its full-name language
UDHR_CODES = {
    "zulu": "zul", "xhosa": "xho", "ndebele": "nbl", "swati": "ssw", "shona": "sna",
    "zu": "zul", "xh": "xho", "nd": "nbl", "sw": "ssw",
    "sotho": "sot", "tswana": "tsn", "swahili": "swh", "lozi": "loz",
}
UDHR_BASE = "https://raw.githubusercontent.com/eric-muller/udhr/main/data/udhr/udhr_{code}.xml"
UDHR_CACHE = ROOT / "data" / "udhr_xml"

# v2 is NCHLT only, and v3 uses NCHLT where it exists and other sources elsewhere
V2_LANGS = ["zu", "xh", "nd", "sw", "sotho", "tswana"]
V3_LANGS = ["zu", "xh", "nd", "shona", "sotho", "tswana", "swahili", "lozi"]


def fetch_udhr(lang, dest):
    """Downloads the UDHR XML of a language to dest. Returns False when it is not available."""
    code = UDHR_CODES.get(lang)
    if not code:
        logger.warning("[%s] no UDHR code mapping", lang); return False
    url = UDHR_BASE.format(code=code)
    try:
        logger.info("[%s] fetching %s", lang, url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SpellChecker-Research)"})
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
    path = UDHR_CACHE / f"{lang}.xml"
    if path.exists() and path.stat().st_size < 100:
        path.unlink(); logger.info("[%s] deleted empty UDHR cache", lang)
    if not path.exists() and not fetch_udhr(lang, path):
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


def _run_singlesource(checker, source, targets, seeds, out, sample_mode="type", min_word_length=3):
    """Applies the model of one source language to the UDHR of each target (v1)."""
    order = [source] + [t for t in targets if t != source]
    per, tests = {}, {}
    for tgt in order:
        words = load_udhr_words(tgt)
        if not words or len(words) < 50:
            logger.warning("[%s] insufficient UDHR words (%s), skipping",
                        tgt, len(words) if words else 0); continue
        test = (list(dict.fromkeys(words)) if sample_mode == "type" else list(words))
        tests[tgt] = test
        runs = [evaluate_balanced_checker(checker, tgt, test, s, source_language=source,
                                        count_mode=sample_mode, min_word_length=min_word_length)
                for s in range(seeds)]
        m, sd = mean_std([r["mf1"] for r in runs])
        per[tgt] = {"label": LANG_LABELS.get(tgt, tgt), "udhr_code": UDHR_CODES.get(tgt),
                    "n_test_words": len(test), "mf1_mean": m, "mf1_std": sd,
                    "tp": float(np.mean([r["tp"] for r in runs])),
                    "fp": float(np.mean([r["fp"] for r in runs])),
                    "fn": float(np.mean([r["fn"] for r in runs])),
                    "tn": float(np.mean([r["tn"] for r in runs]))}
        logger.info("[v1 %s->%s] words=%d mF1=%.4f+/-%.4f", source, tgt, len(test), m, sd)
    if source not in per:
        logger.error("v1: source self-transfer missing"); return
    base = per[source]["mf1_mean"]
    for t in per:
        per[t]["transfer_ratio"] = per[t]["mf1_mean"] / base if base else 0.0

    per_sub = {}; min_n = min(len(tests[t]) for t in per)
    for tgt in order:
        if tgt not in tests:
            continue
        sub = random.Random(42).sample(tests[tgt], min_n)
        runs = [evaluate_balanced_checker(checker, tgt, sub, s, source_language=source,
                                        count_mode=sample_mode, min_word_length=min_word_length)
                for s in range(seeds)]
        m, sd = mean_std([r["mf1"] for r in runs])
        per_sub[tgt] = {"n_test_words": min_n, "mf1_mean": m, "mf1_std": sd}
    base_sub = per_sub.get(source, {}).get("mf1_mean", 0.0)
    for t in per_sub:
        per_sub[t]["transfer_ratio"] = per_sub[t]["mf1_mean"] / base_sub if base_sub else 0.0

    (out / "udhr_transfer.json").write_text(json.dumps(
        {"version": "v1_single_source", "source": source, "test_set": "UDHR (parallel)",
        "seeds": seeds, "baseline_mf1": base, "targets": per,
        "subsampled": {"n_per_language": min_n, "baseline_mf1": base_sub, "targets": per_sub}},
        indent=2))
    rows = [{"source": source, "target": t, "label": per[t]["label"],
            "full_n": per[t]["n_test_words"], "full_mf1": round(per[t]["mf1_mean"], 6),
            "full_ratio": round(per[t]["transfer_ratio"], 6), "sub_n": min_n,
            "sub_mf1": round(per_sub.get(t, {}).get("mf1_mean", 0.0), 6),
            "sub_ratio": round(per_sub.get(t, {}).get("transfer_ratio", 0.0), 6)}
            for t in order if t in per]
    write_csv(out / "udhr_transfer_summary.csv", rows)

    tgts = [t for t in order if t in per and t != source]
    if tgts:
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, t in enumerate(tgts):
            ax.bar(i, per[t]["transfer_ratio"], color=LANG_COLOURS.get(t, "#888"), alpha=0.88)
            ax.text(i, per[t]["transfer_ratio"] + 0.01, f"{per[t]['transfer_ratio']:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.axhline(1.0, color="#333", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(range(len(tgts)))
        ax.set_xticklabels([LANG_LABELS.get(t, t).split("(")[0].strip() for t in tgts],
                        rotation=20, ha="right")
        ax.set_ylabel("Transfer ratio  (target mF1 / source-on-own-UDHR)")
        ax.set_title(f"UDHR parallel-text transfer - source: {source} (baseline mF1={base:.3f})")
        ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
        fig.savefig(out / "udhr_transfer.pdf", dpi=150); plt.close(fig)

    print(f"\n{'='*70}\n  UDHR transfer v1 - single-source: {source}\n{'='*70}")
    print(f"  {'target':<24}{'words':>6}{'mF1':>8}{'ratio':>7}")
    for t in order:
        if t in per:
            r = per[t]
            print(f"  {r['label']:<24}{r['n_test_words']:>6}{r['mf1_mean']:>8.3f}"
                    f"{r['transfer_ratio']:>7.2f}")
    print(f"  baseline mF1 = {base:.3f}  (subsample n={min_n}, baseline {base_sub:.3f})\n")


def _run_allpairs(checker, langs, seeds, suffix, title, out, sample_mode="type", min_word_length=3):
    """Applies the model of every language to the UDHR of every other language (v2 and v3)."""
    langs = [l for l in langs if l in checker.detection_models]
    udhr = {}
    for l in langs:
        w = load_udhr_words(l)
        if w and len(w) >= 50:
            udhr[l] = (list(dict.fromkeys(w)) if sample_mode == "type" else list(w))
        else:
            logger.warning("[%s] insufficient UDHR, dropping from %s", l, title)
    langs = [l for l in langs if l in udhr]

    cell = {}
    for src in langs:
        for tgt in langs:
            runs = [evaluate_balanced_checker(checker, tgt, udhr[tgt], s, source_language=src,
                                            count_mode=sample_mode, min_word_length=min_word_length)
                    for s in range(seeds)]
            m, sd = mean_std([r["mf1"] for r in runs])
            mn = lambda k: float(np.mean([r[k] for r in runs]))
            cell[(src, tgt)] = {"mf1": m, "std": sd, "tp": mn("tp"), "fp": mn("fp"),
                                "fn": mn("fn"), "tn": mn("tn")}
            logger.info("[%s] %-7s -> %-7s mF1=%.4f", title.split()[0], src, tgt, m)

    rows, dists, ratios, mf1s, pt_src = [], [], [], [], []
    dists_alpha, dists_lex = [], []
    for src in langs:
        base = cell[(src, src)]["mf1"] or 1.0
        for tgt in langs:
            c = cell[(src, tgt)]; ratio = c["mf1"] / base
            zd, gd, ld = guthrie_distance(src, tgt); full = guthrie_full_distance(src, tgt)
            graph = guthrie_graph_distance(src, tgt)
            lex = guthrie_graph_lexrank(src, tgt)
            rows.append({"source": src, "target": tgt, "guthrie_zone_diff": zd,
                        "guthrie_group_diff": gd, "guthrie_lang_diff": ld,
                        "guthrie_graph_distance": graph, "guthrie_graph_lexrank": lex,
                        "guthrie_full_distance": full,
                        "tp": round(c["tp"], 1), "fp": round(c["fp"], 1),
                        "fn": round(c["fn"], 1), "tn": round(c["tn"], 1),
                        "mf1": round(c["mf1"], 6), "mf1_std": round(c["std"], 6),
                        "transfer_ratio": round(ratio, 6)})
            if src != tgt:
                dists.append(graph); ratios.append(ratio); mf1s.append(c["mf1"]); pt_src.append(src)
                dists_alpha.append(full); dists_lex.append(lex)

    def _sp(x, y): return spearmanr(x, y) if len(set(x)) > 1 else (float("nan"), float("nan"))
    rho_r, p_r = _sp(dists, ratios); rho_m, p_m = _sp(dists, mf1s)
    rho_alpha, p_alpha = _sp(dists_alpha, ratios)
    rho_lex, p_lex = _sp(dists_lex, ratios)

    write_csv(out / f"udhr_transfer_pairs{suffix}.csv", rows)
    write_csv(out / f"udhr_transfer_matrix{suffix}.csv",
              [{"source": s, **{t: round(cell[(s, t)]["mf1"], 6) for t in langs}} for s in langs])
    (out / f"udhr_transfer{suffix}.json").write_text(json.dumps(
        {"version": ("v2_nchlt6" if suffix == "_2" else "v3_metric8"), "langs": langs,
        "seeds": seeds,
        "distance": "guthrie_graph_weighted = 10*zone_hops + 3*group + 1*lang (zone-adjacency)",
        "spearman_dist_vs_transfer_ratio": {"rho": rho_r, "p": p_r, "n_pairs": len(dists)},
        "spearman_dist_vs_mf1": {"rho": rho_m, "p": p_m, "n_pairs": len(dists)},
        "spearman_alphabetical_vs_transfer_ratio": {"rho": rho_alpha, "p": p_alpha},
        "spearman_lexicographic_vs_transfer_ratio": {"rho": rho_lex, "p": p_lex},
        "matrix": {f"{s}->{t}": cell[(s, t)]["mf1"] for s in langs for t in langs}}, indent=2))

    M = np.array([[cell[(s, t)]["mf1"] for t in langs] for s in langs])
    fig, ax = plt.subplots(figsize=(1.15 * len(langs) + 2, 1.05 * len(langs) + 2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1)
    names = [disp(l) for l in langs]
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=11)
    ax.set_yticks(range(len(langs))); ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("target (tested on)", fontsize=12); ax.set_ylabel("source (trained on)", fontsize=12)
    for i in range(len(langs)):
        for j in range(len(langs)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=12, fontweight="bold")
    fig.colorbar(im, label="mF1")
    ax.set_title(f"UDHR parallel-text transfer: directed\nSpearman(dist, ratio) "
                f"rho={rho_r:.3f} p={p_r:.3f}, n={len(dists)}")
    fig.tight_layout(); fig.savefig(out / f"udhr_transfer_heatmap{suffix}.pdf", dpi=150); plt.close(fig)

    # distance against transfer ratio, coloured by source
    fig, ax = plt.subplots(figsize=(9, 5.5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]; cmap = plt.get_cmap("tab10")
    for i, l in enumerate(langs):
        xs = [d for d, s in zip(dists, pt_src) if s == l]
        ys = [r for r, s in zip(ratios, pt_src) if s == l]
        if xs:
            ax.scatter(xs, ys, color=cmap(i % 10), marker=markers[i % len(markers)], s=55,
                    alpha=0.85, edgecolor="white", linewidth=0.5, label=f"from {l}")
    if len(set(dists)) > 1:
        z = np.polyfit(dists, ratios, 1); gx = np.linspace(min(dists), max(dists), 100)
        ax.plot(gx, np.polyval(z, gx), "k--", alpha=0.45,
                label=f"trend (rho={rho_r:.3f}, p={p_r:.3f})")
    ax.set_xlabel("Guthrie zone-adjacency distance (10*zone_hops + 3*group + 1*lang)")
    ax.set_ylabel("Transfer ratio (mF1 / source self-mF1)")
    ax.set_title(f"UDHR parallel-text transfer vs Guthrie distance "
                f"({len(dists)} pairs)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2, loc="lower left", title="source")
    fig.tight_layout(); fig.savefig(out / f"udhr_transfer_scatter{suffix}.pdf", dpi=150); plt.close(fig)

    print(f"\n{'='*72}\n  UDHR transfer {title}  ({len(langs)} langs, {seeds} seeds)\n{'='*72}")
    print("  source\\target" + "".join(f"{t[:6]:>8}" for t in langs))
    for s in langs:
        print(f"  {s:<12}" + "".join(f"{cell[(s,t)]['mf1']:>8.3f}" for t in langs))
    print(f"\n  Spearman(full Guthrie dist, transfer ratio) rho={rho_r:.3f} p={p_r:.3f} "
            f"(n={len(dists)} off-diagonal pairs)")
    print(f"  Spearman(full Guthrie dist, mF1)            rho={rho_m:.3f} p={p_m:.3f}\n{'='*72}\n")

    tag = ("_v2" if suffix == "_2" else "_v3") + f"_{sample_mode}"
    mlflow.log_metrics({f"spearman_ratio_rho{tag}": rho_r, f"spearman_ratio_p{tag}": p_r,
                        f"n_pairs{tag}": len(dists)})
    return {"rho_ratio": rho_r, "p_ratio": p_r, "n_pairs": len(dists)}


def _run_mode(source, targets, seeds, out, sample_mode="type", min_word_length=3):
    checker = SpellChecker()
    checker.load_models(ROOT / "data" / "models")
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=== UDHR transfer v1 (single-source %s) ===", source)
    _run_singlesource(checker, source, targets, seeds, out, sample_mode, min_word_length)
    logger.info("=== UDHR transfer v2 (NCHLT-6 all-pairs) ===")
    r2 = _run_allpairs(checker, V2_LANGS, seeds, "_2", "v2 NCHLT-6 (corpus-controlled)",
                    out, sample_mode, min_word_length)
    logger.info("=== UDHR transfer v3 (metric-8 all-pairs, NCHLT-preferred) ===")
    r3 = _run_allpairs(checker, V3_LANGS, seeds, "_3", "v3 metric-8 (NCHLT-preferred)",
                    out, sample_mode, min_word_length)

    for f in sorted(out.glob("udhr_transfer*")):
        mlflow.log_artifact(str(f), artifact_path=f"{sample_mode}-based")
    logger.info("UDHR transfer done. v2 rho=%.3f (p=%.3f, n=%d) | v3 rho=%.3f (p=%.3f, n=%d)",
                r2["rho_ratio"], r2["p_ratio"], r2["n_pairs"],
                r3["rho_ratio"], r3["p_ratio"], r3["n_pairs"])


def run(source, targets, seeds, modes=("type", "token")):
    """Runs the three versions once per mode. Short words are scored (min_word_length=1)."""
    base = ROOT / "results" / "udhr_transfer"
    mlflow.set_experiment("udhr_transfer")
    with mlflow.start_run():
        mlflow.log_params({"experiment": "udhr_transfer", "source": source, "targets": targets,
                            "seeds": seeds, "v2_langs": V2_LANGS, "v3_langs": V3_LANGS,
                            "modes": ",".join(modes)})
        for mode in modes:
            _run_mode(source, targets, seeds, base / f"{mode}-based",
                        sample_mode=mode, min_word_length=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="zulu")
    ap.add_argument("--targets", nargs="+", default=["zulu", "xhosa", "ndebele", "swati", "shona"])
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    run(a.source, a.targets, a.seeds)
