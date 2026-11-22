"""
exp_rq2_cross_language.py - cross-language transfer for every pair of languages, against their
Guthrie distance (RQ2).

The model of each source language is applied to the test words of each target language. Six
language sets run in one pass and all write to results/rq2/<mode>-based/. The version labels
v1 to v6 appear in the file names and in the version column of the CSVs.

    v1  zulu, xhosa, ndebele, swati, shona
    v2  zu, xh, nd, shona, sotho, tswana, swahili, lozi
    v3  v1 + sotho, tswana, swahili, lozi
    v4  v2 + sw
    v5  v4 + cross-zone languages spanning C/E/H/J/M/N/R
    v6  v5 minus the high-short-word languages

The distance is 10*zone + 3*group + 1*language, where the zone term is the number of zone
borders crossed.

Outputs per version (suffix _2.._6):
    rq2_transfer.json            languages, Spearman results, full matrix
    rq2_transfer_summary.csv     one row per directed pair
    rq2_matrix.csv               NxN mF1 matrix (rows=source, cols=target)
    rq2_correlation_matrix.csv   symmetric similarity matrix
    rq2_words_seed0.csv          per-word audit, seed 0
    rq2_transfer.pdf             heatmap
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import (ROOT, load_words, corpus_path, sample_unique, sample_tokens,
                        evaluate_balanced_checker, mean_std, write_csv, audit_rows,
                        disp, write_figure)
from src.core.spell_checker import SpellChecker
from src.utils.guthrie_distance import (guthrie_full_distance, guthrie_distance,
                                        guthrie_graph_distance, guthrie_graph_lexrank,
                                        guthrie_graph_components, parse_code, GUTHRIE_CODES,
                                        _ZONE_ADJ)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

V1_LANGS = ["zulu", "xhosa", "ndebele", "swati", "shona"]

V2_LANGS = ["zu", "xh", "nd", "shona", "sotho", "tswana", "swahili", "lozi"]

V4_LANGS = ["zu", "xh", "nd", "sw", "shona", "sotho", "tswana", "swahili", "lozi"]

V3_LANGS = ["zulu", "xhosa", "ndebele", "swati", "shona", "sotho", "tswana", "swahili", "lozi"]

# v5 is v4 plus the languages of the other zones, and a language with no trained model is skipped
V5_LANGS = ["zu", "xh", "nd", "sw", "shona", "sotho", "tswana", "swahili", "lozi",
            "sepedi", "venda", "tsonga",
            "kinyarwanda", "kirundi", "luganda", "runyankore",
            "kikuyu", "kamba", "chichewa", "tumbuka", "bemba",
            "lingala", "kongo", "umbundu", "herero",
            "luba", "yao", "kimbundu", "lumbu"]

# v6 drops the languages in which more than about 20% of tokens have fewer than three characters
_HIGH_SHORTWORD = {"sotho", "sesotho", "tswana", "setswana", "lozi", "silozi", "sepedi", "pedi",
                    "lingala", "kongo", "kikongo", "kimbundu"}
V6_LANGS = [l for l in V5_LANGS if l not in _HIGH_SHORTWORD]


# positions of the zones in the zone-adjacency figure, roughly geographic
_ZONE_POS = {"A": (0.03, 2.94), "B": (0.03, 2.53), "C": (0.85, 2.74), "D": (1.68, 2.78),
            "J": (2.34, 2.53), "E": (3.07, 2.78), "G": (3.49, 2.24), "F": (2.71, 2.12),
            "L": (1.27, 2.12), "H": (0.35, 1.91), "M": (2.09, 1.71), "P": (3.07, 1.71),
            "K": (0.94, 1.42), "N": (2.25, 1.18), "R": (0.53, 0.89), "S": (1.43, 0.48)}


def _zone_graph_figure(out, used_zones=None, print_size=False):
    """
    Draws the zone-adjacency graph on which the distance is measured. A node is a Guthrie zone
    and an edge is a border shared on the map. With print_size the figure has no title and
    larger type.
    """
    used = set(used_zones or [])
    fs, nd, txt = ((7.5, 4.4), 13, 10) if print_size else ((9.5, 6.0), 12, 12)
    fig, ax = plt.subplots(figsize=fs)
    seen = set()
    for z, nbrs in _ZONE_ADJ.items():
        for nb in nbrs:
            if (nb, z) in seen:
                continue
            seen.add((z, nb))
            x1, y1 = _ZONE_POS[z]; x2, y2 = _ZONE_POS[nb]
            ax.plot([x1, x2], [y1, y2], "-", color="#9aa5b1", linewidth=1.2, zorder=1)
    for z, (x, y) in _ZONE_POS.items():
        on = z in used
        ax.scatter([x], [y], s=760, zorder=2, edgecolor="#36454F", linewidth=1.4,
                color="#4e79a7" if on else "#e8ecf1")
        ax.text(x, y, z, ha="center", va="center", fontsize=nd, fontweight="bold",
                color="white" if on else "#36454F", zorder=3)
    ax.text(0.01, 0.02, r"$d(x,y) = 10\,h + 3\,|\Delta_{group}| + 1\,|\Delta_{lang}|$"
            "\n" r"$h$ = borders crossed (shortest path, BFS)",
            transform=ax.transAxes, fontsize=txt, va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f8", edgecolor="#9aa5b1"))
    if used:
        ax.scatter([], [], s=180, color="#4e79a7", edgecolor="#36454F",
                label="zones covered by this study")
        ax.scatter([], [], s=180, color="#e8ecf1", edgecolor="#36454F", label="other Bantu zones")
        ax.legend(loc="lower right", fontsize=txt, frameon=False)
    if not print_size:
        ax.set_title("Guthrie zone-adjacency graph — the geography behind the distance\n"
                    "(e.g. S→R = 1 border, S→K = 2 via R, S→G = 3 via N and P)",
                    fontsize=12, fontweight="bold")
    ax.set_xlim(-0.35, 3.9); ax.set_ylim(-0.55, 3.3); ax.axis("off")
    fig.tight_layout()
    write_figure(fig, out, "guthrie_zone_graph" + ("_print" if print_size else ""))
    plt.close(fig)


def _dist_heatmap(langs, dist_fn, out_path, title, cmap="YlOrRd", as_rank=False, annot_fn=None):
    """
    Draws the matrix of one distance definition as a heatmap. With as_rank the cells are
    coloured by rank and not by value, for the lexicographic distance, whose key is an order
    and not a size. annot_fn replaces the text of each cell.
    """
    n = len(langs)
    D = np.array([[0.0 if s == t else float(dist_fn(s, t) or 0) for t in langs]
                for s in langs])
    if as_rank:
        rank = {v: i for i, v in enumerate(sorted(set(D.flatten().tolist())))}
        D = np.array([[rank[D[i, j]] for j in range(n)] for i in range(n)], dtype=float)
    vmax = D.max() or 1.0
    names = [disp(l) for l in langs]
    fig, ax = plt.subplots(figsize=(1.15 * n + 2, 1.05 * n + 2))
    im = ax.imshow(D, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("target", fontsize=12); ax.set_ylabel("source", fontsize=12)
    afs = 7 if annot_fn else 9
    for i in range(n):
        for j in range(n):
            txt = annot_fn(langs[i], langs[j]) if annot_fn else f"{D[i, j]:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=afs,
                    color="black" if D[i, j] < 0.6 * vmax else "white")
    fig.colorbar(im, label="rank order" if as_rank else "distance")
    ax.set_title(title, fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _run_version(checker, langs, seeds, test_size, suffix, title, out, version, mode="type"):
    langs = [l for l in langs
            if l in checker.detection_models and corpus_path(l, "test").exists()]
    _sample = sample_tokens if mode == "token" else sample_unique
    words = {l: _sample(load_words(corpus_path(l, "test")), test_size, seed=42)
            for l in langs}
    cmode = "token" if mode == "token" else "type"
    # short words are scored in both modes
    min_wl = 1

    # macro-F1 of every source on every target, as the mean over the seeds
    cell = {}
    for src in langs:
        for tgt in langs:
            res = [evaluate_balanced_checker(checker, tgt, words[tgt], s,
                                            source_language=src, count_mode=cmode,
                                            min_word_length=min_wl)
                    for s in range(seeds)]
            m, sd = mean_std([r["mf1"] for r in res])
            mn = lambda k: float(np.mean([r[k] for r in res]))
            tp, fp, fn, tn = mn("tp"), mn("fp"), mn("fn"), mn("tn")
            p_e, r_e = mn("precision"), mn("recall")
            f1_e, f1_c = mn("f1_error"), mn("f1_correct")
            p_c = tn / (tn + fn) if tn + fn else 0.0
            r_c = tn / (tn + fp) if tn + fp else 0.0
            cell[(src, tgt)] = {"mf1": m, "std": sd, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                "ip": p_e, "ir": r_e, "if1": f1_e,
                                "cp": p_c, "cr": r_c, "cf1": f1_c}
            logger.info("%-8s -> %-8s  mF1=%.4f +/- %.4f", src, tgt, m, sd)

    # one row per directed pair, with the weighted distance in dists_g and the lexicographic one in dists_lex
    summary_rows, dists, dists_g, dists_lex, ratios, mf1s, pt_src = [], [], [], [], [], [], []
    for src in langs:
        base = cell[(src, src)]["mf1"] or 1.0
        for tgt in langs:
            c = cell[(src, tgt)]; ratio = c["mf1"] / base
            zd, gd, ld = guthrie_distance(src, tgt)
            full = guthrie_full_distance(src, tgt)
            hops = guthrie_graph_components(src, tgt)[0]
            gdist = guthrie_graph_distance(src, tgt)
            lexr = guthrie_graph_lexrank(src, tgt)
            summary_rows.append({
                "version": version, "source": src, "target": tgt,
                "guthrie_zone_diff": zd, "guthrie_group_diff": gd, "guthrie_lang_diff": ld,
                "guthrie_full_distance": full,
                "zone_hops": hops, "guthrie_graph_distance": gdist,
                "guthrie_graph_lexrank": lexr,
                "tp": round(c["tp"], 1), "fp": round(c["fp"], 1),
                "fn": round(c["fn"], 1), "tn": round(c["tn"], 1),
                "incorrect_precision": round(c["ip"], 6), "incorrect_recall": round(c["ir"], 6),
                "incorrect_f1": round(c["if1"], 6),
                "correct_precision": round(c["cp"], 6), "correct_recall": round(c["cr"], 6),
                "correct_f1": round(c["cf1"], 6),
                "mf1": round(c["mf1"], 6), "mf1_std": round(c["std"], 6),
                "transfer_ratio": round(ratio, 6)})
            if src != tgt:
                dists.append(full); dists_g.append(gdist); dists_lex.append(lexr)
                ratios.append(ratio); mf1s.append(c["mf1"]); pt_src.append(src)

    def _spear(x, y):
        return spearmanr(x, y) if len(set(x)) > 1 else (float("nan"), float("nan"))
    rho_g, p_g = _spear(dists_g, ratios)
    rho_lex, p_lex = _spear(dists_lex, ratios)
    rho_r, p_r = _spear(dists, ratios)
    rho_m, p_m = _spear(dists_g, mf1s)

    write_csv(out / f"rq2_transfer_summary{suffix}.csv", summary_rows)
    matrix_rows = [{"source": s, **{t: round(cell[(s, t)]["mf1"], 6) for t in langs}}
                    for s in langs]
    write_csv(out / f"rq2_matrix{suffix}.csv", matrix_rows)

    src0 = langs[0]; audit = []
    for tgt in langs:
        rows, _, _ = audit_rows(checker.detection_models[src0],
                                checker.threshold_classifiers.get(src0),
                                list(dict.fromkeys(words[tgt])), seed=0,
                                extra={"source": src0, "target": tgt},
                                min_word_length=min_wl)
        audit.extend(rows)
    write_csv(out / f"rq2_words_seed0{suffix}.csv", audit)

    (out / f"rq2_transfer{suffix}.json").write_text(json.dumps({
        "version": version, "languages": langs, "seeds": seeds,
        "distance_primary_weighted": "10*zone_hops + 3*group_diff + 1*lang_diff "
                                    "(zone_hops = BFS over the geographic zone-adjacency graph)",
        "distance_weightless": "lexicographic key on (zone_hops, group_diff, lang_diff)",
        "distance_legacy_alphabetical": "10*|zone_index diff| + 3*group_diff + 1*lang_diff",
        "spearman_graphweighted_vs_transfer_ratio": {"rho": rho_g, "p": p_g, "n_pairs": len(dists_g)},
        "spearman_weightless_vs_transfer_ratio": {"rho": rho_lex, "p": p_lex, "n_pairs": len(dists_lex)},
        "spearman_alphabetical_vs_transfer_ratio": {"rho": rho_r, "p": p_r, "n_pairs": len(dists)},
        "spearman_graphweighted_vs_mf1": {"rho": rho_m, "p": p_m, "n_pairs": len(dists_g)},
        "matrix": {f"{s}->{t}": cell[(s, t)]["mf1"] for s in langs for t in langs}},
        indent=2))

    names = [disp(l) for l in langs]
    M = np.array([[cell[(s, t)]["mf1"] for t in langs] for s in langs])
    fig, ax = plt.subplots(figsize=(1.15 * len(langs) + 2, 1.05 * len(langs) + 2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=11)
    ax.set_yticks(range(len(langs))); ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("target (tested on)", fontsize=12)
    ax.set_ylabel("source (trained on)", fontsize=12)

    for i in range(len(langs)):
        for j in range(len(langs)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=12, fontweight="bold")
    fig.colorbar(im, label="mF1")
    ax.set_title(f"Directed cross-language transfer\nSpearman(zone-adjacency dist, ratio) "
                f"rho={rho_g:.3f} p={p_g:.3f}, n={len(dists_g)} pairs")
    fig.tight_layout(); fig.savefig(out / f"rq2_transfer{suffix}.pdf", dpi=150); plt.close(fig)

    # a cell of the symmetric matrix is the mean of the two directed transfer ratios of a pair
    ratio_cell = {}
    for s in langs:
        base = cell[(s, s)]["mf1"] or 1.0
        for t in langs:
            ratio_cell[(s, t)] = cell[(s, t)]["mf1"] / base
    S = np.zeros((len(langs), len(langs)))
    for i, s in enumerate(langs):
        for j, t in enumerate(langs):
            S[i, j] = 1.0 if i == j else 0.5 * (ratio_cell[(s, t)] + ratio_cell[(t, s)])
    # upper triangle, for the correlation with the graph distance
    sim_u, dist_u = [], []
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            sim_u.append(S[i, j]); dist_u.append(guthrie_graph_distance(langs[i], langs[j]))
    rho_sim, p_sim = (_spear(dist_u, sim_u) if len(sim_u) > 1 else (float("nan"), float("nan")))

    corr_rows = [{"language": disp(s),
                  **{disp(t): round(float(S[i, j]), 6) for j, t in enumerate(langs)}}
                for i, s in enumerate(langs)]
    write_csv(out / f"rq2_correlation_matrix{suffix}.csv", corr_rows)

    figc, axc = plt.subplots(figsize=(1.15 * len(langs) + 2, 1.05 * len(langs) + 2))
    imc = axc.imshow(S, cmap="RdYlGn", vmin=0, vmax=1)
    axc.set_xticks(range(len(langs))); axc.set_xticklabels(names, rotation=40, ha="right", fontsize=11)
    axc.set_yticks(range(len(langs))); axc.set_yticklabels(names, fontsize=11)
    axc.set_xlabel("language", fontsize=12); axc.set_ylabel("language", fontsize=12)
    for i in range(len(langs)):
        for j in range(len(langs)):
            axc.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=11, fontweight="bold")
    figc.colorbar(imc, label="symmetric transfer similarity (mean of both directions)")
    axc.set_title("RQ2 cross-language transfer-similarity matrix (symmetric)\n"
                    f"Spearman(zone-adjacency distance, similarity) rho={rho_sim:.3f} "
                    f"p={p_sim:.3f}, n={len(sim_u)} pairs")
    figc.tight_layout()
    write_figure(figc, out, f"rq2_correlation_matrix{suffix}"); plt.close(figc)

    # one heatmap per distance definition
    _dist_heatmap(langs, guthrie_graph_distance, out / f"rq2_dist_weighted{suffix}.pdf",
                "Guthrie zone-adjacency distance (WEIGHTED: 10*hops + 3*group + 1*lang)")
    _dist_heatmap(langs, guthrie_graph_lexrank, out / f"rq2_dist_weightless{suffix}.pdf",
                "Guthrie zone-adjacency distance (WEIGHTLESS lexicographic; cells = "
                "(zone hops, group, language), colour = rank order)",
                as_rank=True,
                annot_fn=lambda s, t: "({},{},{})".format(*guthrie_graph_components(s, t)))
    _dist_heatmap(langs, guthrie_full_distance, out / f"rq2_dist_legacy{suffix}.pdf",
                "Guthrie distance (LEGACY alphabetical - geographically incorrect)")

    print(f"\n{'='*74}\n  RQ2 {title}  ({len(langs)} langs, {seeds} seeds, "
        f"{test_size} words/lang)\n{'='*74}")
    print("  source\\target" + "".join(f"{t[:7]:>8}" for t in langs))

    for s in langs:
        print(f"  {s:<12}" + "".join(f"{cell[(s,t)]['mf1']:>8.3f}" for t in langs))
    print(f"\n  Spearman vs transfer ratio  (n={len(dists_g)} directed off-diagonal pairs):")
    print(f"    zone-ADJACENCY, WEIGHTED   (10*hops+3*grp+1*lang)  rho={rho_g:.3f}  p={p_g:.3f}  <- primary")
    print(f"    zone-adjacency, WEIGHTLESS (lexicographic)         rho={rho_lex:.3f}  p={p_lex:.3f}")
    print(f"    legacy alphabetical         (reference)            rho={rho_r:.3f}  p={p_r:.3f}")
    print(f"  Spearman(zone-adjacency weighted, mF1)               rho={rho_m:.3f}  p={p_m:.3f}")
    print(f"  full numbers in rq2_transfer_summary{suffix}.csv / rq2_matrix{suffix}.csv")
    print(f"{'='*74}\n")

    tag = f"_{version}_{mode}"
    mlflow.log_metrics({f"spearman_graphw_rho{tag}": rho_g, f"spearman_graphw_p{tag}": p_g,
                        f"spearman_lex_rho{tag}": rho_lex, f"spearman_lex_p{tag}": p_lex,
                        f"spearman_alpha_rho{tag}": rho_r,
                        f"spearman_mf1_rho{tag}": rho_m, f"n_pairs{tag}": len(dists_g)})
    return {"rho_ratio": rho_g, "p_ratio": p_g, "rho_lex": rho_lex, "p_lex": p_lex,
            "rho_alpha": rho_r, "rho_mf1": rho_m, "n_pairs": len(dists_g)}


def _available(checker, langs):
    return [l for l in langs
            if l in checker.detection_models and corpus_path(l, "test").exists()]


def run(seeds, test_size, v1_langs=None, v2_langs=None, v4_langs=None, v3_langs=None,
        v5_langs=None, modes=("type", "token")):
    """
    Runs every language set once per mode. The type mode samples distinct words and the token
    mode samples running text, and each writes to its own subfolder of results/rq2/.
    """
    mlflow.set_experiment("RQ2_Cross_Language")

    with mlflow.start_run():
        mlflow.log_params({"experiment": "rq2_all_pairs", "seeds": seeds,
                        "test_size": test_size, "modes": ",".join(modes)})
        checker = SpellChecker()
        checker.load_models(ROOT / "data" / "models")
        base = ROOT / "results" / "rq2"
        allres = {}

        zones = set()
        for l in _available(checker, V5_LANGS):
            try:
                zones.add(parse_code(GUTHRIE_CODES[l])[0])
            except Exception:
                pass
        base.mkdir(parents=True, exist_ok=True)
        _zone_graph_figure(base, zones)
        mlflow.log_artifact(str(base / "guthrie_zone_graph.pdf"))

        for mode in modes:
            out = base / f"{mode}-based"; out.mkdir(parents=True, exist_ok=True)
            logger.info("=== RQ2 %s-based ===", mode)
            r1 = _run_version(checker, v1_langs or V1_LANGS, seeds, test_size,
                            "", "agglutinative", out, "v1", mode)
            r2 = _run_version(checker, v2_langs or V2_LANGS, seeds, test_size,
                            "_2", "incl. disjunctive", out, "v2", mode)
            r3 = _run_version(checker, v3_langs or V3_LANGS, seeds, test_size,
                            "_3", "SADiLaR-Nguni set", out, "v3", mode)
            r4 = _run_version(checker, v4_langs or V4_LANGS, seeds, test_size,
                            "_4", "incl. siSwati", out, "v4", mode)

            v5 = v5_langs or V5_LANGS
            r5 = None
            if len(_available(checker, v5)) > len(_available(checker, v4_langs or V4_LANGS)):
                r5 = _run_version(checker, v5, seeds, test_size,
                                "_5", "expanded multi-zone", out, "v5", mode)
            else:
                logger.info("RQ2 v5 (expanded multi-zone) SKIPPED - no extra corpora present "
                            "yet; train the new languages to activate it.")

            r6 = None
            if len(_available(checker, V6_LANGS)) >= 3:
                r6 = _run_version(checker, V6_LANGS, seeds, test_size,
                                "_6", "low-short-word control", out, "v6", mode)
            else:
                logger.info("RQ2 v6 (low-short-word control) SKIPPED - fewer than 3 languages trained.")

            for f in sorted(out.glob("rq2_*")):
                mlflow.log_artifact(str(f), artifact_path=f"{mode}-based")
            logger.info("RQ2 %s done. v1 rho=%.3f (n=%d) | v2 rho=%.3f (n=%d) | "
                        "v3 rho=%.3f (n=%d) | v4 rho=%.3f (n=%d)%s",
                        mode, r1["rho_ratio"], r1["n_pairs"], r2["rho_ratio"], r2["n_pairs"],
                        r3["rho_ratio"], r3["n_pairs"], r4["rho_ratio"], r4["n_pairs"],
                        f" | v5 rho={r5['rho_ratio']:.3f} (n={r5['n_pairs']})" if r5 else "")
            allres[mode] = {"v1": r1, "v2": r2, "v3": r3, "v4": r4, "v5": r5, "v6": r6}
        return allres


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--test-size", type=int, default=4000)
    ap.add_argument("--v1-langs", nargs="+", default=None,
                    help="override v1 (agglutinative) language set")
    ap.add_argument("--v2-langs", nargs="+", default=None,
                    help="override v2 (incl. disjunctive) language set")
    ap.add_argument("--v3-langs", nargs="+", default=None,
                    help="override v3 (SADiLaR-Nguni) language set")
    ap.add_argument("--v4-langs", nargs="+", default=None,
                    help="override the siSwati-inclusive language set")
    ap.add_argument("--v5-langs", nargs="+", default=None,
                    help="override the expanded multi-zone language set")
    a = ap.parse_args()
    run(a.seeds, a.test_size, a.v1_langs, a.v2_langs, a.v4_langs, a.v3_langs, a.v5_langs)
