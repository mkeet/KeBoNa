import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def experiments_setup():
    # %% [markdown]
    # # Experiment Runner
    # Runs every experiment with MLflow tracking. The runs are logged to the local sqlite
    # `mlflow.db`, which `mlflow ui --backend-store-uri sqlite:///mlflow.db` reads.

    # %%
    import sys
    import os as _os
    import mlflow
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv()
    # the underscore keeps _os private to this cell
    ROOT = Path(__file__).resolve().parent
    # MLFLOW_TRACKING_URI overrides the local store
    mlflow.set_tracking_uri(_os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT}/mlflow.db"))
    sys.path.insert(0, str(ROOT))
    return (ROOT,)


@app.cell
def rq1():
    # %% [markdown]
    # ## 1. Run RQ1 (Corpus Size Saturation)

    # %%
    from analysis.exp_rq1_corpus_size import run as run_rq1
    langs = ["zulu", "xhosa", "ndebele", "swati", "zu", "xh", "sw", "nd"]
    seeds = 30
    test_size = 4000

    run_rq1(langs, seeds, test_size)
    return


@app.cell
def rq2():
    # %% [markdown]
    # ## 2. Run RQ2 (Cross-Language Transfer)

    # %%
    from analysis.exp_rq2_cross_language import run as run_rq2
    # every language set in one call
    run_rq2(seeds=30, test_size=4000)
    return


@app.cell
def rq2_parallel_text():
    # %% [markdown]
    # ## 2b. RQ2 complement - cross-language transfer on UDHR PARALLEL text
    # The UDHR is the same document in every language, so a difference in transfer reflects
    # the languages and not the test corpora.

    # %%
    from analysis.exp_udhr_transfer import run as run_udhr_transfer
    run_udhr_transfer(source="zulu",
                      targets=["zulu", "xhosa", "ndebele", "swati", "shona"],
                      seeds=20)
    return


@app.cell
def rq3():
    # %% [markdown]
    # ## 3. Run RQ3 (Baseline Comparison)

    # %%
    from analysis.exp_rq3_external_eval import run_suite as run_rq3
    run_rq3()
    return


@app.cell
def rq4():
    # %% [markdown]
    # ## 4. Run RQ4 (Correction Accuracy)

    # %%
    from analysis.exp_rq4_correction import run as run_rq4
    run_rq4("zu", seeds=30, quick=None)
    run_rq4("zulu", seeds=30, quick=4000)
    return


@app.cell
def rq4_qualitative():
    # %% [markdown]
    # ## 4b. RQ4 complement - correction failure taxonomy and accuracy by word length
    # Both read results/rq4/rq4_corrections_seed0_*.csv, so run them after cell 4.

    # %%
    from analysis.qual_rq4_corrections import run as run_rq4_failures
    from analysis.qual_rq4_length import run as run_rq4_length

    for _lang in ("zu", "zulu"):
        run_rq4_failures(_lang)
        run_rq4_length(_lang)
    return


@app.cell
def timing_analysis():
    # %% [markdown]
    # ## 5. Run C5 (Timing Analysis)

    # %%
    from analysis.exp_c5_timing import run as run_c5
    run_c5("zu", repeats=5)
    run_c5("zulu", repeats=5)
    return


@app.cell
def error_type_analysis():
    # %% [markdown]
    # ## 6. Run C6 (Error Type Analysis)

    # %%
    from analysis.exp_c6_error_types import run as run_c6
    run_c6("zu", seeds=30, corruption_types=["random", "qwerty"])
    run_c6("zulu", seeds=30, corruption_types=["random", "qwerty"])
    return


@app.cell
def udhr_orthography_analysis():
    # %% [markdown]
    # ## 7. Run C7 (Orthography via UDHR)

    # %%
    from analysis.exp_c7_orthography_udhr import run as run_c7
    run_c7(seeds=30)
    return


@app.cell
def fp_fn_analysis():
    # %% [markdown]
    # ## 8. Run C8 (FP/FN Ceiling Analysis)

    # %%
    from analysis.exp_c8_fp_fn_analysis import run as run_c8
    run_c8("zu", seeds=30, n_examples=25)
    run_c8("zulu", seeds=30, n_examples=25)
    return


@app.cell
def llm_comparison_analysis():
    # %% [markdown]
    # ## 9. Run C9 (LLM Comparison)
    # Needs GROQ_API_KEY, OPENROUTER_API_KEY and GEMINI_API_KEY in .env, and Ollama for the
    # local models.

    # %%
    from analysis.exp_c9_llm_comparison import run as run_c9

    run_c9(lang="zu", n=500, providers=["local"])
    run_c9(lang="zu", n=500, providers=["cloud"])
    return


@app.cell
def find_available_models():
    # %% [markdown]
    # ## 10. Discover available LLM models (OpenRouter free tier + Groq)
    # The ids printed here go into GROQ_MODELS and OPENROUTER_WORKING_MODELS of
    # exp_c9_llm_comparison.py.

    # %%
    import urllib.request, json, os
    _UA = "Mozilla/5.0 (X11; Linux x86_64) NCB-SpellChecker-Research/1.0"

    def _get_models(url, key):
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {key}", "User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    try:
        m = _get_models("https://openrouter.ai/api/v1/models",
                        os.environ.get("OPENROUTER_API_KEY"))
        free = sorted(x["id"] for x in m["data"] if ":free" in x["id"])
        print(f"OpenRouter :free models ({len(free)}):")
        print("\n".join("  " + i for i in free))
    except Exception as e:
        print("OpenRouter model list failed:", e)

    try:
        g = _get_models("https://api.groq.com/openai/v1/models",
                        os.environ.get("GROQ_API_KEY"))
        gids = sorted(x["id"] for x in g.get("data", []))
        print(f"\nGroq models ({len(gids)}):")
        print("\n".join("  " + i for i in gids))
    except Exception as e:
        print("Groq model list failed:", e)
    return


@app.cell
def scalability_analysis():
    # %% [markdown]
    # ## 11. Run C10 (Scaling)

    # %%
    from analysis.exp_c10_scaling import run as run_c10
    run_c10("zu", seeds=30, test_size=4000)
    return


@app.cell
def eval_misspelling_analysis(ROOT):
    # %% [markdown]
    # ## 12. Run Full Detection Evaluation

    # %%
    from analysis.evaluate_detection import run as run_eval

    sadi = ["zulu", "xhosa", "ndebele", "swati"]; nchlt = ["zu", "xh", "nd", "sw"]
    ed = ROOT/"results"/"eval_detection"
    run_eval(sadi,  n_seeds=30, n_words=4000,  output_dir=ed/"sadi_morph"/"output_4k")
    run_eval(nchlt, n_seeds=30, n_words=4000,  output_dir=ed/"nchlt"/"output_4k")
    run_eval(nchlt, n_seeds=30, n_words=20000, output_dir=ed/"nchlt"/"output_20k")
    # all eight corpora in one run, for the grouped chart
    run_eval(sadi + nchlt, n_seeds=30, n_words=4000, output_dir=ed/"mixed"/"output_4k")
    return


if __name__ == "__main__":
    app.run()
