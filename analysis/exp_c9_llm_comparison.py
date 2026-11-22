"""
exp_c9_llm_comparison.py - the checker against large language models on spelling detection.

Each model is given the same words as the checker, clean and corrupted, and is asked for a
verdict and a short reason per word. The cloud models are reached through Groq, OpenRouter and
Google Gemini, and the local models through Ollama. The answers are cached per model in
results/c9/cache/, so a run that is interrupted resumes where it stopped.

The API keys GROQ_API_KEY, OPENROUTER_API_KEY and GEMINI_API_KEY are read from .env.

Outputs (results/c9/): c9_reasoned_{lang}[_local|_cloud].json, the summary, per-word and
disagreement CSVs, and a bar chart of macro-F1
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_common import (ROOT, load_words, corpus_path, sample_unique, confusion_to_mf1,
                        lr_boundary, MIN_WORD_LEN, write_csv, write_figure)
from src.core.spell_checker import SpellChecker
from src.evaluation.spelling_wrecker import SpellingWrecker

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# providers drop model ids without notice, so these lists need checking before a run
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

# the OpenRouter free tier has a low daily cap
OPENROUTER_WORKING_MODELS = [
    "google/gemma-4-31b-it:free",
]

# Gemini uses generateContent, not the OpenAI chat format
GEMINI_MODELS = [
    "gemini-3.6-flash",
]

# local models served by Ollama
OLLAMA_MODELS = [
    "hf.co/mradermacher/AfriqueQwen3.5-9B-50Langs-Instruct-v1-i1-GGUF:Q4_K_M",
    "gemma4:12b",
    "gpt-oss:20b",
]

GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA_URL      = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_NUM_CTX  = 16384   # Ollama defaults to 4096
OLLAMA_THINK    = False  # local models run without reasoning
OPENROUTER_URL  = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL      = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# the default urllib user agent is refused with a 403
_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) NCB-SpellChecker-Research/1.0"

MAX_TOKENS = 8192

# a pass is sent in batches, because a reasoned reply for 500 words overruns MAX_TOKENS
FLAG_BATCH   = 50  # words per request in flag mode
REASON_BATCH = 25  # words per request in reasoned mode
BATCH_SLEEP  = 10.0  # seconds between batches


def _chunks(seq, size):
    """Yields the start index and the items of each chunk of the given size."""
    for i in range(0, len(seq), size):
        yield i, seq[i:i + size]

_PROV_TAG = {"groq": "[Groq]", "openrouter": "[OR]", "gemini": "[Gemini]", "ollama": "[local]"}

# each model has a fixed colour, and an unlisted model is hashed into tab20
_MODEL_COLOURS = {
    "spellchecker": "#d62728",
    "gemini-2.5-flash": "#59a14f", "gemini-2.0-flash": "#8cd17d",
    "gpt-oss-120b": "#b07aa1", "qwen3-32b": "#9c755f",
    "llama-3.3-70b-versatile": "#4e79a7", "llama-3.1-8b-instant": "#a0cbe8",
    "llama-4-scout-17b-16e-instruct": "#1f77b4",
    "gemma-4-31b-it": "#f28e2b", "gemma-4-26b-a4b-it": "#ffbe7d",
    "compound-mini": "#76b7b2", "nemotron-3-super-120b-a12b": "#86bcb6",
}
def _model_colour(key):
    """Returns the colour of a model in the charts."""
    name = key.split("/")[-1].replace(":free", "")
    if name in _MODEL_COLOURS:
        return _MODEL_COLOURS[name]
    pal = list(matplotlib.colormaps["tab20"].colors)
    return pal[int(hashlib.md5(name.encode()).hexdigest(), 16) % len(pal)]

# the conjunctively written languages that have trained models and test corpora
AGGLUTINATIVE_LANGS = ["zulu", "xhosa", "ndebele", "swati", "shona", "swahili"]
LANG_DISPLAY = {"zulu": "isiZulu", "xhosa": "isiXhosa", "ndebele": "isiNdebele",
                "swati": "siSwati", "shona": "Shona", "swahili": "Kiswahili"}


PROMPT_TEMPLATE = """\
You are a spelling checker for {language}, a Bantu language.
Below is a numbered list of {language} words. Some are correctly spelled; some \
may contain exactly one character-level typo (insertion, deletion, substitution, \
or adjacent transposition). Any number of them may be correct or misspelled.

Return ONLY a JSON array of the numbers of the MISSPELLED words.
No explanation. No markdown. Just the JSON array.

Example response: [2, 5, 11]

Words:
{numbered}"""


def build_eval_set(lang, n, seed=42):
    """
    Returns n distinct test words, their corrupted forms in the same order, and for each word
    whether the corruption changed it.
    """
    originals = sample_unique(load_words(corpus_path(lang, "test")), n, seed=seed)
    wr = SpellingWrecker(corruption_rate=1.0, seed=seed)
    corrupted, is_err = [], []
    for w in originals:
        c, gt = wr.corrupt_text(w, min_word_length=MIN_WORD_LEN); c = c.strip()
        changed = (c in gt) and (c != w)
        corrupted.append(c if changed else w)
        is_err.append(1 if changed else 0)
    return originals, corrupted, is_err


def _confusion_two_pass(clean_flags, corr_flags, is_err):
    """
    Counts the outcomes of a pass over the original words and a pass over their corrupted
    forms. Returns (tp, fp, fn, tn) and the outcome of every word.
    """
    tp = fp = fn = tn = 0; oc_clean, oc_corr = [], []
    for f in clean_flags:
        if f: fp += 1; oc_clean.append("fp")
        else: tn += 1; oc_clean.append("tn")
    for f, lab in zip(corr_flags, is_err):
        if   lab and f: tp += 1; oc_corr.append("tp")
        elif f:         fp += 1; oc_corr.append("fp")
        elif lab:       fn += 1; oc_corr.append("fn")
        else:           tn += 1; oc_corr.append("tn")
    return (tp, fp, fn, tn), oc_clean, oc_corr


def _llm_flags(call_fn, model, words, lang, batch=FLAG_BATCH):
    """
    Runs one model over a word list in batches. Returns one flag per word, or None with the
    error when every batch failed. A single failed batch leaves its own words unflagged.
    """
    flags = [False] * len(words)
    n_ok = 0
    last_err = None
    for start, chunk in _chunks(words, batch):
        numbered = "\n".join(f"{i+1}. {w}" for i, w in enumerate(chunk))
        prompt = PROMPT_TEMPLATE.format(language=LANG_DISPLAY.get(lang, lang), numbered=numbered)
        nums, err = call_fn(model, prompt)
        if err or nums is None:
            last_err = err or "no parseable JSON array in response"
            logger.warning("  [%s] flag batch @%d-%d failed: %s", model, start,
                        start + len(chunk) - 1, last_err)
        else:
            for i in range(len(chunk)):
                flags[start + i] = ((i + 1) in nums)
            n_ok += 1
        time.sleep(BATCH_SLEEP)
    if n_ok == 0:
        return None, (last_err or "all batches failed")
    return flags, None


# in reasoned mode each word gets one object with the verdict and a short reason
PROMPT_TEMPLATE_REASONED = """\
You are a spelling checker for {language}, a Bantu language.
Below is a numbered list of {language} words. Some are correctly spelled; some may contain \
exactly one character-level typo (insertion, deletion, substitution, or adjacent \
transposition). Any number may be correct or misspelled.

For EVERY numbered word, decide whether it is misspelled and give a SHORT reason (one clause, \
max ~12 words) grounded in {language} spelling/morphology.

Return ONLY a JSON array with one object per word, in order:
[{{"n": 1, "misspelled": true, "reason": "..."}}, {{"n": 2, "misspelled": false, "reason": "..."}}]
No markdown, no prose outside the JSON.

Words:
{numbered}"""


def _parse_json_objects(text, n):
    """
    Parses a reasoned reply into n flags and n reasons, or returns None for both. Markdown
    fences, objects out of order and missing entries are tolerated.
    """
    text = text.strip()
    for fence in ["```json", "```"]:
        if fence in text:
            text = text.split(fence, 1)[-1].rsplit("```", 1)[0]
    try:
        start = text.index("["); end = text.rindex("]") + 1
        arr = json.loads(text[start:end])
    except Exception:
        return None, None
    if not isinstance(arr, list) or not arr:
        return None, None
    flags = [False] * n
    reasons = [""] * n
    matched = 0
    for i, obj in enumerate(arr):
        if not isinstance(obj, dict):
            continue
        # the index is the 1-based n when it is present and in range, and the position otherwise
        idx = obj.get("n")
        idx = (idx - 1) if isinstance(idx, int) and 1 <= idx <= n else i
        if not (0 <= idx < n):
            continue
        val = obj.get("misspelled", obj.get("bad", obj.get("error")))
        flags[idx] = bool(val) if isinstance(val, bool) else str(val).strip().lower() in ("true", "1", "yes")
        reasons[idx] = str(obj.get("reason", obj.get("why", ""))).replace("\n", " ").strip()[:200]
        matched += 1
    if matched == 0:
        return None, None
    return flags, reasons


def _load_cache(path, words):
    """
    Returns the cached answers of a model when they were given for the same words, and an empty
    cache otherwise.
    """
    key = hashlib.md5("\n".join(words).encode("utf-8")).hexdigest()
    if path is not None and path.exists():
        c = json.loads(path.read_text(encoding="utf-8"))
        if c.get("words_md5") == key:
            return c
    return {"words_md5": key, "batches": {}}


def _llm_reasoned(call_fn, model, words, lang, think, batch=REASON_BATCH, cache_path=None,
                  pause=BATCH_SLEEP):
    """
    Runs one reasoned pass in batches and returns the flags, the reasons and an error. A batch
    answered on an earlier run is read from the cache and is not sent again. The error is set
    while any batch is still missing.
    """
    flags = [False] * len(words)
    reasons = [""] * len(words)
    cache = _load_cache(cache_path, words)
    n_ok = 0
    last_err = None
    for start, chunk in _chunks(words, batch):
        done = cache["batches"].get(str(start))
        if done:
            flags[start:start + len(chunk)] = done["flags"]
            reasons[start:start + len(chunk)] = done["reasons"]
            n_ok += 1
            continue
        numbered = "\n".join(f"{i+1}. {w}" for i, w in enumerate(chunk))
        prompt = PROMPT_TEMPLATE_REASONED.format(language=LANG_DISPLAY.get(lang, lang),
                                                numbered=numbered)
        raw, err = call_fn(model, prompt, return_raw=True, think=think)
        bf, br = (None, None)
        if not err and raw is not None:
            bf, br = _parse_json_objects(raw, len(chunk))
        if bf is None:
            last_err = err or "no parseable per-word JSON objects in response"
            logger.warning("  [%s] reasoned batch @%d-%d failed: %s", model, start,
                        start + len(chunk) - 1, last_err)
        else:
            for i in range(len(chunk)):
                flags[start + i] = bf[i]
                reasons[start + i] = br[i]
            n_ok += 1
            cache["batches"][str(start)] = {"flags": bf, "reasons": br}
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        time.sleep(pause)
    n_total = -(-len(words) // batch)
    if n_ok < n_total:
        return None, None, f"incomplete: {n_ok}/{n_total} batches cached ({last_err})"
    return flags, reasons, None


def _parse_json_array(text):
    """Extracts a JSON array from the reply of a model."""
    text = text.strip()
    for fence in ["```json", "```"]:
        if fence in text:
            text = text.split(fence, 1)[-1].rsplit("```", 1)[0]
    try:
        start = text.index("[")
        end   = text.rindex("]") + 1
        return set(int(x) for x in json.loads(text[start:end]))
    except Exception:
        return None


def _post_with_retry(url, headers, body, timeout=60, max_retries=6):
    """
    Posts JSON and retries on a 429 or 503. The wait follows the Retry-After header, or
    doubles from five seconds.
    """
    delay = 5.0
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read()), None
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < max_retries - 1:
                ra = e.headers.get("Retry-After")
                wait = float(ra) if (ra and str(ra).strip().isdigit()) else delay
                wait += random.uniform(0, 1.0)
                logger.warning("%d from %s; backing off %.1fs (attempt %d/%d)",
                            e.code, url.split("/")[2], wait, attempt + 1, max_retries)
                time.sleep(wait); delay *= 2; continue
            return None, f"HTTP {e.code}: {e.reason}"
        except Exception as e:
            return None, str(e)
    return None, "max retries exceeded"


def _content_openai(data):
    """Returns the text of an OpenAI-style reply, or None when it is missing."""
    try:
        c = data["choices"][0]["message"]["content"]
        return c or None
    except Exception:
        return None


def call_groq(model, prompt, return_raw=False, think=False):
    """Sends a prompt to a model on Groq."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None, "GROQ_API_KEY not set"
    content = prompt
    payload = {"model": model, "temperature": 0.0, "max_tokens": MAX_TOKENS}
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "medium" if think else "low"
    if "qwen3" in model and not think:
        content = prompt + "\n/no_think"
    payload["messages"] = [{"role": "user", "content": content}]
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                "User-Agent": _USER_AGENT}
    data, err = _post_with_retry(GROQ_URL, headers, body, timeout=30)
    if err:
        return None, err
    text = _content_openai(data)
    if text is None:
        return None, "empty content"
    return (text if return_raw else _parse_json_array(text)), None


def call_ollama(model, prompt, return_raw=False, think=False):
    # the native /api/chat endpoint takes think and num_ctx
    """Sends a prompt to a local model served by Ollama."""
    payload = {"model": model, "stream": False,
               "messages": [{"role": "user", "content": prompt}],
               "think": ("medium" if think else "low") if "gpt-oss" in model else bool(think),
               "options": {"temperature": 0.0, "num_ctx": OLLAMA_NUM_CTX,
                           "num_predict": MAX_TOKENS}}
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "User-Agent": _USER_AGENT}
    # on a CPU one batch can take minutes
    data, err = _post_with_retry(OLLAMA_URL, headers, body, timeout=1800)
    if err:
        return None, err
    text = (data.get("message") or {}).get("content") or None
    if text is None:
        return None, "empty content"
    return (text if return_raw else _parse_json_array(text)), None


def call_openrouter(model, prompt, return_raw=False, think=False):
    """Sends a prompt to a model on OpenRouter."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None, "OPENROUTER_API_KEY not set"
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": MAX_TOKENS,
                    "messages": [{"role": "user", "content": prompt}]}).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
            "User-Agent": _USER_AGENT,
            "HTTP-Referer": "https://github.com/ncb-spellchecker",
            "X-Title": "NCB Spell Checker Research"}
    data, err = _post_with_retry(OPENROUTER_URL, headers, body, timeout=60)
    if err:
        return None, err
    text = _content_openai(data)
    if text is None:
        return None, "empty content"
    return (text if return_raw else _parse_json_array(text)), None


def _content_gemini(data):
    """Returns the text of a Gemini reply, or None when it is missing or was blocked."""
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        return text or None
    except Exception:
        return None


def call_gemini(model, prompt, return_raw=False, think=False):
    """Sends a prompt to a Gemini model, trying each API key in turn."""
    keys = [k for k in (os.environ.get("GEMINI_API_KEY"), os.environ.get("GeminiK2")) if k]
    if not keys:
        return None, "GeminiK2 / GEMINI_API_KEY not set"
    # thinkingBudget is 0 unless think is set, so the model does not spend its budget before answering
    gen_cfg = {"temperature": 0.0, "maxOutputTokens": MAX_TOKENS}
    if "2.5" in model:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 1024 if think else 0}
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    }).encode()
    # the key goes in a header and not in the URL, so it never reaches the logs
    # on a quota or permission error, try the next key
    for i, key in enumerate(keys):
        last = i == len(keys) - 1
        headers = {"Content-Type": "application/json", "User-Agent": _USER_AGENT,
                    "x-goog-api-key": key}
        data, err = _post_with_retry(GEMINI_URL.format(model=model), headers, body,
                                     timeout=60, max_retries=6 if last else 2)
        if err and not last and ("429" in err or "403" in err or "max retries" in err):
            continue
        break
    if err:
        return None, err
    text = _content_gemini(data)
    if text is None:
        return None, "empty content"
    return (text if return_raw else _parse_json_array(text)), None


PROVIDER_GROUPS = {"cloud": ["groq", "openrouter", "gemini"], "local": ["ollama"]}


def _output_tag(lang, providers):
    """
    Returns the tag of the output files, which is the language followed by the providers, so
    that a run of one group never overwrites the files of another.
    """
    if not providers:
        return lang
    return lang + "_" + "_".join(sorted(set(providers)))


def run(lang, n=500, providers=None):
    """Runs the comparison for one language on n words."""
    return run_reasoned(lang, n, providers)


def run_reasoned(lang, n=500, providers=None):
    """
    Collects the verdict and the reason of every model for every word, and tabulates where a
    model and the checker disagree.
    """
    think = True
    res_dir = ROOT / "results" / "c9"; res_dir.mkdir(parents=True, exist_ok=True)
    out_tag = _output_tag(lang, providers)
    groq_models, openrouter_models, gemini_models = GROQ_MODELS, OPENROUTER_WORKING_MODELS, GEMINI_MODELS
    ollama_models = OLLAMA_MODELS

    originals, corrupted, is_err = build_eval_set(lang, n)

    # the checker, with its score against the boundary as the reason
    checker = SpellChecker(); checker.load_models(ROOT / "data" / "models")
    det = checker.detection_models[lang]; clf = checker.threshold_classifiers[lang]
    b = lr_boundary(clf); tau = b["tau"] if b else None
    def chk_detail(words):
        flags, reasons = [], []
        for w in words:
            if len(w) < MIN_WORD_LEN:
                flags.append(False); reasons.append("skipped (<3 chars)"); continue
            s = det.score_word(w); fl = clf.predict_error(s)
            flags.append(bool(fl))
            reasons.append(f"score {s:.2f} {'<' if s < (tau or 0) else '>='} tau {tau:.2f}"
                            if tau is not None else f"score {s:.2f}")
        return flags, reasons

    systems = {}
    cf, cr = chk_detail(originals); rf, rr = chk_detail(corrupted)
    systems["spellchecker"] = {"clean": cf, "corr": rf, "clean_reason": cr, "corr_reason": rr,
                            "provider": "spellchecker"}

    providers_all = [("groq", groq_models, call_groq, 3.0, BATCH_SLEEP),
                ("openrouter", openrouter_models, call_openrouter, 12.0, BATCH_SLEEP),
                ("gemini", gemini_models, call_gemini, 20.0, BATCH_SLEEP),
                ("ollama", ollama_models, call_ollama, 0.0, 0.0)]
    selected = None if providers is None else {
        x for g in providers for x in PROVIDER_GROUPS.get(g, [g])}
    providers = [t for t in providers_all if selected is None or t[0] in selected]
    for prov, models, call_fn, sleep_s, pause in providers:
        th = OLLAMA_THINK if prov == "ollama" else think
        for model in models:
            key = f"{prov}/{model}"
            cdir = (res_dir / "cache" / lang / f"think{int(th)}"
                    / key.replace("/", "__").replace(":", "_"))
            logger.info("[%s] %s: reasoned clean pass ...", prov, model)
            cf, cre, e1 = _llm_reasoned(call_fn, model, originals, lang, th,
                                        cache_path=cdir / "clean.json", pause=pause); time.sleep(sleep_s)
            logger.info("[%s] %s: reasoned corrupted pass ...", prov, model)
            rf, rre, e2 = _llm_reasoned(call_fn, model, corrupted, lang, th,
                                        cache_path=cdir / "corrupted.json", pause=pause); time.sleep(sleep_s)
            if cf is None or rf is None:
                systems[key] = {"error": (e1 or e2), "provider": prov}
                logger.warning("[%s] %s failed: %s", prov, model, e1 or e2)
            else:
                systems[key] = {"clean": cf, "corr": rf, "clean_reason": cre,
                                "corr_reason": rre, "provider": prov}

    summary = {}
    for key, s in systems.items():
        if "clean" not in s:
            summary[key] = {"error": s.get("error"), "provider": s["provider"]}; continue
        (tp, fp, fn, tn), ocC, ocR = _confusion_two_pass(s["clean"], s["corr"], is_err)
        mf1, p_e, r_e, *_ = confusion_to_mf1(tp, fp, fn, tn)
        summary[key] = {"mF1": round(mf1, 4), "provider": s["provider"],
                        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
                        "incorrect_precision": round(p_e, 4), "incorrect_recall": round(r_e, 4),
                        "oc_clean": ocC, "oc_corr": ocR}
    ok = [k for k in summary if "mF1" in summary[k]]

    # one row per word and form, with the flag, outcome and reason of every system
    rows = []
    for i in range(n):
        for inst, word, lab in [("clean", originals[i], 0), ("corrupted", corrupted[i], is_err[i])]:
            row = {"idx": i + 1, "instance": inst, "original": originals[i],
                    "word": word, "is_error": lab}
            for k in ok:
                tag = k.split("/")[-1].replace(":free", "")
                oc = summary[k]["oc_clean"][i] if inst == "clean" else summary[k]["oc_corr"][i]
                reason = (systems[k]["clean_reason"] if inst == "clean" else systems[k]["corr_reason"])[i]
                row[f"{tag}_flag"] = int(oc in ("tp", "fp"))
                row[f"{tag}_outcome"] = oc
                row[f"{tag}_reason"] = reason
            rows.append(row)
    write_csv(res_dir / f"c9_reasoned_words_{out_tag}.csv", rows)

    # where a language model and the checker disagree, and which was right
    dis_rows = []
    for k in ok:
        if k == "spellchecker":
            continue
        tag = k.split("/")[-1].replace(":free", "")
        for i in range(n):
            for inst, word, lab in [("clean", originals[i], 0), ("corrupted", corrupted[i], is_err[i])]:
                cflag = systems["spellchecker"]["clean"][i] if inst == "clean" else systems["spellchecker"]["corr"][i]
                lflag = systems[k]["clean"][i] if inst == "clean" else systems[k]["corr"][i]
                if bool(cflag) == bool(lflag):
                    continue
                who = "checker" if (bool(cflag) == bool(lab)) else "llm"
                lreason = (systems[k]["clean_reason"] if inst == "clean" else systems[k]["corr_reason"])[i]
                creason = (systems["spellchecker"]["clean_reason"] if inst == "clean"
                            else systems["spellchecker"]["corr_reason"])[i]
                dis_rows.append({"model": tag, "instance": inst, "original": originals[i],
                                "word": word, "is_error": lab, "correct_answer_flag": lab,
                                "checker_flag": int(bool(cflag)), "llm_flag": int(bool(lflag)),
                                "who_was_right": who, "llm_reason": lreason,
                                "checker_reason": creason})
    if dis_rows:
        write_csv(res_dir / f"c9_disagreements_{out_tag}.csv", dis_rows)
        won = {"checker": sum(r["who_was_right"] == "checker" for r in dis_rows),
                "llm": sum(r["who_was_right"] == "llm" for r in dis_rows)}
    else:
        won = {"checker": 0, "llm": 0}

    word_records = []
    for i in range(n):
        rec = {"idx": i + 1, "original": originals[i], "corrupted": corrupted[i],
                "is_error": is_err[i], "judgements": {"clean": {}, "corrupted": {}}}
        for k in ok:
            tag = k.split("/")[-1].replace(":free", "")
            rec["judgements"]["clean"][tag] = {
                "misspelled": bool(systems[k]["clean"][i]),
                "reason": systems[k]["clean_reason"][i],
                "outcome": summary[k]["oc_clean"][i]}
            rec["judgements"]["corrupted"][tag] = {
                "misspelled": bool(systems[k]["corr"][i]),
                "reason": systems[k]["corr_reason"][i],
                "outcome": summary[k]["oc_corr"][i]}
        word_records.append(rec)

    final = {"language": lang, "n_words": n, "think": think, "think_local": OLLAMA_THINK,
            "date": datetime.date.today().isoformat(),
            "mode": "reasoned (per-word rationale; SEPARATE from frozen flag-mode C9)",
            "systems": {k: {"mF1": summary[k].get("mF1"), "confusion": summary[k].get("confusion"),
                            "provider": summary[k].get("provider"),
                            "error": summary[k].get("error")} for k in summary},
            "disagreements_won": won,
            "words": word_records}
    (res_dir / f"c9_reasoned_{out_tag}.json").write_text(json.dumps(final, ensure_ascii=False, indent=2))

    write_csv(res_dir / f"c9_reasoned_summary_{out_tag}.csv",
              [{"system": k, "provider": summary[k]["provider"], **summary[k]["confusion"],
                "mF1": summary[k]["mF1"]} for k in ok])

    ranked = sorted(ok, key=lambda k: -summary[k]["mF1"])
    labels, vals, cols = [], [], []
    for k in ranked:
        name = k.split("/")[-1].replace(":free", "")
        tag = _PROV_TAG.get(summary[k]["provider"], "")
        labels.append("spell checker" if k == "spellchecker" else f"{name} {tag}".strip())
        vals.append(summary[k]["mF1"]); cols.append(_model_colour(k))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (v, c) in enumerate(zip(vals, cols)):
        ax.bar(i, v, color=c, alpha=0.9, edgecolor="white", linewidth=0.6)
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title(f"C9: spell checker vs free LLMs - {LANG_DISPLAY.get(lang, lang)} "
                f"({n} words, balanced)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    write_figure(fig, res_dir, f"c9_reasoned_{out_tag}", bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'='*74}\n  C9-REASONED - {LANG_DISPLAY.get(lang, lang)} ({n} words, "
            f"think={think})\n{'='*74}")
    print(f"  {'system':<42} {'mF1':>6}   (disagreements won: checker {won['checker']} / "
            f"llm {won['llm']})")
    for k in ["spellchecker"] + [x for x in ok if x != "spellchecker"]:
        print(f"  {k.split('/')[-1][:40]:<42} {summary[k]['mF1']:>6.3f}")
    for k in summary:
        if "mF1" not in summary[k]:
            print(f"  {k.split('/')[-1][:40]:<42} {'FAILED':>6}  ({summary[k].get('error')})")
    print(f"{'='*74}")
    print(f"  wrote c9_reasoned_words_{out_tag}.csv, c9_disagreements_{out_tag}.csv, "
            f"c9_reasoned_{out_tag}.json\n")

    mlflow.set_experiment("llm_comparison_analysis")
    with mlflow.start_run(run_name=f"c9_reasoned_{out_tag}"):
        mlflow.log_params({"lang": lang, "n": n, "think": think,
                            "groq_models": ",".join(groq_models),
                            "openrouter_models": ",".join(openrouter_models),
                            "gemini_models": ",".join(gemini_models),
                            "ollama_models": ",".join(ollama_models)})
        for k in ok:
            mlflow.log_metric(f"mF1_{k.split('/')[-1].replace(':free', '')}", summary[k]["mF1"])
        mlflow.log_metrics({"disagreements_checker_won": won["checker"],
                            "disagreements_llm_won": won["llm"]})
        for fn in [f"c9_reasoned_{out_tag}.json", f"c9_reasoned_words_{out_tag}.csv",
                    f"c9_reasoned_summary_{out_tag}.csv", f"c9_disagreements_{out_tag}.csv",
                    f"c9_reasoned_{out_tag}.pdf", f"c9_reasoned_{out_tag}.png"]:
            fp = res_dir / fn
            if fp.exists():
                mlflow.log_artifact(str(fp))
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zulu",
                    help="a language, or 'all' for the six agglutinative langs "
                        + "/".join(AGGLUTINATIVE_LANGS))
    ap.add_argument("--providers", nargs="+", default=None,
                    choices=["cloud", "local", "groq", "openrouter", "gemini", "ollama"],
                    help="run only these providers or groups (cloud = groq+openrouter+gemini, "
                         "local = ollama); output files get a matching suffix")
    ap.add_argument("--n", type=int, default=None,
                    help="words, each evaluated clean AND corrupted (default 500; "
                        "batched, so 1000 is feasible where the daily API cap allows)")
    a = ap.parse_args()
    langs = AGGLUTINATIVE_LANGS if a.lang == "all" else [a.lang]
    for L in langs:
        run(L, a.n or 500, a.providers)
