"""
exp_common.py - helpers shared by the experiment scripts.

Holds the language labels and colours used in the figures, the corpus and model paths, the
sampling of test words, the training of a fresh model, and the evaluation protocols.
"""

import csv
import math
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.spelling_wrecker import SpellingWrecker
from src.models.trigram_model import START_TOKEN, TrigramModel
from src.utils.text_utils import TextUtils
from src.utils.trigram_tokenizer import CharTrigramTokenizer

NGUNI = ["zulu", "xhosa", "ndebele", "swati"]

LANG_LABELS = {
    "zulu":    "isiZulu (SADiLaR)",
    "xhosa":   "isiXhosa (SADiLaR)",
    "ndebele": "isiNdebele (SADiLaR)",
    "swati":   "siSwati (SADiLaR)",
    "zu":      "isiZulu (NCHLT)",
    "xh":      "isiXhosa (NCHLT)",
    "nd":      "isiNdebele (NCHLT)",
    "sw":      "siSwati (NCHLT)",
    "shona":   "Shona (S10)",
}

# one colour per corpus, so the two corpora of a language never share one
LANG_COLOURS = {
    "zulu":    "#1f77b4",
    "xhosa":   "#d62728",
    "ndebele": "#2ca02c",
    "swati":   "#9467bd",
    "zu":      "#17becf",
    "xh":      "#ff7f0e",
    "nd":      "#bcbd22",
    "sw":      "#e377c2",
    "shona":   "#8c564b",
}

ERROR_TYPES  = ["insertion", "deletion", "substitution", "transposition"]
MIN_WORD_LEN = 3

# names on chart axes, where a language not listed is capitalised
LANG_DISPLAY = {
    "zu": "Zulu", "xh": "Xhosa", "nd": "Ndebele", "sw": "Swati",
    "zulu": "Zulu", "xhosa": "Xhosa", "ndebele": "Ndebele", "swati": "Swati",
    "shona": "Shona", "sotho": "Sotho", "tswana": "Tswana", "swahili": "Swahili",
    "lozi": "Lozi", "sepedi": "Sepedi", "venda": "Venda", "tsonga": "Tsonga",
    "kinyarwanda": "Kinyarwanda", "kirundi": "Kirundi", "luganda": "Luganda",
    "runyankore": "Runyankore", "kikuyu": "Kikuyu", "kamba": "Kamba",
    "chichewa": "Chichewa", "tumbuka": "Tumbuka", "bemba": "Bemba",
    "lingala": "Lingala", "kongo": "Kongo", "umbundu": "Umbundu", "herero": "Herero",
    "luba": "Luba", "yao": "Yao", "kimbundu": "Kimbundu", "lumbu": "Lumbu",
}


def disp(lang):
    """Returns the name of a language as it is written on chart axes."""
    return LANG_DISPLAY.get(lang, lang.capitalize())


def write_figure(fig, out_dir, stem, dpi=150, **savefig_kw):
    """Saves a figure as <stem>.pdf and <stem>.png in out_dir and returns the two paths."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"; png = out_dir / f"{stem}.png"
    fig.savefig(pdf, dpi=dpi, **savefig_kw)
    fig.savefig(png, dpi=dpi, **savefig_kw)
    return pdf, png


def load_words(path):
    """Reads a file of one word per line into a list."""
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]


def corpus_path(lang, which):
    """Returns the path of a split of a language, where which is train, test or validation."""
    return ROOT / "data" / "corpora" / lang / f"{which}.txt"


def model_path(lang, kind):
    """Returns the path of a model of a language, where kind is word, trigram or classifier."""
    return ROOT / "data" / "models" / f"{lang}_{kind}.pkl"


def sample_unique(words, n, seed=42):
    """Samples n word types, or returns every type when the corpus has fewer than n."""
    unique = list(dict.fromkeys(words))
    rng = random.Random(seed)
    if n is None or n >= len(unique):
        return unique
    return rng.sample(unique, n)


def sample_tokens(all_words, n, seed=42):
    """
    Samples n tokens from the running text, so that a word keeps the frequency it has in the
    corpus. Returns every token when the corpus has fewer than n.
    """
    rng = random.Random(seed)
    if n is None or n >= len(all_words):
        return list(all_words)
    return rng.sample(all_words, n)


def train_detection_model(words, lang):
    """Trains a detection model on a list of words, without saving it."""
    tok = CharTrigramTokenizer(n=3)
    lines = [" ".join(tris) for w in words
            for tris in [tok.segment_word(w)] if tris]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                    delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines)); tmp = f.name

    model = TrigramModel(n=3)
    model.train(Path(tmp), language=lang, tokenization="trigram")
    Path(tmp).unlink()
    return model


def train_lr(model, val_words, corruption_rate=1.0, seed=42):
    """
    Fits the threshold regression of a detection model on a list of validation words, in the
    same way as ThresholdClassifier.train. Returns the fitted regression, or None when either
    class is empty.
    """
    pos, neg = [], []
    for w in val_words:
        if len(w) >= MIN_WORD_LEN:
            s = model.score_word(w)
            if s > -19.0:
                pos.append(s)
    wrecker = SpellingWrecker(corruption_rate=corruption_rate, seed=seed)
    corrupted_text, gt = wrecker.corrupt_text(" ".join(val_words))
    for w in TextUtils.tokenize(TextUtils.normalize(corrupted_text)):
        if w in gt and len(w) >= MIN_WORD_LEN:
            s = model.score_word(w)
            if s > -19.0:
                neg.append(s)
    if not pos or not neg:
        return None
    X = np.array([[s] for s in pos] + [[s] for s in neg])
    y = np.array([1]*len(pos) + [0]*len(neg))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X, y)
    return clf


def confusion_to_mf1(tp, fp, fn, tn):
    """
    Returns macro-F1, which is the mean of the F1 on incorrect words and the F1 on correct
    words, followed by the precision, recall and F1 on incorrect words and the F1 on correct
    words.
    """
    p_e = tp/(tp+fp) if tp+fp else 0.0
    r_e = tp/(tp+fn) if tp+fn else 0.0
    f1_e = 2*p_e*r_e/(p_e+r_e) if p_e+r_e else 0.0
    p_c = tn/(tn+fn) if tn+fn else 0.0
    r_c = tn/(tn+fp) if tn+fp else 0.0
    f1_c = 2*p_c*r_c/(p_c+r_c) if p_c+r_c else 0.0
    return (f1_e+f1_c)/2.0, p_e, r_e, f1_e, f1_c


def class_metrics(tp, fp, fn, tn):
    """Returns precision, recall and F1 for incorrect and for correct words, with macro-F1."""
    mf1, p_e, r_e, f1_e, f1_c = confusion_to_mf1(tp, fp, fn, tn)
    r_c = tn/(tn+fp) if tn+fp else 0.0
    p_c = tn/(tn+fn) if tn+fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "incorrect_precision": p_e, "incorrect_recall": r_e, "incorrect_f1": f1_e,
            "correct_precision": p_c, "correct_recall": r_c, "correct_f1": f1_c,
            "mf1": mf1}


def lr_boundary(clf):
    """
    Returns the boundary tau = -intercept / coef of a loaded classifier, with its intercept
    and coefficient, or None when the classifier is untrained. A word is flagged when its
    score is at or below tau.
    """
    m = getattr(clf, "_clf", None)
    if m is None:
        return None
    try:
        intercept = float(m.intercept_[0]); coef = float(m.coef_[0][0])
        return {"tau": (-intercept/coef if coef else float("nan")),
                "intercept": intercept, "coef": coef}
    except Exception:
        return None


def _trigram_steps(model, word):
    """
    Returns the trigrams of a word and, for each one, its context, its counts, its
    probability and its log-probability under the model.
    """
    tris = CharTrigramTokenizer(n=3).segment_word(word)
    steps = []

    if not tris:
        return tris, steps
    padded = [START_TOKEN, START_TOKEN] + tris

    for i in range(2, len(padded)):
        token = padded[i]; ctx = tuple(padded[i - 2:i])
        denom = sum(model.ngrams[ctx].values()) if ctx in model.ngrams else 0
        num = model.ngrams[ctx].get(token, 0) if ctx in model.ngrams else 0
        prob = (num / denom) if denom > 0 else 0.0
        logp = math.log(prob) if prob > 0 else -20.0
        steps.append({"ctx": ctx, "token": token, "num": num,
                    "denom": denom, "prob": prob, "logp": logp})

    return tris, steps


def trigram_breakdown(model, word):
    """Returns the trigrams of a word, their log-probabilities and the mean, which is the score."""
    tris, steps = _trigram_steps(model, word)
    if not tris:
        return [], [], -20.0
    logps = [s["logp"] for s in steps]
    return tris, logps, (sum(logps) / len(tris))


def trigram_calc(model, word):
    """
    Writes out the arithmetic behind the score of a word, one step per trigram, in the form

        <ctxA>+<ctxB>><token>=num/denom=prob ln=logp

    ctxA and ctxB are the two preceding trigrams, with <S> for a START pad. num is the count
    of the token after that context and denom is the count of the context. A step marked
    (floor) had an unseen context or token and contributes -20. The mean of the ln values is
    the score of the word.
    """

    tris, steps = _trigram_steps(model, word)
    if not tris:
        return ""
    def _s(t): return "<S>" if t == START_TOKEN else t
    parts = []
    for s in steps:
        a, b = s["ctx"]
        floored = " (floor)" if s["prob"] == 0.0 else ""
        parts.append(f"{_s(a)}+{_s(b)}>{s['token']}={s['num']}/{s['denom']}="
                    f"{s['prob']:.6f} ln={s['logp']:.4f}{floored}")
    return " ; ".join(parts)


def audit_rows(det, clf, words, seed, extra=None, min_word_length=MIN_WORD_LEN):
    """
    Builds the word-by-word record of one evaluation run.

    Each word gives a clean row, which counts as a false positive or a true negative, and a
    corrupted row, which counts as a true positive or a false negative. A corruption that
    reproduces the word is recorded as wasted. Every row carries the trigrams, their
    log-probabilities and the outcome, so the confusion counts can be recomputed from the CSV.

    Args:
        det: The detection model that scores the words. In a transfer run this is the model
            of the source language.
        clf: The classifier of the same language.
        words: The test words.
        seed: Seed of the corruption.
        extra: Columns added to the front of every row.
        min_word_length: Words shorter than this are not scored and are not flagged.

    Returns:
        The rows, the counts (tp, fp, fn, tn) and the boundary tau.
    """

    b = lr_boundary(clf); tau = b["tau"] if b else None
    tau_s = "" if tau is None else round(tau, 6)
    wr = SpellingWrecker(corruption_rate=1.0, seed=seed)
    extra = extra or {}

    def flag(w):
        if len(w) < min_word_length: return False, None
        s = det.score_word(w)
        return (clf.predict_error(s) if clf is not None else (s < -10.0)), s

    def tg(w):
        tris, lps, _ = trigram_breakdown(det, w)
        return "|".join(tris), "|".join(f"{x:.3f}" for x in lps)
    rows = []; tp = fp = fn = tn = 0

    for w in words:
        fl, s = flag(w)
        if fl: fp += 1; oc = "fp"
        else:  tn += 1; oc = "tn"
        t, l = tg(w)
        rows.append({**extra, "instance": "clean", "original": w, "corrupted": "",
                    "trigrams": t, "trigram_logprobs": l,
                    "trigram_calc": trigram_calc(det, w),
                    "detection_score": ("" if s is None else round(s, 4)),
                    "tau": tau_s, "flagged": int(fl), "outcome": oc})
        c, gt = wr.corrupt_text(w, min_word_length=min_word_length); c = c.strip()

        if c not in gt:
            rows.append({**extra, "instance": "corrupted", "original": w, "corrupted": c,
                        "trigrams": "", "trigram_logprobs": "", "trigram_calc": "",
                        "detection_score": "",
                        "tau": tau_s, "flagged": "", "outcome": "wasted_noop"})
            continue
        fl, s = flag(c)
        if fl: tp += 1; oc = "tp"
        else:  fn += 1; oc = "fn"
        t, l = tg(c)
        rows.append({**extra, "instance": "corrupted", "original": w, "corrupted": c,
                    "trigrams": t, "trigram_logprobs": l,
                    "trigram_calc": trigram_calc(det, c),
                    "detection_score": ("" if s is None else round(s, 4)),
                    "tau": tau_s, "flagged": int(fl), "outcome": oc})

    return rows, (tp, fp, fn, tn), tau


def write_csv(path, rows, fieldnames=None):
    """
    Writes a list of dict rows to a CSV file and returns its path, or None when there are no rows.
    """
    rows = list(rows)
    if not rows:
        return None
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(p)


def _flag_with_model(model, clf, word, min_word_length=MIN_WORD_LEN):
    """Returns True when a model and a fitted regression flag the word."""
    if len(word) < min_word_length:
        return False
    score = model.score_word(word)
    if clf is None:
        return score < -8.0
    return (clf.coef_[0][0] * score + clf.intercept_[0]) <= 0.0


def evaluate_balanced_model(model, clf, test_words, seed, error_types=None,
                            min_word_length=MIN_WORD_LEN):
    """
    Evaluates a model and a fitted regression on a list of test words. Every word is scored
    once as it is and once corrupted, so the two classes are balanced. Returns the confusion
    counts, macro-F1 and the per-class scores.
    """
    wrecker = SpellingWrecker(corruption_rate=1.0, seed=seed,
                            error_types=error_types)
    corrupted = [wrecker.corrupt_text(w, min_word_length=min_word_length)[0].strip()
                for w in test_words]
    tp = sum(1 for w in corrupted  if _flag_with_model(model, clf, w, min_word_length))
    fp = sum(1 for w in test_words if _flag_with_model(model, clf, w, min_word_length))
    fn = len(corrupted)  - tp
    tn = len(test_words) - fp
    mf1, p_e, r_e, f1_e, f1_c = confusion_to_mf1(tp, fp, fn, tn)
    return {"mf1": mf1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p_e, "recall": r_e,
            "f1_error": f1_e, "f1_correct": f1_c}


def evaluate_balanced_checker(checker, language, test_words, seed,
                            source_language=None,
                            min_word_length=MIN_WORD_LEN, count_mode="type"):
    """
    Evaluates a loaded checker under the balanced protocol, in which every test word is
    checked once as it is and once corrupted.

    Args:
        checker: A SpellChecker with its models loaded.
        language: The language of the test words.
        test_words: The test words.
        seed: Seed of the corruption.
        source_language: The language whose models are applied. It defaults to the language
            of the test words, and differs from it in a transfer run.
        min_word_length: Words shorter than this are skipped.
        count_mode: "type" counts each distinct word once and "token" counts every
            occurrence.

    Returns:
        A dict with the confusion counts, macro-F1 and the per-class scores.
    """
    lang = source_language or language

    wrecker = SpellingWrecker(corruption_rate=1.0, seed=seed)
    corrupted = [wrecker.corrupt_text(w, min_word_length=min_word_length)[0].strip()
                for w in test_words]
    res_c = checker.check(" ".join(corrupted), language=lang,
                        include_suggestions=False, min_word_length=min_word_length)
    flagged_c = {e["word"].lower() for e in res_c["errors"]
                if len(e["word"]) >= min_word_length}
    res_clean = checker.check(" ".join(test_words), language=lang,
                            include_suggestions=False, min_word_length=min_word_length)
    flagged_clean = {e["word"].lower() for e in res_clean["errors"]
                    if len(e["word"]) >= min_word_length}

    if count_mode == "token":
        corr_l = [c.lower() for c in corrupted]; test_l = [w.lower() for w in test_words]
        tp = sum(1 for c in corr_l if c in flagged_c); fn = len(corr_l) - tp
        fp = sum(1 for w in test_l if w in flagged_clean); tn = len(test_l) - fp
    else:
        tp = len(flagged_c     & {w.lower() for w in corrupted})
        fp = len(flagged_clean & {w.lower() for w in test_words})
        fn = len({w.lower() for w in corrupted}  - flagged_c)
        tn = len({w.lower() for w in test_words} - flagged_clean)
    mf1, p_e, r_e, f1_e, f1_c = confusion_to_mf1(tp, fp, fn, tn)
    return {"mf1": mf1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p_e, "recall": r_e,
            "f1_error": f1_e, "f1_correct": f1_c}


def evaluate_at_rate_checker(checker, language, test_words, rate, seed,
                            min_word_length=MIN_WORD_LEN):
    """
    Evaluates a loaded checker on running text in which only a fraction of the words, given
    by rate, is corrupted. Every token is counted once, by its position in the text.
    """
    wrecker = SpellingWrecker(corruption_rate=rate, seed=seed)
    original = " ".join(test_words).split()
    corrupted_text, _ = wrecker.corrupt_text(" ".join(test_words), min_word_length=min_word_length)
    corrupted = corrupted_text.split()
    res = checker.check(corrupted_text, language=language, include_suggestions=False,
                        min_word_length=min_word_length)
    assert len(corrupted) == len(original) == res["total_words"]
    flagged = {e["position"] for e in res["errors"]}
    is_error = [c != o for c, o in zip(corrupted, original)]

    tp = sum(1 for i, err in enumerate(is_error) if err and i in flagged)
    fp = sum(1 for i, err in enumerate(is_error) if not err and i in flagged)
    fn = sum(1 for i, err in enumerate(is_error) if err and i not in flagged)
    tn = sum(1 for i, err in enumerate(is_error) if not err and i not in flagged)
    mf1, p_e, r_e, f1_e, f1_c = confusion_to_mf1(tp, fp, fn, tn)

    return {"mf1": mf1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p_e, "recall": r_e,
            "f1_error": f1_e, "f1_correct": f1_c}


def mean_std(values):
    """Returns the mean and the standard deviation of a list of values."""
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())
