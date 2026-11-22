"""
_zone_wiki_dl.py - downloads Wikipedia text for one or more languages per Guthrie zone, from
wikimedia/wikipedia.

The text is written to data/corpora/<lang>/raw/<lang>_wiki.txt, so an existing raw file is
never overwritten. Each target is (Wikipedia code, folder, zone). The line and word counts of
each language are printed.
"""

from pathlib import Path
import sys
import time

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (Wikipedia code, corpus folder as in GUTHRIE_CODES, zone)
TARGETS = [
    ("rw",  "kinyarwanda", "J"),   # JD61
    ("rn",  "kirundi",     "J"),   # JD62
    ("lg",  "luganda",     "J"),   # JE15
    ("ki",  "kikuyu",      "E"),   # E51
    ("ny",  "chichewa",    "N"),   # N31
    ("bem", "bemba",       "M"),   # M42
    ("ln",  "lingala",     "C"),   # C36
    ("kg",  "kongo",       "H"),  # H16
]
SNAPSHOT = "20231101"
LIMIT = 200_000  # cap on articles, which the small Wikipedias never reach


def main():
    """
    Streams each target from Wikipedia and writes its articles to the raw folder of the language.
    """
    for code, lang, zone in TARGETS:
        out = ROOT / "data" / "corpora" / lang / "raw" / f"{lang}_wiki.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        src = f"huggingface.co/datasets/wikimedia/wikipedia [{SNAPSHOT}.{code}]"
        print(f"[..]   {zone} {lang:<12} connecting -> {src}", flush=True)
        try:
            ds = load_dataset("wikimedia/wikipedia", f"{SNAPSHOT}.{code}",
                            split="train", streaming=True)
            n = words = 0
            with open(out, "w", encoding="utf-8") as f:
                for ex in ds:
                    txt = (ex.get("text") or "").strip()
                    if txt:
                        f.write(txt + "\n"); n += 1; words += len(txt.split())
                    if n % 2000 == 0 and n:
                        print(f"       {lang}: {n} articles, ~{words} words ...", flush=True)
                    if n >= LIMIT:
                        break
            print(f"[OK]   {zone} {lang:<12} ({code}) articles={n:<6} ~words={words:<9} "
                f"{time.time()-t:.0f}s -> {out.relative_to(ROOT)}", flush=True)
        except Exception as e:
            print(f"[FAIL] {zone} {lang:<12} ({code}) {type(e).__name__}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
