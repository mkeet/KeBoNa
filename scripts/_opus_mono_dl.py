"""
_opus_mono_dl.py - downloads monolingual corpora from the OPUS object store into
data/corpora/<lang>/raw/.

The files are fetched directly over HTTPS from

    https://object.pouta.csc.fi/OPUS-<CORPUS>/<VERSION>/mono/<code>.txt.gz

where <code> is the OPUS language code, usually ISO 639-3. The NLLB and CCMatrix monolingual
text is mined from the web, so it is noisier and has more short tokens than Wikipedia or
Glot500. Corpora based on the Bible carry archaic vocabulary.
"""

import gzip
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent.parent

CORPUS, VERSION = "NLLB", "v1"
# (OPUS code, corpus folder, Guthrie code)
TARGETS = [
    ("umb", "umbundu",  "R11"),
    ("lua", "luba",     "L31"),
    ("kam", "kamba",    "E55"),
    ("kmb", "kimbundu", "H21"),
]
BASE = "https://object.pouta.csc.fi/OPUS-{c}/{v}/mono/{code}.txt.gz"


def main():
    """Downloads and unpacks the corpus of each language listed in TARGETS."""
    for code, lang, guth in TARGETS:
        url = BASE.format(c=CORPUS, v=VERSION, code=code)
        out = ROOT / "data" / "corpora" / lang / "raw" / f"{lang}_{CORPUS.lower()}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[..]   {guth} {lang:<10} downloading <- {url}", flush=True)
        try:
            tmp = out.with_suffix(".gz")
            def _progress(block, bsize, total):
                got = block * bsize
                if total > 0 and block % 200 == 0:
                    print(f"       {lang}: {got/1e6:.1f} / {total/1e6:.1f} MB", flush=True)
            urllib.request.urlretrieve(url, tmp, reporthook=_progress)
            print(f"       {lang}: decompressing {tmp.name} ...", flush=True)
            with gzip.open(tmp, "rb") as f_in, open(out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            tmp.unlink()
            n = sum(1 for _ in open(out, encoding="utf-8", errors="ignore"))
            print(f"[OK]   {guth} {lang:<10} {n} lines <- OPUS-{CORPUS} mono/{code}", flush=True)
        except Exception as e:
            print(f"[FAIL] {guth} {lang:<10} {type(e).__name__}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
