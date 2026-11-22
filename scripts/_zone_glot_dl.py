"""
_zone_glot_dl.py - downloads corpora from cis-lmu/Glot500 for the Guthrie zones that Wikipedia
does not cover.

The text is written to data/corpora/<lang>/raw/<lang>_glot.txt, so an existing raw file is
never overwritten. Each target is (Glot500 config, folder, zone, Guthrie code).
"""

from pathlib import Path
import time

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    ("lua_Latn", "luba",     "L", "L31"),  # Luba-Kasai, zone L
    ("yao_Latn", "yao",      "P", "P21"),  # Yao, zone P
    ("umb_Latn", "umbundu",  "R", "R11"),  # Umbundu, zone R
    ("tum_Latn", "tumbuka",  "N", "N21"),  # Tumbuka, zone N
    ("toi_Latn", "tonga",    "M", "M64"),  # Tonga, zone M
]
LIMIT = 200_000


def main():
    """Streams each target from Glot500 and writes its lines to the raw folder of the language."""
    for cfg, lang, zone, guth in TARGETS:
        out = ROOT / "data" / "corpora" / lang / "raw" / f"{lang}_glot.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        print(f"[..]   {zone} {lang:<10} connecting -> huggingface.co/datasets/cis-lmu/Glot500 "
            f"[{cfg}]", flush=True)
        try:
            ds = load_dataset("cis-lmu/Glot500", cfg, split="train", streaming=True)
            n = words = 0
            with open(out, "w", encoding="utf-8") as f:
                for ex in ds:
                    txt = (ex.get("text") or "").strip()
                    if txt:
                        f.write(txt + "\n"); n += 1; words += len(txt.split())
                    if n % 5000 == 0 and n:
                        print(f"       {lang}: {n} lines ...", flush=True)
                    if n >= LIMIT:
                        break
            print(f"[OK]   {zone} {guth} {lang:<10} lines={n:<7} ~words={words:<9} "
                f"{time.time()-t:.0f}s", flush=True)
        except Exception as e:
            print(f"[FAIL] {zone} {lang:<10} {type(e).__name__}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
