"""
download_hf_corpus.py - downloads a monolingual corpus from a HuggingFace dataset.

The text is written to data/corpora/<lang>/raw/<lang>.txt, which is where setup.py looks for
it. The dataset is streamed line by line and the download stops at --limit, so the whole
dataset is never fetched. The datasets package is required.

    python scripts/download_hf_corpus.py --dataset statmt/cc100 --config sw --lang swahili --limit 200000
    python scripts/download_hf_corpus.py --dataset <repo> --config loz --lang silozi --text-field text --limit 100000

The models are then trained with python setup.py --lang <lang>.
"""

import argparse
from pathlib import Path
import sys

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent


def main():
    """Streams the dataset and writes its non-empty lines to the raw folder of the language."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. statmt/cc100")
    ap.add_argument("--config", default=None, help="config / language code, e.g. sw")
    ap.add_argument("--lang", required=True, help="output corpus folder name, e.g. swahili")
    ap.add_argument("--text-field", default="text", help="field holding the text (default 'text')")
    ap.add_argument("--translation-key", default=None,
                    help="for parallel datasets (e.g. NLLB): take ex['translation'][KEY], "
                         "e.g. loz_Latn")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=200000, help="max non-empty lines to write")
    a = ap.parse_args()

    kw = {"split": a.split, "streaming": True}
    # some datasets take the config as a positional name and CC-100 takes it as lang
    try:
        ds = (load_dataset(a.dataset, a.config, **kw) if a.config
              else load_dataset(a.dataset, **kw))
    except Exception as e1:
        try:
            ds = load_dataset(a.dataset, lang=a.config, **kw)
        except Exception as e2:
            sys.exit(f"Could not load. Tried name= ({e1}) and lang= ({e2}).\n"
                     f"Check the dataset's 'Use this dataset' snippet on HF for the "
                     f"exact config/field names.")

    out = ROOT / "data" / "corpora" / a.lang / "raw" / f"{a.lang}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for ex in ds:
            if a.translation_key:
                t = ((ex.get("translation") or {}).get(a.translation_key) or "").strip()
            else:
                t = (ex.get(a.text_field) or "").strip()
            if t:
                f.write(t + "\n"); n += 1
            if n % 20000 == 0 and n:
                print(f"  ... {n} lines", flush=True)
            if n >= a.limit:
                break
    print(f"wrote {n} lines -> {out}")
    print(f"next: venv/bin/python setup.py --lang {a.lang}")


if __name__ == "__main__":
    main()
