"""
setup.py - cleans a corpus and trains the three models of a language in one command.

    python setup.py                 every language found in data/corpora/
    python setup.py --lang zulu     one language
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.threshold_classifier import ThresholdClassifier
from src.models.trigram_model import TrigramModel
from src.utils.corpus_cleaner import CorpusCleaner
from src.utils.data_splitter import DataSplitter
from src.utils.trigram_tokenizer import CharTrigramTokenizer

logger = logging.getLogger(__name__)


def discover_languages() -> list:
    """Returns every folder of data/corpora/ that has a raw/ subfolder."""
    corpora_dir = PROJECT_ROOT / "data" / "corpora"
    if not corpora_dir.exists():
        return []
    languages = []
    for lang_dir in sorted(corpora_dir.iterdir()):
        if lang_dir.is_dir() and (lang_dir / "raw").exists():
            languages.append(lang_dir.name)
    return languages


def clean_corpus(lang: str) -> bool:
    """Step 1 writes the raw files of the language to cleaned.txt as one word per line."""
    raw_dir = PROJECT_ROOT / "data" / "corpora" / lang / "raw"
    out_file = PROJECT_ROOT / "data" / "corpora" / lang / "cleaned.txt"
    if not raw_dir.exists():
        logger.error("[%s] Raw corpus directory not found: %s", lang, raw_dir)
        return False

    logger.info("[%s] Step 1 - Cleaning corpus (word-level) [foreign-sentence filter ON]...", lang)

    cleaner = CorpusCleaner(min_word_length=2)
    try:
        cleaner.clean_directory(raw_dir, out_file)
        s = cleaner.stats
        drop_msg = (f", {s['lines_dropped_foreign']} foreign lines dropped"
                    if s.get("lines_dropped_foreign") else "")
        logger.info("[%s]   -> %d files, %d words written%s",
                    lang, s["files_processed"], s["words_written"], drop_msg)
        return True
    except FileNotFoundError as e:
        logger.error("[%s] %s", lang, e)
        return False


def split_corpus(lang: str) -> bool:
    """Step 2 splits cleaned.txt into train.txt, test.txt and validation.txt."""
    in_file = PROJECT_ROOT / "data" / "corpora" / lang / "cleaned.txt"
    out_dir = PROJECT_ROOT / "data" / "corpora" / lang
    if not in_file.exists():
        logger.error("[%s] cleaned.txt missing - run step 1 first.", lang)
        return False
    logger.info("[%s] Step 2 - Splitting 80/10/10 (seed=42)...", lang)
    splitter = DataSplitter(0.80, 0.10, 0.10, random_seed=42)
    train, test, val = splitter.split_file(in_file, out_dir)
    logger.info("[%s]   -> train:%d  test:%d  val:%d",
                lang, _count(train), _count(test), _count(val))
    return True


def train_word_model(lang: str) -> bool:
    """Step 3 trains the word model, which supplies the candidates for correction."""
    train_file = PROJECT_ROOT / "data" / "corpora" / lang / "train.txt"
    out_path   = PROJECT_ROOT / "data" / "models" / f"{lang}_word.pkl"
    if not train_file.exists():
        logger.error("[%s] train.txt missing - run steps 1-2 first.", lang)
        return False
    logger.info("[%s] Step 3 - Training word-level model...", lang)
    model = TrigramModel(n=3)
    model.train(train_file, language=lang, tokenization="word")
    model.trained_on = _relative(train_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    info = model.get_info()
    logger.info("[%s]   -> vocab:%d  trigrams:%d  -> %s",
                lang, info["vocab_size"], info["unique_trigrams"], out_path.name)
    return True


def build_trigram_corpus(lang: str) -> bool:
    """
    Step 4 rewrites each word of train.txt as its character trigrams. Only the training split
    is used, so the detection model never sees a validation or test word.
    """
    train_words = PROJECT_ROOT / "data" / "corpora" / lang / "train.txt"
    out_file = PROJECT_ROOT / "data" / "corpora" / lang / "train_trigram.txt"
    if not train_words.exists():
        logger.error("[%s] train.txt not found - run step 2 first.", lang)
        return False
    logger.info("[%s] Step 4 - Building char-trigram corpus from train.txt...", lang)
    tok = CharTrigramTokenizer(n=3)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(train_words, "r", encoding="utf-8") as fin, \
            open(out_file, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                trigrams = []
                for word in line.split():
                    trigrams.extend(tok.segment_word(word))
                if trigrams:
                    fout.write(" ".join(trigrams) + "\n")
        logger.info("[%s]   -> Written to %s", lang, out_file.name)
        return True
    except Exception as e:
        logger.error("[%s] Char-trigram corpus build failed: %s", lang, e)
        return False


def train_trigram_model(lang: str) -> bool:
    """Step 5 trains the detection model on the character trigrams of the training split."""
    train_file = PROJECT_ROOT / "data" / "corpora" / lang / "train_trigram.txt"
    out_path = PROJECT_ROOT / "data" / "models" / f"{lang}_trigram.pkl"
    if not train_file.exists():
        logger.warning("[%s] train_trigram.txt missing.", lang)
        return False
    logger.info("[%s] Step 5 - Training char-trigram model...", lang)
    model = TrigramModel(n=3)
    model.train(train_file, language=lang, tokenization="trigram")
    model.trained_on = _relative(train_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    info = model.get_info()
    logger.info("[%s]   -> vocab:%d  trigrams:%d  -> %s",
                lang, info["vocab_size"], info["unique_trigrams"], out_path.name)
    return True


def train_classifier(lang: str):
    """Step 6 fits the threshold of the detection model on the validation split."""
    trigram_model_path = PROJECT_ROOT / "data" / "models" / f"{lang}_trigram.pkl"
    val_file = PROJECT_ROOT / "data" / "corpora" / lang / "validation.txt"

    if trigram_model_path.exists() and val_file.exists():
        logger.info("[%s] Step 6 - Training LR classifier...", lang)
        try:
            model = TrigramModel.load(trigram_model_path)
            clf = ThresholdClassifier()
            clf.train(model, val_file, language=lang, corruption_rate=1.0, seed=42)
            if clf.is_trained:
                out_path = PROJECT_ROOT / "data" / "models" / f"{lang}_classifier.pkl"
                clf.save(out_path)
                logger.info("[%s]   -> LR classifier saved.", lang)
        except Exception as e:
            logger.error("[%s] LR training failed: %s", lang, e)
    else:
        logger.warning("[%s] Skipping LR: model or validation file missing.", lang)


def setup_language(lang: str) -> dict:
    """Runs the six steps for one language and returns which of them succeeded."""
    results = {}
    results["clean_word"]      = clean_corpus(lang)
    results["split_word"]      = split_corpus(lang) if results["clean_word"] else False
    results["train_word"]      = train_word_model(lang) if results["split_word"] else False
    results["build_char_tri"]  = build_trigram_corpus(lang) if results["split_word"] else False
    results["train_detection"] = train_trigram_model(lang) if results["build_char_tri"] else False
    train_classifier(lang)
    return results


def main():
    parser = argparse.ArgumentParser(description="Clean a corpus and train its models.")
    parser.add_argument("--lang", default=None,
                        help="a folder of data/corpora/ (default: every folder with a raw/ subfolder)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    languages = [args.lang] if args.lang else discover_languages()
    all_ok = True

    for lang in languages:
        print(f"\n{'='*55}")
        print(f"  Setting up: {lang.upper()}")
        print(f"{'='*55}")
        results = setup_language(lang)
        failed = [k for k, v in results.items() if not v]
        if failed:
            logger.warning("[%s] Steps with issues: %s", lang, failed)
            all_ok = False
        else:
            logger.info("[%s] All steps completed successfully.", lang)

    print(f"\n{'='*55}")
    if all_ok:
        print("  Setup complete.")
    else:
        print("  Finished with warnings (see above).")
    print(f"{'='*55}\n")


def _relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _count(path: Path) -> int:
    try:
        return sum(1 for _ in open(path, encoding="utf-8"))
    except Exception:
        return 0


if __name__ == "__main__":
    main()
