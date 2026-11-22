"""
corpus_cleaner.py - turns raw corpus files into a list of words.

Every file under data/corpora/<lang>/raw/ is read, its tokens are normalised and filtered,
and the words that remain are written one per line to data/corpora/<lang>/cleaned.txt.
"""

import logging
import unicodedata
from pathlib import Path
from typing import List, Optional

from src.utils.text_utils import TextUtils

logger = logging.getLogger(__name__)

try:
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 42
except ImportError:
    detect_langs = None

# English, French and Portuguese function words, because langdetect is unreliable on single words
_FOREIGN_STOPWORDS = frozenset({
    # English
    'the', 'and', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'of', 'in', 'to', 'for',
    'on', 'with', 'at', 'by', 'from', 'not', 'but', 'or', 'if', 'this',
    'that', 'it', 'its', 'he', 'she', 'they', 'we', 'you', 'my', 'your',
    'his', 'her', 'our', 'their',
    # French
    'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est',
    'en', 'il', 'elle', 'ils', 'elles', 'nous', 'vous', 'son', 'sa',
    'ses', 'sur', 'dans', 'avec', 'par', 'que', 'qui',
    # Portuguese
    'os', 'as', 'ao', 'da', 'do', 'dos', 'das', 'uma', 'para', 'com',
    'por', 'mas', 'ele', 'ela', 'eles', 'elas', 'seu', 'sua',
})

# English content words that are frequent across the corpora
_FOREIGN_CONTENT_WORDS = frozenset({
    "about", "academy", "action", "affairs", "agency", "agenda", "agricultural", "agriculture",
    "airport", "album", "alliance", "assembly", "association", "august", "authority", "avenue",
    "award", "awards", "banana", "beach", "bible", "billion", "bishop", "black", "board",
    "bridge", "brown", "building", "bureau", "business", "capital", "center", "central",
    "certificate", "challenge", "champions", "championship", "change", "chief", "child",
    "church", "close", "coast", "college", "commission", "committee", "commonwealth",
    "community", "company", "conference", "congress", "convention", "corporation", "council",
    "country", "court", "credit", "cross", "culture", "current", "daily", "delta", "democratic",
    "department", "design", "development", "diamond", "district", "doctor", "drama", "dream",
    "eastern", "economic", "education", "email", "energy", "exchange", "faith", "falls",
    "family", "female", "festival", "financial", "first", "football", "force", "forum",
    "foundation", "frank", "freedom", "front", "games", "general", "girls", "global", "golden",
    "government", "grace", "grade", "grand", "great", "green", "group", "health", "heart",
    "history", "hospital", "hotel", "hotels", "house", "human", "image", "information",
    "institute", "international", "internet", "island", "japan", "jersey", "jimmy", "junior",
    "justice", "kingdom", "language", "league", "library", "limited", "links", "local", "lodge",
    "magazine", "magic", "management", "manager", "march", "maria", "market", "martin",
    "master", "media", "million", "minister", "mission", "money", "months", "mountain",
    "movement", "movie", "municipality", "museum", "music", "natal", "national", "nations",
    "nature", "network", "newton", "north", "northern", "number", "office", "official",
    "online", "orange", "order", "organization", "original", "paper", "parliament", "party",
    "pasha", "peace", "people", "peter", "photo", "photos", "pictures", "point", "police",
    "policy", "power", "premier", "president", "primary", "prime", "prince", "private",
    "profile", "programme", "project", "public", "queen", "radio", "records", "relations",
    "report", "republic", "research", "resources", "revenue", "review", "rights", "river",
    "royal", "saint", "school", "science", "search", "secondary", "security", "service",
    "services", "single", "smith", "soccer", "social", "society", "songs", "south", "special",
    "sport", "sports", "square", "stadium", "standard", "stars", "state", "states", "station",
    "story", "street", "studies", "studio", "super", "system", "table", "technology", "thank",
    "theatre", "total", "tower", "trade", "training", "translation", "trust", "twitter",
    "under", "union", "united", "university", "valley", "victor", "video", "villa", "village",
    "watch", "water", "website", "western", "white", "woman", "women", "works", "world",
    "young", "youth"
    "arts", "bank", "city", "club", "east", "park", "road", "town", "west",
})

# characters that are never valid inside a word token (hyphens are allowed)
_INVALID_INTERNAL_CHARS = frozenset(
    '"\'`""''.,;:!?()[]{}|\\<>@#$%^&*+=~/_'
)

_LATIN_CHAR_CACHE: dict = {}


def is_latin_letter(ch: str) -> bool:
    """
    Returns True for a letter of the Latin script. The answer is cached per character. A
    character that is not a letter also returns True, because the other filters handle it.
    """
    if not ch.isalpha():
        return True
    cached = _LATIN_CHAR_CACHE.get(ch)
    if cached is None:
        try:
            cached = "LATIN" in unicodedata.name(ch)
        except ValueError:
            cached = False
        _LATIN_CHAR_CACHE[ch] = cached
    return cached


def has_non_latin_token(token: str) -> bool:
    """Returns True if the token contains a letter from another script."""
    return any(not is_latin_letter(c) for c in token)


class CorpusCleaner:
    """Cleans the raw files of one corpus into a single word list, one word per line."""

    def __init__(self, min_word_length: int = 2):
        self.min_word_length = min_word_length
        self._stats = {
            "files_processed": 0,
            "lines_read": 0,
            "lines_dropped_foreign": 0,
            "tokens_extracted": 0,
            "tokens_after_filter": 0,
            "words_written": 0,
        }

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def clean_directory(
        self,
        input_dir: Path,
        output_file: Path,
        file_extensions: Optional[List[str]] = None,
    ) -> Path:
        """
        Cleans every file under a folder, including its subfolders, into one word list.

        Args:
            input_dir: Folder holding the raw files.
            output_file: File that receives the words, one per line.
            file_extensions: Extensions of the files to read. Only .txt is read by default.

        Returns:
            The path of the output file.
        """
        if file_extensions is None:
            file_extensions = [".txt"]

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        files = []
        for ext in file_extensions:
            files.extend(sorted(input_dir.rglob(f"*{ext}")))

        if not files:
            raise FileNotFoundError(
                f"No files with extensions {file_extensions} found in {input_dir}"
            )

        logger.info(f"Found {len(files)} files in {input_dir}")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        all_words: List[str] = []
        for filepath in files:
            logger.info(f"Processing: {filepath.name}")
            all_words.extend(self._process_file(filepath))
            self._stats["files_processed"] += 1

        with open(output_file, "w", encoding="utf-8") as f:
            for word in all_words:
                if word.strip():
                    f.write(word + "\n")
                    self._stats["words_written"] += 1

        logger.info(
            f"Cleaning complete. {self._stats['words_written']} words "
            f"written to {output_file} as a vertical list."
        )
        return output_file

    def _process_file(self, filepath: Path) -> List[str]:
        """
        Chooses how to read a file. A tab in any of its first five lines marks it as
        tab-separated, and anything else is read as plain text.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                first_lines = [f.readline() for _ in range(5)]
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="latin-1") as f:
                first_lines = [f.readline() for _ in range(5)]

        is_sadii = any("\t" in line for line in first_lines if line.strip())
        if is_sadii:
            return self._process_sadii(filepath)
        return self._process_plain_text(filepath)

    def _is_foreign_sentence(self, text: str) -> bool:
        """
        Returns True when langdetect identifies a line of at least four words as English, French,
        Portuguese or Spanish with a probability above 0.9.
        """
        if detect_langs is None or len(text.split()) < 4:
            return False
        try:
            results = detect_langs(text)
            if results:
                top = results[0]
                return top.lang in ('en', 'fr', 'pt', 'es') and top.prob > 0.9  # native NCB text mis-tags near 0.7
        except Exception:
            pass
        return False

    def _process_plain_text(self, filepath: Path) -> List[str]:
        output_lines = []
        encoding = self._detect_encoding(filepath)

        with open(filepath, "r", encoding=encoding) as f:
            for line in f:
                self._stats["lines_read"] += 1
                line = line.strip()
                if not line or TextUtils.is_metadata_line(line):
                    continue
                if self._is_foreign_sentence(line):
                    self._stats["lines_dropped_foreign"] += 1
                    continue
                normalized = TextUtils.normalize(line)
                tokens = TextUtils.tokenize(normalized)
                for token in self._filter_tokens(tokens):
                    output_lines.append(token)

        return output_lines

    def _process_sadii(self, filepath: Path) -> List[str]:
        """
        Reads a tab-separated file. In a SADII file the first field of a line is the word form,
        and that word is kept. In a Leipzig file each line is a number, a tab and a sentence,
        so the line is cleaned as plain text.
        """
        output_lines = []
        encoding = self._detect_encoding(filepath)

        with open(filepath, "r", encoding=encoding) as f:
            for line in f:
                self._stats["lines_read"] += 1
                line = line.strip()
                if not line or TextUtils.is_metadata_line(line):
                    continue

                if "\t" in line:
                    token = TextUtils.extract_sadii_tokens(line)
                    if token and ' ' not in token:
                        self._stats["tokens_extracted"] += 1
                        filtered = self._filter_tokens([token])
                        if filtered:
                            output_lines.append(filtered[0])
                        continue
                    content = line.split('\t', 1)[-1]
                else:
                    content = line

                if self._is_foreign_sentence(content):
                    self._stats["lines_dropped_foreign"] += 1
                    continue

                normalized = TextUtils.normalize(line)
                tokens = TextUtils.tokenize(normalized)
                for tok in self._filter_tokens(tokens):
                    output_lines.append(tok)

        return output_lines

    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Keeps the tokens that start with a letter, consist only of Latin letters and hyphens,
        meet the minimum length and are not in the two lists of foreign words.
        """
        ANNOTATION_TAGS = {'Punc', 'PUNC', 'NUM', 'punc', 'num', 'SENT', 'sent'}
        result = []
        for token in tokens:
            if token in ANNOTATION_TAGS:
                continue
            if not token[0].isalpha():
                continue
            if has_non_latin_token(token):
                continue
            if any(c in _INVALID_INTERNAL_CHARS for c in token):
                continue
            if any(c.isdigit() for c in token):
                continue
            if len(token) < self.min_word_length:
                continue
            if token.lower() in _FOREIGN_STOPWORDS:
                continue
            if token.lower() in _FOREIGN_CONTENT_WORDS:
                continue
            result.append(token)
            self._stats["tokens_after_filter"] += 1
        return result

    def _detect_encoding(self, filepath: Path) -> str:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                f.read()
            return "utf-8"
        except UnicodeDecodeError:
            return "latin-1"
