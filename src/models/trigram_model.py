"""
trigram_model.py - trigram counts over a sequence of tokens, with no smoothing.

The same class serves two models. The detection model counts character trigrams and scores a
word from them. The word model counts whole words and supplies the candidates and frequencies
for correction.
"""

import logging
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import process as rf_process
from rapidfuzz.distance import OSA

from src.utils.text_utils import TextUtils
from src.utils.trigram_tokenizer import CharTrigramTokenizer

logger = logging.getLogger(__name__)

START_TOKEN = "<START>"
UNSEEN_PENALTY = -20.0

_TOKENIZER = CharTrigramTokenizer(n=3)


class TrigramModel:
    """Counts of each token and of each token given the two tokens before it."""

    def __init__(self, n: int = 3):
        self.n = n

        # maps the two preceding tokens to the counts of each token that follows them
        self.ngrams: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.counts: Counter = Counter()
        self.vocab: Set[str] = set()
        self.total_tokens: int = 0

        self.language: Optional[str] = None
        self.trained_on: Optional[str] = None
        self.tokenization: str = "word"

    def train(
        self,
        corpus_file: Path,
        language: Optional[str] = None,
        tokenization: str = "word",
    ) -> None:
        """
        Counts the tokens of a corpus file in which each line holds one word.

        For the word model the line is the word itself. For the detection model it is the word
        written as its character trigrams, separated by spaces. Each line is padded on the left
        with START tokens, so the counts never run across two words.

        Args:
            corpus_file: Path to the corpus file.
            language: Name of the language, stored with the model.
            tokenization: "word" or "trigram", stored with the model to say which of the two
                it is.
        """
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

        self.language = language
        self.trained_on = str(corpus_file)
        self.tokenization = tokenization

        logger.info(
            f"Training {self.n}-gram model on {corpus_file} "
            f"(tokenization={tokenization})..."
        )

        line_count = 0
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                tokens = TextUtils.tokenize(line)
                if not tokens:
                    continue

                padded = [START_TOKEN] * (self.n - 1) + tokens

                for token in tokens:
                    self.vocab.add(token)
                    self.counts[token] += 1
                    self.total_tokens += 1

                for i in range(len(padded) - self.n + 1):
                    ngram = tuple(padded[i : i + self.n])
                    self.ngrams[ngram[:-1]][ngram[-1]] += 1

                line_count += 1

        logger.info(
            f"Training complete. {line_count} words processed, "
            f"{self.total_tokens} tokens, {len(self.vocab)} unique tokens"
        )

        self._context_totals = None
        self._build_length_index()

    def score_word(self, word: str) -> float:
        """
        Scores a word by the mean log-probability of each of its character trigrams given the
        two trigrams before it.

        No smoothing is applied, so a context or a transition never seen in training contributes
        a fixed penalty of -20.

        Args:
            word: The word to score. It is lower-cased before it is segmented.

        Returns:
            The score. A word with no trigram, such as one of fewer than three characters,
            scores -20.
        """
        trigrams = _TOKENIZER.segment_word(word)
        if not trigrams:
            return UNSEEN_PENALTY

        # the denominators are summed once per model and are not pickled
        totals = getattr(self, "_context_totals", None)
        if totals is None:
            totals = self._context_totals = {c: sum(cnt.values())
                                            for c, cnt in self.ngrams.items()}

        padded = [START_TOKEN, START_TOKEN] + trigrams
        total_log_p = 0.0
        for i in range(2, len(padded)):
            token = padded[i]
            ctx = tuple(padded[i - 2 : i])
            total = totals.get(ctx, 0)
            prob = self.ngrams[ctx].get(token, 0) / total if total > 0 else 0.0
            total_log_p += math.log(prob) if prob > 0 else UNSEEN_PENALTY

        return total_log_p / len(trigrams)

    def suggest_corrections(
        self,
        word: str,
        max_suggestions: int = 5,
        max_edit_distance: int = 2,
        tie_breaker_model=None,
    ) -> List[Tuple[str, float]]:
        """
        Finds the vocabulary words closest to a word.

        Distance is optimal string alignment, which counts an adjacent transposition as one edit.
        It is computed on lower-cased forms, so a difference in case costs no edit, but candidates
        are returned in their original casing. Candidates are ranked by distance, then by
        descending frequency, then by the score of the tie-breaker model. Where two candidates
        differ only in case, the higher-ranked one is kept.

        Args:
            word: The word to correct.
            max_suggestions: Number of candidates to return.
            max_edit_distance: Largest distance at which a word is still a candidate.
            tie_breaker_model: Detection model whose score orders candidates of equal distance
                and frequency.

        Returns:
            A list of (candidate, corpus frequency) pairs, best first.
        """
        word_lower = word.lower()
        min_len = max(1, len(word) - max_edit_distance)
        max_len = len(word) + max_edit_distance

        candidates = []
        for length in range(min_len, max_len + 1):
            originals = self._vocab_by_length.get(length)
            if not originals:
                continue
            matches = rf_process.extract(
                word_lower, self._vocab_by_length_lower[length], scorer=OSA.distance,
                score_cutoff=max_edit_distance, limit=None)
            for _, dist, idx in matches:
                vocab_word = originals[idx]
                if vocab_word == word:
                    continue
                freq = self.counts.get(vocab_word, 0)
                tie_score = tie_breaker_model.score_word(vocab_word) if tie_breaker_model else 0.0
                candidates.append((dist, -freq, -tie_score, vocab_word))

        candidates.sort()

        seen_lower = set()
        deduped = []
        for _, neg_freq, _, vocab_word in candidates:
            lowered = vocab_word.lower()
            if lowered not in seen_lower:
                seen_lower.add(lowered)
                deduped.append((vocab_word, -neg_freq))

        return deduped[:max_suggestions]

    def save(self, filepath: Path) -> None:
        """Saves the counts, the vocabulary and the metadata to a pickle file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n": self.n,
            "ngrams": dict(self.ngrams),
            "counts": dict(self.counts),
            "vocab": self.vocab,
            "total_tokens": self.total_tokens,
            "language": self.language,
            "trained_on": self.trained_on,
            "tokenization": self.tokenization,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Model saved to {filepath} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, filepath: Path) -> "TrigramModel":
        """Loads a model saved by save() and returns it ready for use."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        model = cls(n=data["n"])
        model.ngrams = defaultdict(Counter, data["ngrams"])
        model.counts = Counter(data["counts"])
        model.vocab = data["vocab"]
        model.total_tokens = data["total_tokens"]
        model.language = data.get("language")
        model.trained_on = data.get("trained_on")
        model.tokenization = data.get("tokenization", "word")

        logger.info(
            f"Model loaded: {model.language}, "
            f"{len(model.vocab)} words, "
            f"{model.total_tokens} total tokens"
        )

        model._build_length_index()

        return model

    def get_info(self) -> Dict:
        """Returns a summary of the model as a dict."""
        return {
            "language": self.language,
            "n": self.n,
            "vocab_size": len(self.vocab),
            "total_tokens": self.total_tokens,
            "unique_trigrams": sum(len(v) for v in self.ngrams.values()),
            "trained_on": self.trained_on,
            "tokenization": self.tokenization,
        }

    def _build_length_index(self) -> None:
        """
        Groups the vocabulary by word length, so that suggest_corrections compares a word
        only with candidates of a similar length. A lower-cased copy of each group is kept
        beside it, in the same order, for the distance computation.
        """
        by_length = defaultdict(list)
        for w in self.vocab:
            by_length[len(w)].append(w)
        self._vocab_by_length: Dict[int, List[str]] = dict(by_length)
        self._vocab_by_length_lower: Dict[int, List[str]] = {
            length: [w.lower() for w in words]
            for length, words in self._vocab_by_length.items()
        }
