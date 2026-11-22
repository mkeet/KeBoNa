"""
spelling_wrecker.py - generates synthetic spelling errors for evaluation.

One of the four single-edit Damerau operations is applied to a word.

    insertion      umuntu -> umuntuu
    deletion       umuntu -> umunt
    substitution   umuntu -> umuntx
    transposition  umuntu -> umutnu

The corruption type decides where substituted and inserted characters come from. Under
"random" they are drawn uniformly from a-z, and under "qwerty" from the keys next to the
affected character. Deletion and transposition depend only on position, so the two settings
differ only in substitution and insertion.
"""

import random
import string
from typing import Dict, List, Optional, Tuple

# maps each key to the keys next to it on a QWERTY keyboard, diagonal neighbours included
QWERTY_ADJACENT: Dict[str, str] = {
    'q': 'wa',          'w': 'qeasd',       'e': 'wrsdf',
    'r': 'etdfg',       't': 'ryfgh',       'y': 'tughj',
    'u': 'yihjk',       'i': 'uojkl',       'o': 'ipkl',
    'p': 'ol',
    'a': 'qwsz',        's': 'awedxz',      'd': 'serfcx',
    'f': 'drtgvc',      'g': 'ftyhbv',      'h': 'gyujnb',
    'j': 'huikmn',      'k': 'jiolm',       'l': 'kop',
    'z': 'asx',         'x': 'zsdc',        'c': 'xdfv',
    'v': 'cfgb',        'b': 'vghn',        'n': 'bhjm',
    'm': 'njk',
}


class SpellingWrecker:
    """
    Injects spelling errors into text.

    Args:
        corruption_rate: Fraction of the words to corrupt, between 0 and 1.
        seed: Random seed, so that the same errors are generated on every run.
        error_types: The operations to choose from. All four are used by default.
        corruption_type: "random" draws characters uniformly from a-z and "qwerty" draws
            them from the keys next to the affected character.
    """

    ALPHABET = string.ascii_lowercase

    def __init__(
        self,
        corruption_rate: float = 0.10,
        seed: int = 42,
        error_types: Optional[List[str]] = None,
        corruption_type: str = "random",
    ):
        if not 0.0 <= corruption_rate <= 1.0:
            raise ValueError(f"Corruption rate must be [0,1], got {corruption_rate}")
        if corruption_type not in ("random", "qwerty"):
            raise ValueError(f"corruption_type must be 'random' or 'qwerty', got {corruption_type!r}")

        self.corruption_rate = corruption_rate
        self.rng = random.Random(seed)
        self.error_types = error_types or [
            "insertion", "deletion", "substitution", "transposition"]
        self.corruption_type = corruption_type

        self._stats = {
            "words_processed": 0, "words_corrupted": 0,
            "insertions": 0, "deletions": 0,
            "substitutions": 0, "transpositions": 0,
        }

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def corrupt_text(
        self, text: str, min_word_length: int = 3
    ) -> Tuple[str, Dict[str, str]]:
        """
        Corrupts the words of a text, one error per word.

        A corruption that reproduces the original word is discarded, and that word is left as
        it was.

        Args:
            text: The text to corrupt, split on whitespace.
            min_word_length: Words shorter than this are never corrupted.

        Returns:
            The corrupted text, and a dict that maps each corrupted form to its original in
            lower case.
        """
        words = text.split()
        corrupted_words = []
        ground_truth: Dict[str, str] = {}

        for word in words:
            self._stats["words_processed"] += 1
            # letters and hyphens only, with at least one letter
            wreckable = (any(c.isalpha() for c in word)
                        and all(c.isalpha() or c == '-' for c in word))
            if (len(word) >= min_word_length
                    and self.rng.random() < self.corruption_rate
                    and wreckable):
                corrupted = self._corrupt_word(word)
                if corrupted != word.lower():
                    corrupted_words.append(corrupted)
                    ground_truth[corrupted] = word.lower()
                    self._stats["words_corrupted"] += 1
                else:
                    corrupted_words.append(word)
            else:
                corrupted_words.append(word)

        return " ".join(corrupted_words), ground_truth

    def _corrupt_word(self, word: str) -> str:
        word = word.lower()
        error_type = self.rng.choice(self.error_types)
        if   error_type == "insertion":     return self._insert_char(word)
        elif error_type == "deletion":      return self._delete_char(word)
        elif error_type == "substitution":  return self._substitute_char(word)
        elif error_type == "transposition": return self._transpose_chars(word)
        return word

    def _random_char(self) -> str:
        """Draws a character uniformly from a-z."""
        return self.rng.choice(self.ALPHABET)

    def _adjacent_char(self, original: str) -> str:
        """
        Draws a key next to the given character on a QWERTY keyboard, or a uniformly random
        character when it has no neighbours listed.
        """
        neighbours = QWERTY_ADJACENT.get(original, "")
        if not neighbours:
            return self._random_char()
        return self.rng.choice(neighbours)

    def _insert_char(self, word: str) -> str:
        pos = self.rng.randint(0, len(word))
        if self.corruption_type == "qwerty":
            anchor = word[pos] if pos < len(word) else word[pos - 1]
            char = self._adjacent_char(anchor)
        else:
            char = self._random_char()
        self._stats["insertions"] += 1
        return word[:pos] + char + word[pos:]

    def _delete_char(self, word: str) -> str:
        if len(word) <= 1:
            return word
        pos = self.rng.randint(0, len(word) - 1)
        self._stats["deletions"] += 1
        return word[:pos] + word[pos + 1:]

    def _substitute_char(self, word: str) -> str:
        if not word:
            return word
        pos = self.rng.randint(0, len(word) - 1)
        original = word[pos]
        if self.corruption_type == "qwerty":
            neighbours = [c for c in QWERTY_ADJACENT.get(original, "")
                            if c != original]
            new_char = (self.rng.choice(neighbours) if neighbours
                        else self._random_char())
        else:
            new_char = self.rng.choice(
                [c for c in self.ALPHABET if c != original])
        self._stats["substitutions"] += 1
        return word[:pos] + new_char + word[pos + 1:]

    def _transpose_chars(self, word: str) -> str:
        if len(word) <= 1:
            return word
        pos = self.rng.randint(0, len(word) - 2)
        chars = list(word)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        self._stats["transpositions"] += 1
        return "".join(chars)
