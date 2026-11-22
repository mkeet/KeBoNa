"""
trigram_tokenizer.py - splits a word into its character trigrams.
"""

from typing import List


class CharTrigramTokenizer:
    """
    A sliding window of three characters over a word, so that umuntu gives umu, mun, unt
    and ntu.
    """

    def __init__(self, n: int = 3):
        self.n = n

    def segment_word(self, word: str) -> List[str]:
        """
        Lower-cases the word and returns its windows in order. A window that holds anything other
        than letters or hyphens is dropped, and a word shorter than the window gives an empty list.
        """
        if len(word) < self.n:
            return []
        word = word.lower()
        trigrams = [word[i:i+self.n] for i in range(len(word)-self.n+1)]
        return [t for t in trigrams if all(c.isalpha() or c == '-' for c in t)]
