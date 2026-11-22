"""
test_spelling_wrecker.py - tests for the synthetic error generator.

Covers the corruption rate, reproducibility with a seed, the error types and the edge cases.
"""
from src.evaluation.spelling_wrecker import SpellingWrecker


def _corrupt(text, rate, seed=42):
    """Corrupt `text` at `rate` and return only the corrupted string (corrupt_text returns
    (corrupted_text, ground_truth))."""
    return SpellingWrecker(seed=seed, corruption_rate=rate).corrupt_text(text)[0]


class TestCorruption:
    def test_text_is_modified(self):
        original = "umuntu ngumuntu ngabantu inkosi yezwe enkulu abantu bahamba"
        assert _corrupt(original, 1.0) != original

    def test_zero_rate_no_change(self):
        original = "umuntu ngumuntu ngabantu"
        assert _corrupt(original, 0.0) == original

    def test_partial_rate(self):
        """With rate=0.5, some words change and some don't."""
        original = "umuntu ngumuntu ngabantu inkosi yezwe enkulu"
        corrupted = _corrupt(original, 0.5)
        orig_words, corr_words = original.split(), corrupted.split()
        same = sum(1 for o, c in zip(orig_words, corr_words) if o == c)
        assert 0 < same < len(orig_words)


class TestReproducibility:
    def test_same_seed_same_output(self):
        text = "umuntu ngumuntu ngabantu inkosi yezwe"
        assert _corrupt(text, 0.5, seed=42) == _corrupt(text, 0.5, seed=42)

    def test_different_seeds_different_output(self):
        text = "umuntu ngumuntu ngabantu inkosi yezwe enkulu abantu bahamba"
        assert _corrupt(text, 0.8, seed=42) != _corrupt(text, 0.8, seed=99)


class TestErrorTypes:
    def test_insertion(self):
        """Across many seeds, at least one corruption is longer (an insertion)."""
        word = "umuntu"
        lengths = [len(_corrupt(word, 1.0, seed=i)) for i in range(100)]
        assert max(lengths) > len(word)

    def test_deletion(self):
        """Across many seeds, at least one corruption is shorter (a deletion)."""
        word = "umuntu"
        lengths = [len(_corrupt(word, 1.0, seed=i)) for i in range(100)]
        assert min(lengths) < len(word)


class TestEdgeCases:
    def test_empty_text(self):
        assert _corrupt("", 0.5) == ""

    def test_single_char_word(self):
        # Single-char words are below min length, so they pass through unchanged.
        assert isinstance(_corrupt("a", 1.0), str)

    def test_single_word(self):
        result = _corrupt("umuntu", 1.0)
        assert isinstance(result, str)
        assert len(result) > 0
