"""
test_text_utils.py - tests for the text utilities.
"""

from src.utils.text_utils import TextUtils


class TestNormalize:
    def test_lowercase(self):
        # normalize() preserves case by default
        assert TextUtils.normalize("UMUNTU") == "UMUNTU"
        assert TextUtils.normalize("UMUNTU", preserve_case=False) == "umuntu"

    def test_strip_whitespace(self):
        assert TextUtils.normalize("  hello   world  ") == "hello world"

    def test_collapse_newlines(self):
        assert TextUtils.normalize("hello\n\n\nworld") == "hello world"

    def test_empty_string(self):
        assert TextUtils.normalize("") == ""

    def test_unicode_preserved(self):
        text = "Ngithanda ukudla okumnandi"
        assert TextUtils.normalize(text) == "Ngithanda ukudla okumnandi"


class TestTokenize:
    def test_simple_split(self):
        assert TextUtils.tokenize("umuntu omkhulu") == ["umuntu", "omkhulu"]

    def test_strip_punctuation(self):
        tokens = TextUtils.tokenize("hello, world! test.")
        assert tokens == ["hello", "world", "test"]

    def test_preserve_hyphens(self):
        tokens = TextUtils.tokenize("well-known word")
        assert "well-known" in tokens

    def test_empty_input(self):
        assert TextUtils.tokenize("") == []

    def test_punctuation_only(self):
        assert TextUtils.tokenize("... !!! ???") == []


class TestMetadataLine:
    def test_sadii_marker(self):
        assert TextUtils.is_metadata_line("<LINE# 0001>") is True

    def test_separator(self):
        assert TextUtils.is_metadata_line("----------") is True

    def test_equals_separator(self):
        assert TextUtils.is_metadata_line("==========") is True

    def test_normal_text(self):
        assert TextUtils.is_metadata_line("Umuntu ngumuntu") is False

    def test_empty_line(self):
        assert TextUtils.is_metadata_line("") is False


class TestNumeric:
    def test_number(self):
        assert TextUtils.is_numeric("1994") is True

    def test_word(self):
        assert TextUtils.is_numeric("hello") is False

    def test_mixed(self):
        assert TextUtils.is_numeric("abc123") is False


class TestExtractSadiiTokens:
    def test_standard_line(self):
        line = "Ukwengeza\tu[NPrePre15]-kw[BPre15]-engez[VRoot]-a[VerbTerm]\tengeza\tV"
        assert TextUtils.extract_sadii_tokens(line).lower() == "ukwengeza"

    def test_metadata_line(self):
        assert TextUtils.extract_sadii_tokens("<LINE# 0001>") is None

    def test_empty_line(self):
        assert TextUtils.extract_sadii_tokens("") is None

    def test_punctuation_only(self):
        line = ".\t.[Punc]\t.\tPUNC"
        assert TextUtils.extract_sadii_tokens(line) is None

    def test_number_token(self):
        line = "1994\t1994[Num]\t1994\tNUM"
        assert TextUtils.extract_sadii_tokens(line) is None
