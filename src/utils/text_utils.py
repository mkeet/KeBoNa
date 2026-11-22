"""
text_utils.py - the string operations used by the corpus cleaner and the models.

Nothing here is specific to a language. corpus_cleaner.py reads the files and decides which
tokens to keep, and this module holds the string-level steps it calls.
"""

import re
import unicodedata
from typing import List, Optional


class TextUtils:
    """Static methods for normalising and tokenising text."""

    # metadata / markup lines, e.g. <LINE# 0001>, a bare page number, a row of dashes
    _METADATA_PATTERN = re.compile(r"^<.*>$|^\d+\.\s*$|^[-=_]{3,}$")

    # tokens with no letter in them
    _NUMERIC_PATTERN = re.compile(r"^[\d\W]+$")

    @staticmethod
    def normalize(text: str, preserve_case: bool = True) -> str:
        """
        Normalises a string to Unicode NFC and collapses every run of whitespace into one space.
        Case is preserved unless preserve_case is False, because NCB names carry capitals inside
        the word, as in kwaZulu.
        """
        text = unicodedata.normalize("NFC", text)
        if not preserve_case:
            text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Splits a normalised string on whitespace and strips the punctuation from both ends of
        each token. Punctuation inside a token is kept, and so is case.
        """
        tokens = []
        for raw_token in text.split():
            cleaned = raw_token.strip(".,;:!?\"'()[]{}...--/\\<>@#$%^&*+=|~`")
            if cleaned:
                tokens.append(cleaned)
        return tokens

    @staticmethod
    def is_metadata_line(line: str) -> bool:
        """
        Returns True for a line that is markup and not text. These are the paragraph markers of
        the SADII corpora such as <LINE# 0001>, a bare page number, and a row of dashes, equals
        signs or underscores.
        """
        return bool(TextUtils._METADATA_PATTERN.match(line.strip()))

    @staticmethod
    def is_numeric(token: str) -> bool:
        """Returns True for a token made up only of digits and punctuation, such as 1994 or 42."""
        return bool(TextUtils._NUMERIC_PATTERN.match(token))

    @staticmethod
    def extract_sadii_tokens(line: str) -> Optional[str]:
        """
        Returns the word form from one line of a tab-separated SADII corpus file, which is its
        first field. Returns None for a blank line, a metadata line, a first field that holds
        spaces and is therefore a whole sentence, or a first field with no letter in it. Case
        is preserved.
        """
        line = line.strip()
        if not line or TextUtils.is_metadata_line(line):
            return None

        parts = line.split("\t")
        token = parts[0].strip()

        if ' ' in token:
            return None

        if token and not TextUtils.is_numeric(token) and any(c.isalpha() for c in token):
            return token
        return None
