"""
test_corpus_cleaner.py - tests for the corpus cleaner.

Covers the cleaning of plain text, the reading of tab-separated SADII files, the statistics
and the edge cases.
"""
import pytest
from src.utils.corpus_cleaner import CorpusCleaner


@pytest.fixture
def cleaner():
    return CorpusCleaner()


@pytest.fixture
def plain_text_dir(tmp_path):
    d = tmp_path / "plain"; d.mkdir()
    (d / "sample.txt").write_text(
        "Umfana uhamba esikoleni.\n"
        "\n"
        "123\n"
        "Intombazane iyafunda.\n"
        "<s>\n"
        "Izinkomo zidla utshani.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def sadii_dir(tmp_path):
    d = tmp_path / "sadii"; d.mkdir()
    (d / "corpus.txt").write_text(
        "# Document\tSample\n"
        "Umfana\tumfana\tN\n"
        "uhamba\t-hamba\tV\n"
        "\n"
        "Intombazane\tintombazane\tN\n"
        "iyafunda\t-funda\tV\n",
        encoding="utf-8",
    )
    return d


class TestPlainTextCleaning:
    def test_basic_cleaning(self, cleaner, plain_text_dir, tmp_path):
        out = tmp_path / "cleaned.txt"
        cleaner.clean_directory(plain_text_dir, out)
        result = out.read_text(encoding="utf-8").lower()
        assert "umfana" in result
        assert "intombazane" in result
        assert "izinkomo" in result

    def test_metadata_and_numbers_filtered(self, cleaner, plain_text_dir, tmp_path):
        out = tmp_path / "cleaned.txt"
        cleaner.clean_directory(plain_text_dir, out)
        words = out.read_text(encoding="utf-8").split()
        assert "<s>" not in words
        assert "123" not in words


class TestSADIICleaning:
    def test_sadii_extraction(self, cleaner, sadii_dir, tmp_path):
        out = tmp_path / "cleaned.txt"
        cleaner.clean_directory(sadii_dir, out)
        result = out.read_text(encoding="utf-8").lower()
        assert "umfana" in result or "intombazane" in result

    def test_sadii_extracts_word_forms_only(self, cleaner, sadii_dir, tmp_path):
        # only the word forms of the first column are kept, not the morpheme analysis or the POS tags
        out = tmp_path / "cleaned.txt"
        cleaner.clean_directory(sadii_dir, out)
        words = out.read_text(encoding="utf-8").split()
        assert "-hamba" not in words
        assert "N" not in words and "V" not in words


class TestStats:
    def test_stats_populated(self, cleaner, plain_text_dir, tmp_path):
        out = tmp_path / "cleaned.txt"
        cleaner.clean_directory(plain_text_dir, out)
        assert cleaner.stats["files_processed"] > 0


class TestEdgeCases:
    def test_empty_directory_raises(self, cleaner, tmp_path):
        d = tmp_path / "empty"; d.mkdir()
        with pytest.raises(FileNotFoundError):
            cleaner.clean_directory(d, tmp_path / "cleaned.txt")

    def test_nonexistent_directory_raises(self, cleaner, tmp_path):
        with pytest.raises(FileNotFoundError):
            cleaner.clean_directory(tmp_path / "nope", tmp_path / "cleaned.txt")
