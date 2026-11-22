"""
test_data_splitter.py - tests for the split into training, test and validation sets.

Covers the ratios of the split, the three output files, reproducibility with a seed and the
edge cases.
"""
import pytest
from src.utils.data_splitter import DataSplitter


@pytest.fixture
def splitter():
    return DataSplitter(train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, random_seed=42)


@pytest.fixture
def corpus_file(tmp_path):
    """A cleaned.txt with enough lines to split meaningfully (stem 'cleaned' -> no tag)."""
    f = tmp_path / "cleaned.txt"
    f.write_text("".join(f"sentence number {i} with some words\n" for i in range(100)),
                 encoding="utf-8")
    return f


def _lines(path):
    return len(path.read_text(encoding="utf-8").strip().split("\n"))


class TestRandomSplit:
    def test_creates_three_files(self, splitter, corpus_file, tmp_path):
        out = tmp_path / "out"; out.mkdir()
        splitter.split_file(corpus_file, out)
        assert (out / "train.txt").exists()
        assert (out / "test.txt").exists()
        assert (out / "validation.txt").exists()

    def test_no_data_loss(self, splitter, corpus_file, tmp_path):
        out = tmp_path / "out"; out.mkdir()
        splitter.split_file(corpus_file, out)
        total = sum(_lines(out / f"{n}.txt") for n in ("train", "test", "validation"))
        assert total == 100

    def test_approximate_ratios(self, splitter, corpus_file, tmp_path):
        out = tmp_path / "out"; out.mkdir()
        splitter.split_file(corpus_file, out)
        train, test, val = (_lines(out / f"{n}.txt") for n in ("train", "test", "validation"))
        assert 75 <= train <= 85
        assert 5 <= test <= 15
        assert 5 <= val <= 15

    def test_deterministic_with_seed(self, corpus_file, tmp_path):
        s1 = DataSplitter(random_seed=42)
        s2 = DataSplitter(random_seed=42)
        out1 = tmp_path / "out1"; out1.mkdir()
        out2 = tmp_path / "out2"; out2.mkdir()
        s1.split_file(corpus_file, out1)
        s2.split_file(corpus_file, out2)
        assert (out1 / "train.txt").read_text() == (out2 / "train.txt").read_text()


class TestEdgeCases:
    def test_too_small_raises(self, splitter, tmp_path):
        """split_file refuses a corpus of fewer than ten lines."""
        f = tmp_path / "tiny.txt"
        f.write_text("one\ntwo\nthree\n", encoding="utf-8")
        out = tmp_path / "out"; out.mkdir()
        with pytest.raises(ValueError):
            splitter.split_file(f, out)
