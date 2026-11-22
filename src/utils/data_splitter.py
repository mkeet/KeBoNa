"""
data_splitter.py - splits a cleaned corpus into training, test and validation sets.

The cleaned corpus holds one word token per line, so the split is over word tokens. The
lines are shuffled with a fixed seed and then cut by ratio.
"""

import logging
import random
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class DataSplitter:
    """Splits a cleaned corpus file into train, test and validation files."""

    def __init__(
        self,
        train_ratio: float = 0.80,
        test_ratio: float = 0.10,
        val_ratio: float = 0.10,
        random_seed: int = 42,
    ):
        """
        Sets the ratios of the split, which must sum to 1.

        Args:
            train_ratio: Fraction of the lines used for training.
            test_ratio: Fraction of the lines used for testing.
            val_ratio: Fraction of the lines used for validation.
            random_seed: Seed of the shuffle, so that a corpus always gives the same split.
        """
        assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-9, (
            f"Ratios must sum to 1.0, got {train_ratio + test_ratio + val_ratio}"
        )
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed

    def split_file(
        self, input_file: Path, output_dir: Path
    ) -> Tuple[Path, Path, Path]:
        """
        Shuffles the lines of a cleaned corpus file and writes the three splits.

        Args:
            input_file: Path to cleaned.txt, which holds one word per line.
            output_dir: Folder that receives train.txt, test.txt and validation.txt.

        Returns:
            The paths of the train, test and validation files, in that order.
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        with open(input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        logger.info(f"Read {len(lines)} lines from {input_file}")

        if len(lines) < 10:
            raise ValueError(
                f"Corpus too small ({len(lines)} lines). "
                f"Need at least 10 for meaningful splits."
            )

        rng = random.Random(self.random_seed)
        rng.shuffle(lines)

        n = len(lines)
        train_end = int(n * self.train_ratio)
        test_end = train_end + int(n * self.test_ratio)

        train_data = lines[:train_end]
        test_data = lines[train_end:test_end]
        val_data = lines[test_end:]

        output_dir.mkdir(parents=True, exist_ok=True)
        # anything after "cleaned" in the input name is carried into the output names
        tag = input_file.stem.replace("cleaned", "")
        train_path = output_dir / f"train{tag}.txt"
        test_path = output_dir / f"test{tag}.txt"
        val_path = output_dir / f"validation{tag}.txt"

        self._write_split(train_path, train_data)
        self._write_split(test_path, test_data)
        self._write_split(val_path, val_data)

        logger.info(
            f"Split complete: {len(train_data)} train, "
            f"{len(test_data)} test, {len(val_data)} validation"
        )

        return train_path, test_path, val_path

    @staticmethod
    def _write_split(filepath: Path, lines: List[str]):
        """Writes a list of lines to a file."""
        with open(filepath, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        logger.debug(f"Wrote {len(lines)} lines to {filepath}")
