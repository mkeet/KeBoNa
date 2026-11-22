"""
threshold_classifier.py - the per-language threshold on the detection score.

A one-dimensional logistic regression is fitted on the scores of correct words and of the same
words after corruption. Its decision boundary, tau, replaces a threshold fixed by hand.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.evaluation.spelling_wrecker import SpellingWrecker
from src.utils.text_utils import TextUtils

logger = logging.getLogger(__name__)

# scores at or below this come from the unseen-trigram penalty and are left out of the fit
_SCORE_FLOOR = -19.0
_MIN_WORD_LEN = 3


class ThresholdClassifier:
    """Flags a word as an error when its detection score falls at or below a learned boundary."""

    def __init__(self):
        self._clf = None
        self.language: Optional[str] = None
        self.fallback_threshold: float = -10.0
        self._trained = False

    def train(
        self,
        trigram_model,
        validation_file: Path,
        language: str,
        corruption_rate: float = 1.0,
        seed: int = 42,
    ) -> None:
        """
        Fits the regression on the validation split.

        Every word of at least three characters is scored and labelled correct. The same words
        are then corrupted, and each corrupted form is scored and labelled incorrect.

        Args:
            trigram_model: The detection model that supplies the scores.
            validation_file: Path to validation.txt.
            language: Name of the language, stored with the classifier.
            corruption_rate: Fraction of the words to corrupt. At 1.0 every word receives one
                error, which keeps the two classes close to balanced.
            seed: Seed of the corruption and of the regression.
        """
        if not validation_file.exists():
            logger.error("[%s] Validation file not found: %s", language, validation_file)
            return

        logger.info("[%s] Training threshold classifier from %s...", language, validation_file.name)

        with open(validation_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            logger.error("[%s] Validation file is empty.", language)
            return

        positive_scores = []
        negative_scores = []

        for line in lines:
            tokens = TextUtils.tokenize(TextUtils.normalize(line))
            for word in tokens:
                if len(word) < _MIN_WORD_LEN:
                    continue
                score = trigram_model.score_word(word)
                if score > _SCORE_FLOOR:
                    positive_scores.append(score)

        wrecker = SpellingWrecker(corruption_rate=corruption_rate, seed=seed)
        corrupted_text, ground_truth = wrecker.corrupt_text(" ".join(lines))
        corrupted_set = set(ground_truth.keys())

        for word in TextUtils.tokenize(TextUtils.normalize(corrupted_text)):
            if word not in corrupted_set:
                continue
            if len(word) < _MIN_WORD_LEN:
                continue
            score = trigram_model.score_word(word)
            if score > _SCORE_FLOOR:
                negative_scores.append(score)

        if not positive_scores or not negative_scores:
            logger.error("[%s] Insufficient samples (pos=%d, neg=%d)",
                        language, len(positive_scores), len(negative_scores))
            return

        logger.info("[%s] Training samples - pos: %d, neg: %d",
                    language, len(positive_scores), len(negative_scores))

        X = np.array([[s] for s in positive_scores] + [[s] for s in negative_scores])
        y = np.array([1] * len(positive_scores) + [0] * len(negative_scores))

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf.fit(X, y)
        self._clf = clf
        self.language = language
        self._trained = True

        boundary = -clf.intercept_[0] / clf.coef_[0][0]
        logger.info("[%s] Learned boundary tau ~= %.3f", language, boundary)

    def predict_error(self, log_prob: float) -> bool:
        """
        Returns True when the score indicates an error, that is when coef*score + intercept
        is at or below zero. An untrained classifier falls back to a fixed threshold.
        """
        if not self._trained or self._clf is None:
            return log_prob < self.fallback_threshold
        coef = getattr(self, "_coef", None)
        if coef is None:
            self._coef = coef = float(self._clf.coef_[0][0])
            self._intercept = float(self._clf.intercept_[0])
        return (coef * log_prob + self._intercept) <= 0.0

    def save(self, path: Path) -> None:
        """Saves the fitted regression and its metadata to a pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "clf": self._clf,
                "language": self.language,
                "fallback_threshold": self.fallback_threshold,
                "trained": self._trained,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("[%s] Classifier saved -> %s", self.language, path.name)

    @classmethod
    def load(cls, path: Path) -> "ThresholdClassifier":
        """Loads a classifier saved by save()."""
        if not path.exists():
            raise FileNotFoundError(f"Classifier not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._clf = data["clf"]
        obj.language = data["language"]
        obj.fallback_threshold = data.get("fallback_threshold", -10.0)
        obj._trained = data.get("trained", False)
        logger.info("Loaded classifier for %s", obj.language)
        return obj

    @property
    def is_trained(self) -> bool:
        return self._trained
