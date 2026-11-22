"""
spell_checker.py - loads the models of every language and checks text against them.

Detection scores each word with the character-trigram model and applies the learned threshold
of the language. The word model is used only to suggest corrections for the words flagged.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.models.threshold_classifier import ThresholdClassifier
from src.models.trigram_model import TrigramModel
from src.utils.text_utils import TextUtils

logger = logging.getLogger(__name__)

# used only for a language that has no classifier, when no threshold is passed to check()
DEFAULT_THRESHOLD = -6.0


class SpellChecker:
    """Holds the detection model, the word model and the classifier of each language."""

    def __init__(self, max_suggestions: int = 5):
        self.suggestion_models: Dict[str, TrigramModel] = {}
        self.detection_models: Dict[str, TrigramModel] = {}
        self.threshold_classifiers: Dict[str, ThresholdClassifier] = {}
        self.max_suggestions = max_suggestions

    def load_models(self, models_dir: Path) -> None:
        """
        Loads every model found in a directory. The language of a model is read from its file
        name, <language>_word.pkl, <language>_trigram.pkl or <language>_classifier.pkl, so a
        newly trained language needs no registration.
        """
        models_dir = Path(models_dir)
        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return

        for pkl_file in sorted(models_dir.glob("*_word.pkl")):
            lang = pkl_file.stem.replace("_word", "")
            try:
                self.suggestion_models[lang] = TrigramModel.load(pkl_file)
                logger.info(f"Loaded word model: {lang}")
            except Exception as e:
                logger.error(f"Failed to load word model {pkl_file}: {e}")

        for pkl_file in sorted(models_dir.glob("*_trigram.pkl")):
            lang = pkl_file.stem.replace("_trigram", "")
            try:
                self.detection_models[lang] = TrigramModel.load(pkl_file)
                logger.info(f"Loaded detection model: {lang}")
            except Exception as e:
                logger.error(f"Failed to load detection model {pkl_file}: {e}")

        for pkl_file in sorted(models_dir.glob("*_classifier.pkl")):
            lang = pkl_file.stem.replace("_classifier", "")
            try:
                self.threshold_classifiers[lang] = ThresholdClassifier.load(pkl_file)
                logger.info(f"Loaded classifier: {lang}")
            except Exception as e:
                logger.error(f"Failed to load classifier {pkl_file}: {e}")

        logger.info(f"Models loaded - detection: {list(self.detection_models.keys())}, "
                    f"word: {list(self.suggestion_models.keys())}")

    def check(
        self,
        text: str,
        language: str,
        threshold: Optional[float] = None,
        include_suggestions: bool = True,
        min_word_length: int = 3,
    ) -> Dict:
        """
        Checks a text with the models of one language.

        Args:
            text: The text to check.
            language: A language for which models are loaded.
            threshold: A fixed threshold on the score. The learned threshold of the language
                is used when it is not given.
            include_suggestions: Whether to look up corrections for the words flagged.
            min_word_length: Words shorter than this are skipped, because a word of fewer than
                three characters yields no trigram to score. With 1 they are scored anyway and
                are flagged.

        Returns:
            A dict with the input text, the language, the number of words and the list of
            errors. Each error holds the word, its position, its score and its suggestions.
        """
        if language not in self.detection_models:
            available = self.get_available_languages()
            raise ValueError(f"No detection model for '{language}'. Available: {available}")

        errors = self._check_words(text, language, threshold, include_suggestions, min_word_length)

        tokens = TextUtils.tokenize(TextUtils.normalize(text))

        return {
            "input_text": text,
            "language": language,
            "errors": errors,
            "total_words": len(tokens),
            "error_count": len(errors),
        }

    def _check_words(self, text: str, language: str, threshold, include_suggestions: bool,
                     min_word_length: int = 3) -> List[Dict]:
        detection_model = self.detection_models[language]
        word_model = self.suggestion_models.get(language)

        clf = self.threshold_classifiers.get(language)
        use_classifier = clf is not None and threshold is None

        words = TextUtils.tokenize(TextUtils.normalize(text))
        if not words:
            return []

        errors = []

        for word_idx, word in enumerate(words):

            if len(word) < min_word_length:
                continue

            score = detection_model.score_word(word)

            if use_classifier:
                flag = clf.predict_error(score)
            else:
                flag = score < (threshold if threshold is not None else DEFAULT_THRESHOLD)

            if flag:
                suggestions = []
                if include_suggestions and word_model:
                    suggestions = word_model.suggest_corrections(
                        word, max_suggestions=self.max_suggestions,
                        tie_breaker_model=detection_model
                    )
                errors.append({
                    "word":        word,
                    "position":    word_idx,
                    "probability": score,
                    "suggestions": suggestions,
                })

        return errors

    def get_available_languages(self) -> List[str]:
        """Returns the languages for which a detection model or a word model is loaded."""
        langs = set(self.suggestion_models.keys())
        langs.update(self.detection_models.keys())
        return sorted(langs)
