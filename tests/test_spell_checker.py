"""
test_spell_checker.py - tests for SpellChecker.

Covers the loading of models, detection, the fields of the result and the suggestions. The
fixture trains and saves a detection model and a word model on a small corpus.
"""
import pytest
from src.models.trigram_model import TrigramModel
from src.core.spell_checker import SpellChecker

_CORPUS = (
    "umuntu ngumuntu ngabantu\n"
    "inkosi yezwe enkulu\n"
    "abantu bahamba endlini\n"
    "umfana omncane udlala\n"
    "intombazane ihamba esikoleni\n"
    "izinkomo zidla utshani\n"
    "umuntu ulima insimu\n"
    "abantwana badlala phandle\n"
    "uthisha ufundisa esikoleni\n"
    "ubaba uhamba emsebenzini\n"
)


@pytest.fixture
def model_dir(tmp_path):
    """A model directory with a detection model (zulu_trigram) and a word model (zulu_word)."""
    models = tmp_path / "models"; models.mkdir()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_CORPUS, encoding="utf-8")
    det = TrigramModel(n=3)
    det.train(corpus, language="zulu")
    det.save(models / "zulu_trigram.pkl")
    word = TrigramModel(n=3)
    word.train(corpus, language="zulu")
    word.save(models / "zulu_word.pkl")
    return models


@pytest.fixture
def checker(model_dir):
    sc = SpellChecker()
    sc.load_models(model_dir)
    return sc


class TestModelLoading:
    def test_loads_models(self, checker):
        assert "zulu" in checker.get_available_languages()

    def test_no_models_in_empty_dir(self, tmp_path):
        sc = SpellChecker()
        empty = tmp_path / "empty"; empty.mkdir()
        sc.load_models(empty)
        assert len(sc.get_available_languages()) == 0


class TestTrigramCheck:
    def test_returns_error_list(self, checker):
        result = checker.check("umuntu ngumuntu ngabantu", language="zulu")
        assert isinstance(result["errors"], list)

    def test_misspelled_word_flagged(self, checker):
        result = checker.check("umutnu ngumuntu ngabantu", language="zulu")
        assert "umutnu" in [e["word"] for e in result["errors"]]

    def test_result_has_required_fields(self, checker):
        result = checker.check("umuntu", language="zulu")
        for key in ("errors", "input_text", "language"):
            assert key in result

    def test_empty_text(self, checker):
        result = checker.check("", language="zulu")
        assert len(result["errors"]) == 0


class TestSuggestions:
    def test_errors_include_suggestions(self, checker):
        result = checker.check("umutnu ngumuntu ngabantu", language="zulu")
        if result["errors"]:
            assert "suggestions" in result["errors"][0]


class TestInvalidInput:
    def test_invalid_language(self, checker):
        with pytest.raises((KeyError, ValueError)):
            checker.check("hello", language="klingon")
