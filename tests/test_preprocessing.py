"""Tests for text preprocessing."""

from app import preprocessing


class TestCleanSingleText:
    def test_removes_non_alpha_characters(self):
        result = preprocessing.clean_single_text("Breaking news July 4, 2025!")
        assert "4" not in result
        assert "," not in result
        assert "!" not in result

    def test_lowercases(self):
        result = preprocessing.clean_single_text("BREAKING NEWS")
        # "news" is a stopword removed; "breaking" is lowercased and stemmed.
        assert result == "break news"

    def test_removes_stopwords(self):
        result = preprocessing.clean_single_text("the a and of an article")
        # These are all stopwords apart from "article" which is stemmed.
        assert result == "articl"
        assert "the" not in result
        assert "and" not in result

    def test_keeps_not_stopword(self):
        result = preprocessing.clean_single_text("this is not true")
        assert "not" in result

    def test_stems_words(self):
        result = preprocessing.clean_single_text("runners running ran")
        # Porter stems these forms; "runners" -> "runner".
        assert "runner" in result

    def test_empty_input_returns_empty(self):
        assert preprocessing.clean_single_text("") == ""

    def test_whitespace_trims(self):
        result = preprocessing.clean_single_text("   only   words   here  ")
        # Stopwords are removed so only content words remain (stemmed).
        assert "word" in result

    def test_unicode_is_stripped(self):
        result = preprocessing.clean_single_text("café news")
        assert "é" not in result


def test_clean_corpus_handles_multiple_texts():
    result = preprocessing.clean_corpus(["First article", "SECOND ARTICLE"])
    assert len(result) == 2
    assert all(isinstance(item, str) for item in result)