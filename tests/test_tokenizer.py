"""Tests for Korean tokenizer."""

import pytest
from ko_liwc.core.tokenizer import KiwiTokenizer, Token


class TestKiwiTokenizer:
    """Test KiwiTokenizer functionality."""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer instance."""
        return KiwiTokenizer()

    def test_basic_tokenization(self, tokenizer):
        """Test basic Korean text tokenization."""
        text = "안녕하세요"
        tokens = tokenizer.tokenize(text)

        assert len(tokens) > 0
        assert all(isinstance(t, Token) for t in tokens)

    def test_sentence_tokenization(self, tokenizer):
        """Test sentence with multiple morphemes."""
        text = "저는 회사에서 일합니다."
        tokens = tokenizer.tokenize(text)

        # Should have multiple tokens
        assert len(tokens) >= 4

        # Check token attributes
        for token in tokens:
            assert hasattr(token, "form")
            assert hasattr(token, "tag")

    def test_verb_extraction(self, tokenizer):
        """Test verb morpheme extraction."""
        text = "먹고 자고 뛴다."
        tokens = tokenizer.tokenize(text)

        # Should find verb stems (VV/VA/VX)
        # Note: XSV (verb-forming suffix) also indicates verbal form
        verb_tags = {"VV", "VA", "VX", "VCP", "VCN", "XSV"}
        verb_tokens = [t for t in tokens if t.tag in verb_tags]
        assert len(verb_tokens) >= 1

    def test_pronoun_extraction(self, tokenizer):
        """Test pronoun extraction."""
        text = "나는 우리 회사를 좋아합니다."
        tokens = tokenizer.tokenize(text)

        # Check for pronoun tags
        forms = [t.form for t in tokens]
        assert "나" in forms or any("나" in f for f in forms)

    def test_empty_text(self, tokenizer):
        """Test empty text handling."""
        tokens = tokenizer.tokenize("")
        assert tokens == []

    def test_whitespace_only(self, tokenizer):
        """Test whitespace-only text."""
        tokens = tokenizer.tokenize("   ")
        # May return empty or whitespace tokens
        assert isinstance(tokens, list)

    def test_token_positions(self, tokenizer):
        """Test that token positions are valid."""
        text = "안녕하세요."
        tokens = tokenizer.tokenize(text)

        for token in tokens:
            assert token.start >= 0
            assert token.end >= token.start
            assert token.end <= len(text)

    def test_multiple_sentences(self, tokenizer):
        """Test text with multiple sentences."""
        text = "저는 학생입니다. 열심히 공부합니다."
        tokens = tokenizer.tokenize(text)

        # Should have tokens from both sentences
        assert len(tokens) >= 6

    def test_filler_words(self, tokenizer):
        """Test recognition of filler words."""
        text = "음 그러니까 좀 그렇습니다."
        tokens = tokenizer.tokenize(text)

        forms = [t.form for t in tokens]
        # At least some filler-like words should be present
        assert len(forms) > 0
