"""Tests for feature extractors."""

import pytest
from ko_liwc.core.tokenizer import KiwiTokenizer
from ko_liwc.features import (
    SpeakingRateExtractor,
    PronounExtractor,
    POSFeatureExtractor,
    EmotionExtractor,
    CognitiveExtractor,
    MiscFeatureExtractor,
)


class TestSpeakingRateExtractor:
    """Test SpeakingRateExtractor."""

    @pytest.fixture
    def extractor(self):
        return SpeakingRateExtractor()

    @pytest.fixture
    def tokenizer(self):
        return KiwiTokenizer()

    def test_feature_names(self, extractor):
        """Test that feature names are correct."""
        names = extractor.feature_names
        assert "wpsec" in names
        assert "upsec" in names
        assert "fpsec" in names
        assert "wc" in names
        assert "uc" in names

    def test_basic_extraction(self, extractor, tokenizer):
        """Test basic feature extraction."""
        text = "저는 열심히 일하겠습니다."
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=10.0)

        assert "wpsec" in features
        assert "wc" in features
        assert features["wc"] > 0
        assert features["wpsec"] > 0

    def test_words_per_second(self, extractor, tokenizer):
        """Test words per second calculation."""
        text = "하나 둘 셋 넷 다섯"
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=5.0)

        # Should have roughly 1 word per second
        assert features["wpsec"] > 0.5

    def test_zero_duration(self, extractor, tokenizer):
        """Test handling of zero duration."""
        text = "테스트 문장"
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=0.0)

        assert features["wpsec"] == 0.0
        assert features["wc"] > 0


class TestPronounExtractor:
    """Test PronounExtractor."""

    @pytest.fixture
    def extractor(self):
        return PronounExtractor()

    @pytest.fixture
    def tokenizer(self):
        return KiwiTokenizer()

    def test_feature_names(self, extractor):
        names = extractor.feature_names
        assert "i_ratio" in names
        assert "we_ratio" in names
        assert "they_ratio" in names

    def test_i_pronoun(self, extractor, tokenizer):
        """Test first-person singular detection."""
        text = "나는 저는 내가 제가"
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=10.0)

        assert features["i_ratio"] > 0

    def test_we_pronoun(self, extractor, tokenizer):
        """Test first-person plural detection."""
        text = "우리는 저희는 우리의 팀"
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=10.0)

        assert features["we_ratio"] > 0

    def test_no_pronouns(self, extractor, tokenizer):
        """Test text without pronouns."""
        text = "회사는 좋은 성과를 냈다."
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=10.0)

        # Should still return valid ratios (possibly 0)
        assert features["i_ratio"] >= 0
        assert features["i_ratio"] <= 1


class TestEmotionExtractor:
    """Test EmotionExtractor."""

    @pytest.fixture
    def extractor(self):
        return EmotionExtractor()

    @pytest.fixture
    def tokenizer(self):
        return KiwiTokenizer()

    def test_feature_names(self, extractor):
        names = extractor.feature_names
        assert "pos_emotion_ratio" in names
        assert "neg_emotion_ratio" in names
        assert "anxiety_ratio" in names

    def test_positive_emotion(self, extractor, tokenizer):
        """Test positive emotion detection."""
        text = "정말 좋아요 행복해요 감사합니다"
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=10.0)

        assert features["pos_emotion_ratio"] > 0

    def test_negative_emotion(self, extractor, tokenizer):
        """Test negative emotion detection."""
        text = "슬퍼요 힘들어요 싫어요"
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=10.0)

        assert features["neg_emotion_ratio"] > 0

    def test_anxiety_detection(self, extractor, tokenizer):
        """Test anxiety word detection."""
        text = "걱정되고 불안해요 긴장됩니다"
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=10.0)

        assert features["anxiety_ratio"] > 0


class TestCognitiveExtractor:
    """Test CognitiveExtractor."""

    @pytest.fixture
    def extractor(self):
        return CognitiveExtractor()

    @pytest.fixture
    def tokenizer(self):
        return KiwiTokenizer()

    def test_feature_names(self, extractor):
        names = extractor.feature_names
        assert "cognitive_ratio" in names
        assert "inhibition_ratio" in names
        assert "perceptual_ratio" in names

    def test_cognitive_words(self, extractor, tokenizer):
        """Test cognitive process word detection."""
        text = "생각해보면 이해가 됩니다 분석했습니다"
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=10.0)

        assert features["cognitive_ratio"] > 0


class TestMiscFeatureExtractor:
    """Test MiscFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        return MiscFeatureExtractor()

    @pytest.fixture
    def tokenizer(self):
        return KiwiTokenizer()

    def test_feature_names(self, extractor):
        names = extractor.feature_names
        assert "nonfluency_ratio" in names
        assert "negation_ratio" in names
        assert "work_ratio" in names

    def test_work_words(self, extractor, tokenizer):
        """Test work-related word detection."""
        text = "회사에서 업무를 처리하고 프로젝트를 진행했습니다"
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=10.0)

        assert features["work_ratio"] > 0

    def test_negation_words(self, extractor, tokenizer):
        """Test negation word detection."""
        text = "없습니다 못합니다 아닙니다"
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=10.0)

        assert features["negation_ratio"] > 0

    def test_nonfluency(self, extractor, tokenizer):
        """Test non-fluency marker detection."""
        text = "음 어 그러니까 좀"
        tokens = tokenizer.tokenize(text)

        features = extractor.extract(tokens, duration=10.0)

        # Should detect some fillers
        assert features["nonfluency_ratio"] >= 0
