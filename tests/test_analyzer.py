"""Tests for main analyzer API."""

import pytest
from ko_liwc import InterviewAnalyzer, analyze, Transcript
from ko_liwc.analyzer import AnalysisResult


class TestInterviewAnalyzer:
    """Test InterviewAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        return InterviewAnalyzer()

    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer.tokenizer is not None
        assert analyzer.scorer is not None
        assert len(analyzer._extractors) > 0

    def test_tokenize(self, analyzer):
        """Test tokenization method."""
        tokens = analyzer.tokenize("안녕하세요")
        assert len(tokens) > 0

    def test_extract_features(self, analyzer):
        """Test feature extraction."""
        text = "저는 회사에서 열심히 일하겠습니다."
        features = analyzer.extract_features(text, duration=10.0)

        # Check all expected features are present
        assert "wpsec" in features
        assert "pos_emotion_ratio" in features
        assert "work_ratio" in features
        assert "i_ratio" in features

    def test_analyze(self, analyzer):
        """Test full analysis pipeline."""
        text = "저는 이 직무에 적합합니다. 열심히 하겠습니다."
        result = analyzer.analyze(text, duration=15.0)

        assert isinstance(result, AnalysisResult)
        assert result.scores is not None
        assert result.features is not None
        assert 0 <= result.scores.overall <= 100

    def test_analyze_interview_text(self, analyzer):
        """Test with realistic interview text."""
        text = """
        안녕하세요. 저는 김철수입니다.
        저는 이 직무에 지원하게 된 이유가 있습니다.
        우리 회사의 비전과 제 목표가 일치하기 때문입니다.
        열심히 일하고 성과를 내겠습니다.
        감사합니다.
        """
        result = analyzer.analyze(text, duration=60.0)

        assert result.scores.overall > 0
        assert result.features["wc"] > 0
        assert result.features["work_ratio"] > 0

    def test_analyze_with_breakdown(self, analyzer):
        """Test analysis with detailed breakdown."""
        text = "저는 열정적으로 일하겠습니다."
        result = analyzer.analyze_with_breakdown(text, duration=10.0)

        assert "scores" in result
        assert "contributions" in result
        assert "top_positive" in result

    def test_analyze_transcript(self, analyzer):
        """Test analyzing Transcript object."""
        transcript = Transcript()
        transcript.add_segment("안녕하세요.", 0.0, 2.0)
        transcript.add_segment("열심히 하겠습니다.", 2.0, 5.0)

        result = analyzer.analyze_transcript(transcript)

        assert isinstance(result, AnalysisResult)
        assert result.duration == 5.0

    def test_analyze_segments(self, analyzer):
        """Test analyzing segment dictionaries."""
        segments = [
            {"text": "안녕하세요.", "start": 0.0, "end": 2.0},
            {"text": "감사합니다.", "start": 2.0, "end": 4.0},
        ]

        result = analyzer.analyze_segments(segments)

        assert isinstance(result, AnalysisResult)


class TestAnalysisResult:
    """Test AnalysisResult class."""

    @pytest.fixture
    def result(self):
        analyzer = InterviewAnalyzer()
        return analyzer.analyze("테스트 문장입니다.", duration=5.0)

    def test_to_dict(self, result):
        """Test conversion to dictionary."""
        d = result.to_dict()

        assert "text" in d
        assert "duration" in d
        assert "features" in d
        assert "scores" in d

    def test_summary(self, result):
        """Test summary generation."""
        summary = result.summary()

        assert "Interview Analysis Summary" in summary
        assert "Overall" in summary
        assert "Duration" in summary

    def test_repr(self, result):
        """Test string representation."""
        repr_str = repr(result)
        assert "AnalysisResult" in repr_str

    def test_feature_vector(self, result):
        """Test getting FeatureVector."""
        fv = result.feature_vector
        assert fv is not None


class TestConvenienceFunction:
    """Test analyze() convenience function."""

    def test_basic_analysis(self):
        """Test basic convenience function usage."""
        result = analyze("테스트입니다.", duration=5.0)
        assert isinstance(result, AnalysisResult)

    def test_detailed_analysis(self):
        """Test detailed analysis mode."""
        result = analyze("테스트입니다.", duration=5.0, detailed=True)
        assert isinstance(result, dict)
        assert "scores" in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def analyzer(self):
        return InterviewAnalyzer()

    def test_empty_text(self, analyzer):
        """Test handling empty text."""
        result = analyzer.analyze("", duration=10.0)
        assert result.features["wc"] == 0

    def test_very_short_text(self, analyzer):
        """Test very short text."""
        result = analyzer.analyze("네", duration=1.0)
        assert result.scores is not None

    def test_long_text(self, analyzer):
        """Test longer text."""
        text = "저는 열심히 일하겠습니다. " * 50
        result = analyzer.analyze(text, duration=300.0)
        assert result.features["wc"] > 100

    def test_zero_duration(self, analyzer):
        """Test zero duration handling."""
        result = analyzer.analyze("테스트", duration=0.0)
        # Should not crash, rates should be 0
        assert result.features["wpsec"] == 0.0

    def test_special_characters(self, analyzer):
        """Test text with special characters."""
        text = "안녕하세요! 정말요? 네..."
        result = analyzer.analyze(text, duration=5.0)
        assert result.scores is not None

    def test_mixed_language(self, analyzer):
        """Test mixed Korean/English text."""
        text = "저는 project manager로 일했습니다."
        result = analyzer.analyze(text, duration=5.0)
        assert result.scores is not None
