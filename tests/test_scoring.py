"""Tests for scoring module."""

import pytest
from ko_liwc.scoring.normalizer import MinMaxNormalizer, DEFAULT_FEATURE_RANGES
from ko_liwc.scoring.scorer import InterviewScorer
from ko_liwc.scoring.weights import (
    FEATURE_WEIGHTS,
    TRAIT_NAMES,
    get_trait_weights,
    get_top_features,
)


class TestMinMaxNormalizer:
    """Test MinMaxNormalizer."""

    def test_preset_ranges(self):
        """Test normalizer with preset ranges."""
        normalizer = MinMaxNormalizer(
            preset_ranges={"feature1": (0.0, 10.0)}
        )

        assert normalizer.is_fitted

        result = normalizer.transform({"feature1": 5.0})
        assert result["feature1"] == 0.5

    def test_fit_transform(self):
        """Test fit and transform."""
        normalizer = MinMaxNormalizer()

        features_list = [
            {"a": 0.0, "b": 10.0},
            {"a": 10.0, "b": 20.0},
        ]

        normalizer.fit(features_list)
        assert normalizer.is_fitted

        result = normalizer.transform({"a": 5.0, "b": 15.0})
        assert result["a"] == 0.5
        assert result["b"] == 0.5

    def test_clipping(self):
        """Test that values are clipped to [0, 1]."""
        normalizer = MinMaxNormalizer(
            preset_ranges={"x": (0.0, 10.0)}
        )

        result = normalizer.transform({"x": 15.0})
        assert result["x"] == 1.0

        result = normalizer.transform({"x": -5.0})
        assert result["x"] == 0.0

    def test_zero_range(self):
        """Test handling of zero range (all same values)."""
        normalizer = MinMaxNormalizer()
        normalizer.fit([{"x": 5.0}, {"x": 5.0}])

        result = normalizer.transform({"x": 5.0})
        assert result["x"] == 0.5

    def test_default_ranges(self):
        """Test default feature ranges are defined."""
        assert "wpsec" in DEFAULT_FEATURE_RANGES
        assert "pos_emotion_ratio" in DEFAULT_FEATURE_RANGES
        assert len(DEFAULT_FEATURE_RANGES) >= 20


class TestInterviewScorer:
    """Test InterviewScorer."""

    @pytest.fixture
    def scorer(self):
        return InterviewScorer()

    @pytest.fixture
    def sample_features(self):
        """Sample normalized features for testing."""
        return {
            "wpsec": 0.5,
            "upsec": 0.5,
            "fpsec": 0.2,
            "wc": 0.5,
            "uc": 0.5,
            "i_ratio": 0.3,
            "we_ratio": 0.4,
            "they_ratio": 0.1,
            "pos_emotion_ratio": 0.6,
            "neg_emotion_ratio": 0.1,
            "anxiety_ratio": 0.1,
            "anger_ratio": 0.05,
            "sadness_ratio": 0.05,
            "cognitive_ratio": 0.4,
            "work_ratio": 0.5,
            "nonfluency_ratio": 0.2,
            "negation_ratio": 0.1,
        }

    def test_basic_scoring(self, scorer, sample_features):
        """Test basic score computation."""
        score = scorer.score(sample_features, normalize=False)

        assert hasattr(score, "overall")
        assert hasattr(score, "recommend_hiring")
        assert hasattr(score, "excited")
        assert hasattr(score, "engagement")
        assert hasattr(score, "friendliness")

    def test_score_range(self, scorer, sample_features):
        """Test that scores are in valid range."""
        score = scorer.score(sample_features, normalize=False)

        assert 0 <= score.overall <= 100
        assert 0 <= score.recommend_hiring <= 100
        assert 0 <= score.excited <= 100
        assert 0 <= score.engagement <= 100
        assert 0 <= score.friendliness <= 100

    def test_positive_features_increase_score(self, scorer):
        """Test that positive features increase scores."""
        # Low positive emotion
        low_features = {"pos_emotion_ratio": 0.1}
        # High positive emotion
        high_features = {"pos_emotion_ratio": 0.9}

        low_score = scorer.score(low_features, normalize=False)
        high_score = scorer.score(high_features, normalize=False)

        # Higher positive emotion should increase excited score
        assert high_score.excited > low_score.excited

    def test_negative_features_decrease_score(self, scorer):
        """Test that negative features decrease scores."""
        # Low filler ratio
        low_features = {"fpsec": 0.1, "nonfluency_ratio": 0.1}
        # High filler ratio
        high_features = {"fpsec": 0.9, "nonfluency_ratio": 0.9}

        low_score = scorer.score(low_features, normalize=False)
        high_score = scorer.score(high_features, normalize=False)

        # Higher fillers should decrease overall score
        assert high_score.overall < low_score.overall

    def test_score_with_breakdown(self, scorer, sample_features):
        """Test scoring with detailed breakdown."""
        result = scorer.score_with_breakdown(sample_features, normalize=False)

        assert "scores" in result
        assert "contributions" in result
        assert "top_positive" in result
        assert "top_negative" in result

    def test_batch_scoring(self, scorer, sample_features):
        """Test batch scoring multiple feature sets."""
        features_list = [sample_features, sample_features]
        scores = scorer.batch_score(features_list, normalize=False)

        assert len(scores) == 2


class TestWeights:
    """Test weight configuration."""

    def test_trait_names(self):
        """Test that all trait names are defined."""
        assert "overall" in TRAIT_NAMES
        assert "recommend_hiring" in TRAIT_NAMES
        assert len(TRAIT_NAMES) == 5

    def test_feature_weights_structure(self):
        """Test feature weights structure."""
        for trait in TRAIT_NAMES:
            assert trait in FEATURE_WEIGHTS
            weights = FEATURE_WEIGHTS[trait]
            assert isinstance(weights, dict)
            assert len(weights) > 0

    def test_get_trait_weights(self):
        """Test getting weights for specific trait."""
        weights = get_trait_weights("overall")
        assert "pos_emotion_ratio" in weights
        assert isinstance(weights["pos_emotion_ratio"], float)

    def test_invalid_trait(self):
        """Test error on invalid trait."""
        with pytest.raises(ValueError):
            get_trait_weights("invalid_trait")

    def test_get_top_features(self):
        """Test getting top important features."""
        top = get_top_features(5)
        assert len(top) == 5
        assert all(isinstance(f, str) for f in top)
