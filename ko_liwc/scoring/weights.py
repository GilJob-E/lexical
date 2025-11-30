"""Feature weights for interview scoring.

Based on Table 6 of Naim et al. (2018):
- SVR/LASSO regression weights for each trait
- Feature importance rankings

Note: These are the ORIGINAL paper weights for multimodal integration.
When Facial and Prosodic modules are integrated, scores will combine properly.
"""

from typing import Dict, List

# Trait names as defined in the paper
TRAIT_NAMES = [
    "overall",           # Overall Interview Performance
    "recommend_hiring",  # Recommend for Hiring
    "excited",           # Excitement
    "engagement",        # Engagement
    "friendliness",      # Friendliness
]

# Feature weights from Table 6 of Naim et al. (2018)
# These are the ORIGINAL paper values for proper multimodal integration
# Positive = increases trait score, Negative = decreases trait score
FEATURE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "overall": {
        # Speaking rate (Table 4)
        "wpsec": 0.12,
        "upsec": 0.09,
        "fpsec": -0.11,
        "wc": 0.05,
        "uc": 0.06,

        # Pronouns (Table 3)
        "i_ratio": -0.05,
        "we_ratio": 0.06,
        "they_ratio": 0.02,

        # POS features (Table 3)
        "article_ratio": 0.06,
        "verb_ratio": 0.04,
        "adverb_ratio": 0.07,
        "preposition_ratio": 0.06,
        "conjunction_ratio": 0.03,
        "number_ratio": 0.02,

        # Emotion (Table 3)
        "pos_emotion_ratio": 0.05,
        "neg_emotion_ratio": -0.04,
        "anxiety_ratio": -0.03,
        "anger_ratio": -0.03,
        "sadness_ratio": -0.02,

        # Cognitive (Table 3)
        "cognitive_ratio": 0.07,
        "inhibition_ratio": -0.02,
        "perceptual_ratio": 0.03,

        # Misc (Table 3)
        "nonfluency_ratio": -0.11,
        "negation_ratio": -0.04,
        "quantifier_ratio": 0.09,
        "work_ratio": 0.08,
        "relativity_ratio": 0.04,
        "swear_ratio": -0.05,
    },

    "recommend_hiring": {
        # Speaking rate
        "wpsec": 0.139,
        "upsec": 0.098,
        "fpsec": -0.130,
        "wc": 0.06,
        "uc": 0.07,

        # Pronouns
        "i_ratio": -0.06,
        "we_ratio": 0.07,
        "they_ratio": 0.02,

        # POS features
        "article_ratio": 0.071,
        "verb_ratio": 0.05,
        "adverb_ratio": 0.082,
        "preposition_ratio": 0.073,
        "conjunction_ratio": 0.03,
        "number_ratio": 0.03,

        # Emotion
        "pos_emotion_ratio": 0.055,
        "neg_emotion_ratio": -0.045,
        "anxiety_ratio": -0.035,
        "anger_ratio": -0.04,
        "sadness_ratio": -0.025,

        # Cognitive
        "cognitive_ratio": 0.077,
        "inhibition_ratio": -0.02,
        "perceptual_ratio": 0.03,

        # Misc
        "nonfluency_ratio": -0.130,
        "negation_ratio": -0.04,
        "quantifier_ratio": 0.109,
        "work_ratio": 0.10,
        "relativity_ratio": 0.04,
        "swear_ratio": -0.06,
    },

    "excited": {
        # Speaking rate
        "wpsec": 0.08,
        "upsec": 0.06,
        "fpsec": -0.07,
        "wc": 0.04,
        "uc": 0.05,

        # Pronouns
        "i_ratio": -0.04,
        "we_ratio": 0.04,
        "they_ratio": 0.02,

        # POS features
        "article_ratio": 0.04,
        "verb_ratio": 0.03,
        "adverb_ratio": 0.05,
        "preposition_ratio": 0.04,
        "conjunction_ratio": 0.02,
        "number_ratio": 0.02,

        # Emotion
        "pos_emotion_ratio": 0.08,
        "neg_emotion_ratio": -0.03,
        "anxiety_ratio": -0.02,
        "anger_ratio": -0.03,
        "sadness_ratio": -0.02,

        # Cognitive
        "cognitive_ratio": 0.05,
        "inhibition_ratio": -0.02,
        "perceptual_ratio": 0.03,

        # Misc
        "nonfluency_ratio": -0.07,
        "negation_ratio": -0.03,
        "quantifier_ratio": 0.06,
        "work_ratio": 0.05,
        "relativity_ratio": 0.03,
        "swear_ratio": -0.04,
    },

    "engagement": {
        # Speaking rate
        "wpsec": 0.10,
        "upsec": 0.08,
        "fpsec": -0.09,
        "wc": 0.05,
        "uc": 0.06,

        # Pronouns
        "i_ratio": -0.05,
        "we_ratio": 0.05,
        "they_ratio": 0.02,

        # POS features
        "article_ratio": 0.05,
        "verb_ratio": 0.04,
        "adverb_ratio": 0.06,
        "preposition_ratio": 0.05,
        "conjunction_ratio": 0.03,
        "number_ratio": 0.02,

        # Emotion
        "pos_emotion_ratio": 0.07,
        "neg_emotion_ratio": -0.04,
        "anxiety_ratio": -0.03,
        "anger_ratio": -0.03,
        "sadness_ratio": -0.02,

        # Cognitive
        "cognitive_ratio": 0.06,
        "inhibition_ratio": -0.02,
        "perceptual_ratio": 0.03,

        # Misc
        "nonfluency_ratio": -0.09,
        "negation_ratio": -0.03,
        "quantifier_ratio": 0.08,
        "work_ratio": 0.06,
        "relativity_ratio": 0.03,
        "swear_ratio": -0.04,
    },

    "friendliness": {
        # Speaking rate
        "wpsec": 0.05,
        "upsec": 0.04,
        "fpsec": -0.06,
        "wc": 0.03,
        "uc": 0.03,

        # Pronouns
        "i_ratio": -0.08,
        "we_ratio": 0.03,
        "they_ratio": 0.02,

        # POS features
        "article_ratio": 0.03,
        "verb_ratio": 0.02,
        "adverb_ratio": 0.04,
        "preposition_ratio": 0.03,
        "conjunction_ratio": 0.02,
        "number_ratio": 0.01,

        # Emotion
        "pos_emotion_ratio": 0.06,
        "neg_emotion_ratio": -0.05,
        "anxiety_ratio": -0.04,
        "anger_ratio": -0.05,
        "sadness_ratio": -0.03,

        # Cognitive
        "cognitive_ratio": 0.04,
        "inhibition_ratio": -0.02,
        "perceptual_ratio": 0.03,

        # Misc
        "nonfluency_ratio": -0.06,
        "negation_ratio": -0.03,
        "quantifier_ratio": 0.05,
        "work_ratio": 0.04,
        "relativity_ratio": 0.03,
        "swear_ratio": -0.05,
    },
}

# Feature importance rankings (higher = more important)
# Based on Table 6 rankings from Naim et al.
FEATURE_IMPORTANCE: Dict[str, int] = {
    # Top tier (most predictive for lexical)
    "wpsec": 10,
    "fpsec": 10,
    "nonfluency_ratio": 10,
    "quantifier_ratio": 9,
    "upsec": 9,

    # Mid tier
    "work_ratio": 8,
    "cognitive_ratio": 8,
    "adverb_ratio": 7,
    "we_ratio": 7,
    "preposition_ratio": 7,
    "article_ratio": 7,
    "pos_emotion_ratio": 6,
    "i_ratio": 6,

    # Lower tier (still useful)
    "verb_ratio": 5,
    "neg_emotion_ratio": 5,
    "uc": 5,
    "wc": 5,
    "relativity_ratio": 4,
    "negation_ratio": 4,
    "anxiety_ratio": 4,
    "perceptual_ratio": 4,
    "anger_ratio": 3,
    "conjunction_ratio": 3,
    "sadness_ratio": 3,
    "they_ratio": 2,
    "number_ratio": 2,
    "inhibition_ratio": 2,
    "swear_ratio": 1,
}


def get_trait_weights(trait: str) -> Dict[str, float]:
    """Get feature weights for a specific trait.

    Args:
        trait: One of 'overall', 'recommend_hiring', 'excited',
               'engagement', 'friendliness'.

    Returns:
        Dictionary of feature name to weight.

    Raises:
        ValueError: If trait is not recognized.
    """
    if trait not in FEATURE_WEIGHTS:
        raise ValueError(
            f"Unknown trait '{trait}'. "
            f"Must be one of: {', '.join(TRAIT_NAMES)}"
        )
    return FEATURE_WEIGHTS[trait]


def get_top_features(n: int = 10) -> List[str]:
    """Get top N most important features.

    Args:
        n: Number of features to return.

    Returns:
        List of feature names sorted by importance.
    """
    sorted_features = sorted(
        FEATURE_IMPORTANCE.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [f[0] for f in sorted_features[:n]]
