"""Scoring module for ko-liwc."""

from ko_liwc.scoring.normalizer import MinMaxNormalizer
from ko_liwc.scoring.weights import (
    FEATURE_WEIGHTS,
    FEATURE_IMPORTANCE,
    get_trait_weights,
)
from ko_liwc.scoring.scorer import InterviewScorer

__all__ = [
    "MinMaxNormalizer",
    "FEATURE_WEIGHTS",
    "FEATURE_IMPORTANCE",
    "get_trait_weights",
    "InterviewScorer",
]
