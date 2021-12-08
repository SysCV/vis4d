"""RoI heads."""
from .base import (
    BaseSimilarityHead,
    SimilarityLearningConfig,
    build_similarity_head,
)
from .qdtrack import QDSimilarityHead

__all__ = [
    "BaseSimilarityHead",
    "QDSimilarityHead",
    "build_similarity_head",
    "SimilarityLearningConfig",
]
