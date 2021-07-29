"""RoI heads."""
from .base import (
    BaseSimilarityHead,
    SimilarityLearningConfig,
    build_similarity_head,
)
from .quasi_dense_embedding_head import QDSimilarityHead

__all__ = [
    "BaseSimilarityHead",
    "QDSimilarityHead",
    "build_similarity_head",
    "SimilarityLearningConfig",
]
