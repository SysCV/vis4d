"""RoI heads."""
from .base import (
    BaseSimilarityHead,
    SimilarityLearningConfig,
    build_similarity_head,
)
from .deep_sort_embedding_head import DeepSortSimilarityHead
from .quasi_dense_embedding_head import QDSimilarityHead

__all__ = [
    "BaseSimilarityHead",
    "QDSimilarityHead",
    "DeepSortSimilarityHead",
    "build_similarity_head",
    "SimilarityLearningConfig",
]
