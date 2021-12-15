"""RoI heads."""
from .base import (
    BaseSimilarityHead,
    SimilarityLearningConfig,
    build_similarity_head,
)

from .deepsort import DeepSortSimilarityHead
from .qdtrack import QDSimilarityHead


__all__ = [
    "BaseSimilarityHead",
    "QDSimilarityHead",
    "DeepSortSimilarityHead",
    "build_similarity_head",
    "SimilarityLearningConfig",
]
