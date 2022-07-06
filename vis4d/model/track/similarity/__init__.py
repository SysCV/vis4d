"""RoI heads."""
from .base import BaseSimilarityHead
from .deepsort import DeepSortSimilarityHead
from .qdtrack import QDSimilarityHead

__all__ = ["BaseSimilarityHead", "QDSimilarityHead", "DeepSortSimilarityHead"]
