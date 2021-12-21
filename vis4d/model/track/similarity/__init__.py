"""RoI heads."""
from .base import BaseSimilarityHead
from .qdtrack import QDSimilarityHead

__all__ = ["BaseSimilarityHead", "QDSimilarityHead"]
