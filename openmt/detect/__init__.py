"""Detection code."""
from .config import default_setup, to_detectron2
from .predict import predict
from .train import train

__all__ = [
    "train",
    "predict",
    "to_detectron2",
    "default_setup",
]
