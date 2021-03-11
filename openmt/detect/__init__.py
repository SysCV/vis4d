"""Detection code."""
from .config import default_setup, to_detectron2
from .predict import predict, predict_func
from .train import train, train_func

__all__ = [
    "train",
    "train_func",
    "predict",
    "predict_func",
    "to_detectron2",
    "default_setup",
]
