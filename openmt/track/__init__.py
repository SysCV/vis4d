"""Tracking code."""
from .predict import predict, predict_func
from .train import train, train_func

__all__ = [
    "train",
    "train_func",
    "predict",
    "predict_func",
]
