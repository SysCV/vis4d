"""data module init."""
from .build import build_test_loader, build_train_loader

__all__ = [
    "build_train_loader",
    "build_test_loader",
]
