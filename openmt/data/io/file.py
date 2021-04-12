"""Standard backend for local files on a hard drive."""
import os

from .base import BaseDataBackend, DataBackendConfig


class FileBackend(BaseDataBackend):
    """Raw file from hard disk data backend."""

    def __init__(self, cfg: DataBackendConfig):
        """Init."""
        super().__init__()
        self.cfg = DataBackendConfig(**cfg.dict())

    def get(self, filepath: str) -> bytes:
        """Get file content as bytes."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found:" f" {filepath}")
        with open(filepath, "rb") as f:
            value_buf = f.read()
        return value_buf
