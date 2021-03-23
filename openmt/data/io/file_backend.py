"""Standard backend for local files on a hard drive."""

from .base_backend import BaseDataBackend, DataBackendConfig


class FileBackend(BaseDataBackend):
    """Raw file from hard disk data backend."""

    def __init__(self, cfg: DataBackendConfig):
        """Init."""
        super().__init__()
        self.cfg = DataBackendConfig(**cfg.__dict__)

    def get(self, filepath: str) -> bytes:
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath: str) -> str:
        filepath = str(filepath)
        with open(filepath, 'r') as f:
            value_buf = f.read()
        return value_buf
