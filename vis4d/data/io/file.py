"""Standard backend for local files on a hard drive."""
import os

from .base import DataBackend


class FileBackend(DataBackend):
    """Raw file from hard disk data backend."""

    def exists(self, filepath: str) -> bool:
        """Check if filepath exists."""
        return os.path.exists(filepath)

    def set(self, filepath: str, content: bytes) -> None:
        """Write the file content to disk.

        Args:
            filepath (str): Path to file.
            content (bytes): Content to write in bytes.
        """
        with open(filepath, "wb") as f:
            f.write(content)

    def get(self, filepath: str) -> bytes:
        """Get file content as bytes.

        Args:
            filepath (str): Path to file.

        Raises:
            FileNotFoundError: If filepath does not exist.

        Returns:
            bytes: File content as bytes.
        """
        if not self.exists(filepath):
            raise FileNotFoundError(f"File not found:" f" {filepath}")
        with open(filepath, "rb") as f:
            value_buf = f.read()
        return value_buf
