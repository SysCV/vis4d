"""Backends for the data types a dataset of interest is saved in.

Those can be used to load data from diverse storage backends, e.g. from HDF5
files which are more suitable for data centers. The naive backend is the FileBackend, which loads from / saves to file naively.
"""
from abc import abstractmethod


class DataBackend:
    """Abstract class of storage backends.

    All backends need to implement three functions: get(), set() and exists().
    get() reads the file as a byte stream and set() writes a byte stream to a
    file. exists() checks if a certain filepath exists.
    """

    @abstractmethod
    def set(self, filepath: str, content: bytes) -> None:
        """Set the file content."""
        raise NotImplementedError

    @abstractmethod
    def get(self, filepath: str) -> bytes:
        """Get the file content as bytes."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, filepath: str) -> bool:
        """Check if filepath exists."""
        raise NotImplementedError
