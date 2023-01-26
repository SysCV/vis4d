"""Backends for the data types a dataset of interest is saved in.

Those can be used to load data from diverse storage backends, e.g. from HDF5
files which are more suitable for data centers. The naive backend is the
FileBackend, which loads from / saves to file naively.
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
        """Set the file content at the given filepath.

        Args:
            filepath (str): The filepath to store the data at.
            content (bytes): The content to store as bytes.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, filepath: str) -> bytes:
        """Get the file content at the given filepath as bytes.

        Args:
            filepath (str): The filepath to retrieve the data from."

        Returns:
            bytes: The content of the file as bytes.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): The filepath to check.

        Returns:
            bool: True if the filepath exists, False otherwise.
        """
        raise NotImplementedError
