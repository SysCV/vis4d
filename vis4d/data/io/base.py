"""Backends for the data types a dataset of interest is saved in."""
from abc import abstractmethod

from vis4d.common_to_revise.registry import RegistryHolder


class BaseDataBackend(metaclass=RegistryHolder):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
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
