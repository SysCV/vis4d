"""Backends for the data types a dataset of interest is saved in."""
from abc import abstractmethod

from pydantic import BaseModel

from openmt.common.registry import RegistryHolder


class DataBackendConfig(BaseModel, extra="allow"):
    """Base data backend config."""

    type: str = "FileBackend"


class BaseDataBackend(metaclass=RegistryHolder):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    @abstractmethod
    def get(self, filepath: str) -> bytes:
        """Get the file content as bytes."""
        raise NotImplementedError


def build_data_backend(cfg: DataBackendConfig) -> BaseDataBackend:
    """Build a data backend from config."""
    registry = RegistryHolder.get_registry(__package__)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseDataBackend)
        return module
    raise NotImplementedError(f"Data backend {cfg.type} not found.")
