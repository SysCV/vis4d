"""Hdf5 data backend."""
import os
from typing import Dict

from .base import BaseDataBackend, DataBackendConfig


class HDF5BackendConfig(DataBackendConfig):
    """Config for HDF5 data backend."""

    root_path: str


class HDF5Backend(BaseDataBackend):
    """Backend for loading data from HDF5 files."""

    def __init__(self, cfg: DataBackendConfig):
        """Init."""
        super().__init__()
        try:
            import h5py  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                "Please install h5py to enable HDF5Backend."
            ) from e

        self.cfg = HDF5BackendConfig(**cfg.__dict__)
        self.h5_file_api = h5py.File
        self.db_cache: Dict[str, h5py.File] = dict()

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath as bytes."""
        split, seq_token, column, row = filepath.split("/")
        if not split + "/" + seq_token in self.db_cache.keys():
            db_path = os.path.join(
                self.cfg.root_path, split, seq_token + ".hdf5"
            )
            if os.path.exists(db_path):
                client = self.h5_file_api(db_path, "r")
                self.db_cache[split + "/" + seq_token] = client
            else:
                raise FileNotFoundError(f"File not found: {db_path}")
        else:
            client = self.db_cache[split + "/" + seq_token]

        value_buf = client.get(column)
        if value_buf is not None:
            value_buf = value_buf.get(row)
            if value_buf is not None:
                return value_buf

        raise ValueError(f"Value {filepath} not found in {client.filename}!")

    def get_text(self, filepath: str) -> str:
        """Get values in hdf5 according to filepath as string."""
        raise NotImplementedError
