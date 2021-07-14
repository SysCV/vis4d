"""Hdf5 data backend."""
import os
from typing import Dict

from .base import BaseDataBackend, DataBackendConfig


class HDF5Backend(BaseDataBackend):
    """Backend for loading data from HDF5 files.

    This backend works with filepaths pointing to valid HDF5 files. We assume
    that the given HDF5 file contains the whole dataset associated to this
    backend.

    You can use the provided script at tools/datasets_to_hdf5.py to convert
    your dataset to the expected hdf5 format before using this backend.
    """

    def __init__(self, cfg: DataBackendConfig):
        """Init."""
        super().__init__()
        try:
            import h5py  # pylint: disable=import-outside-toplevel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Please install h5py to enable HDF5Backend."
            ) from e
        self.cfg = DataBackendConfig(**cfg.dict())
        self.h5_file_api = h5py.File
        self.is_hdf5 = h5py.is_hdf5
        self.db_cache: Dict[str, h5py.File] = dict()

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath as bytes."""
        filepath_as_list = filepath.split("/")
        keys = []

        while filepath and not self.is_hdf5(filepath):
            keys.append(filepath_as_list.pop())
            filepath = "/".join(filepath_as_list)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Corresponding HDF5 file not found:" f" {filepath}"
            )

        if filepath not in self.db_cache.keys():
            client = self.h5_file_api(filepath, "r")
            self.db_cache[filepath] = client
        else:
            client = self.db_cache[filepath]

        url = "/".join(reversed(keys))
        value_buf = client
        while keys:
            value_buf = value_buf.get(keys.pop())

        if value_buf is not None:
            return bytes(value_buf[()])

        raise ValueError(f"Value {url} not found in {filepath}!")
