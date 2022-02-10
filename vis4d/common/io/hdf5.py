"""Hdf5 data backend."""
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise ImportError("Please install h5py to enable HDF5Backend.") from e

from .base import BaseDataBackend


class HDF5Backend(BaseDataBackend):
    """Backend for loading data from HDF5 files.

    This backend works with filepaths pointing to valid HDF5 files. We assume
    that the given HDF5 file contains the whole dataset associated to this
    backend.

    You can use the provided script at vis4d/data/datasets/to_hdf5.py to
    convert your dataset to the expected hdf5 format before using this backend.
    """

    def __init__(self) -> None:
        """Init."""
        super().__init__()
        self.db_cache: Dict[str, h5py.File] = {}

    @staticmethod
    def _get_hdf5_path(filepath: str) -> Tuple[str, List[str]]:
        """Get .hdf5 path and keys from filepath."""
        filepath_as_list = filepath.split("/")
        keys = []

        while filepath != ".hdf5" and not h5py.is_hdf5(filepath):
            keys.append(filepath_as_list.pop())
            filepath = "/".join(filepath_as_list)
            # in case data_root is not explicitly set to a .hdf5 file
            if not filepath.endswith(".hdf5"):
                filepath = filepath + ".hdf5"
        return filepath, keys

    def exists(self, filepath: str) -> bool:
        """Check if filepath exists."""
        hdf5_path, keys = self._get_hdf5_path(filepath)
        if not os.path.exists(hdf5_path):
            return False
        value_buf = self._get_client(hdf5_path, "r")

        while keys:
            value_buf = value_buf.get(keys.pop())
            if value_buf is None:
                return False
        return True

    def set(self, filepath: str, content: bytes) -> None:
        """Set the file content.

        Args:
            filepath: path/to/file.hdf5/key1/key2/key3
            content: Bytes to be written to entry key3 within group key2
            within another group key1, for example.

        Raises:
            ValueError: If filepath is not a valid .hdf5 file
        """
        if ".hdf5" not in filepath:
            raise ValueError(f"{filepath} not a valid .hdf5 filepath!")
        hdf5_path, keys_str = filepath.split(".hdf5")
        key_list = keys_str.split("/")
        file = self._get_client(hdf5_path + ".hdf5", "a")
        if len(key_list) > 1:
            group_str = "/".join(key_list[:-1])
            if group_str == "":
                group_str = "/"

            group = file[group_str]
            key = key_list[-1]
            group.create_dataset(
                key, data=np.frombuffer(content, dtype="uint8")
            )

    def _get_client(self, hdf5_path: str, mode: str) -> h5py.File:
        """Get HDF5 client from path."""
        if hdf5_path not in self.db_cache:
            client = h5py.File(hdf5_path, mode)
            self.db_cache[hdf5_path] = [client, mode]
        else:
            client, current_mode = self.db_cache[hdf5_path]
            if current_mode != mode:
                client.close()
                client = h5py.File(hdf5_path, mode)
                self.db_cache[hdf5_path] = [client, mode]
        return client

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath as bytes."""
        hdf5_path, keys = self._get_hdf5_path(filepath)

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(
                f"Corresponding HDF5 file not found:" f" {filepath}"
            )
        value_buf = self._get_client(hdf5_path, "r")
        url = "/".join(reversed(keys))
        while keys:
            value_buf = value_buf.get(keys.pop())
            if value_buf is None:
                raise ValueError(f"Value {url} not found in {filepath}!")

        return bytes(value_buf[()])
