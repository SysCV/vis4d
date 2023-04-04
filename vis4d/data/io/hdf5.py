"""Hdf5 data backend.

This backend works with filepaths pointing to valid HDF5 files. We assume that
the given HDF5 file contains the whole dataset associated to this backend.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm

from vis4d.common.imports import H5PY_AVAILABLE

from .base import DataBackend

if H5PY_AVAILABLE:
    import h5py
    from h5py import File

    t_vlen_uint8 = h5py.special_dtype(vlen=np.uint8)
else:
    File = None  # pylint: disable=invalid-name


class HDF5Backend(DataBackend):
    """Backend for loading data from HDF5 files.

    This backend works with filepaths pointing to valid HDF5 files. We assume
    that the given HDF5 file contains the whole dataset associated to this
    backend.

    You can use the provided script at vis4d/data/datasets/to_hdf5.py to
    convert your dataset to the expected hdf5 format before using this backend.
    """

    def __init__(self) -> None:
        """Creates an instance of the class."""
        super().__init__()
        if not H5PY_AVAILABLE:
            raise ImportError("Please install h5py to enable HDF5Backend.")
        self.db_cache: dict[str, File] = {}

    @staticmethod
    def _get_hdf5_path(filepath: str) -> tuple[str, list[str]]:
        """Get .hdf5 path and keys from filepath.

        Args:
            filepath (str): The filepath to retrieve the data from.
                Should have the following format: 'path/to/file.hdf5/key1/key2'

        Returns:
            tuple[str, list[str]]: The .hdf5 path and the keys to retrieve.
        """
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
        """Check if filepath exists.

        Args:
            filepath (str): Path to file.

        Returns:
            bool: True if file exists, False otherwise.
        """
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

    def _get_client(self, hdf5_path: str, mode: str) -> File:
        """Get HDF5 client from path.

        Args:
            hdf5_path (str): Path to HDF5 file.
            mode (str): Mode to open the file in.

        Returns:
            File: the hdf5 file.
        """
        if hdf5_path not in self.db_cache:
            client = File(hdf5_path, mode)
            self.db_cache[hdf5_path] = [client, mode]
        else:
            client, current_mode = self.db_cache[hdf5_path]
            if current_mode != mode:
                client.close()
                client = File(hdf5_path, mode)
                self.db_cache[hdf5_path] = [client, mode]
        return client

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath as bytes.

        Args:
            filepath (str): The path to the file. It consists of an HDF5 path
                together with the relative path inside it, e.g.: "/path/to/
                file.hdf5/key/subkey/data". If no .hdf5 given inside filepath,
                the function will search for the first .hdf5 file present in
                the path, i.e. "/path/to/file/key/subkey/data" will also /key/
                subkey/data from /path/to/file.hdf5.

        Raises:
            FileNotFoundError: If no suitable file exists.
            ValueError: If key not found inside hdf5 file.

        Returns:
            bytes: The file content in bytes
        """
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
                raise ValueError(f"Value {url} not found in {hdf5_path}!")

        return bytes(value_buf[()])


def convert_dataset(source_dir: str) -> None:
    """Convert particular dataset instance to hdf5."""
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"No such file or directory: {source_dir}")

    source_dir = os.path.join(source_dir, "")  # must end with trailing slash
    hdf5_path = source_dir.rstrip("/") + ".hdf5"
    if os.path.exists(hdf5_path):
        print(f"File {hdf5_path} already exists! Skipping {source_dir}")
        return

    print(f"Converting dataset at: {source_dir}")
    hdf5_file = h5py.File(hdf5_path, mode="w")
    sub_dirs = list(os.walk(source_dir))
    file_count = sum(len(files) for (_, _, files) in sub_dirs)

    with tqdm(total=file_count) as pbar:
        for root, _, files in sub_dirs:
            g_name = root.replace(source_dir, "")
            g = hdf5_file.create_group(g_name) if g_name else hdf5_file
            for f in files:
                filepath = os.path.join(root, f)
                if os.path.isfile(filepath):
                    with open(filepath, "rb") as fp:
                        file_content = fp.read()
                    g.create_dataset(
                        f, data=np.frombuffer(file_content, dtype="uint8")
                    )
                pbar.update()

    hdf5_file.close()
    print("done.")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Converts a dataset at the specified path to hdf5."
    )
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="path to the root folder of a specific dataset to convert",
    )
    args = parser.parse_args()
    convert_dataset(args.path)
