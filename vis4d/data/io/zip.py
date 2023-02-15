"""Hdf5 data backend.

This backend works with filepaths pointing to valid Zip files. We assume that
the given Zip file contains the whole dataset associated to this backend.
"""
from __future__ import annotations

import os
from zipfile import ZipFile

from .base import DataBackend


class ZipBackend(DataBackend):
    """Backend for loading data from Zip files.

    This backend works with filepaths pointing to valid Zip files. We assume
    that the given Zip file contains the whole dataset associated to this
    backend.
    """

    def __init__(self) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.db_cache: dict[str, tuple[ZipFile, str]] = {}

    @staticmethod
    def _get_zip_path(filepath: str) -> tuple[str, list[str]]:
        """Get .zip path and keys from filepath.

        Args:
            filepath (str): The filepath to retrieve the data from.
                Should have the following format: 'path/to/file.zip/key1/key2'

        Returns:
            tuple[str, list[str]]: The .zip path and the keys to retrieve.
        """
        filepath_as_list = filepath.split("/")
        keys = []

        while filepath != ".zip" and not os.path.exists(filepath):
            keys.append(filepath_as_list.pop())
            filepath = "/".join(filepath_as_list)
            # in case data_root is not explicitly set to a .zip file
            if not filepath.endswith(".zip"):
                filepath = filepath + ".zip"
        return filepath, keys

    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): Path to file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        zip_path, keys = self._get_zip_path(filepath)
        if not os.path.exists(zip_path):
            return False
        file = self._get_client(zip_path, "r")
        url = "".join(reversed(keys))
        return url in file.namelist()

    def set(self, filepath: str, content: bytes) -> None:
        """Set the file content.

        Args:
            filepath: path/to/file.zip/key1/key2/key3
            content: Bytes to be written to entry key3 within group key2
            within another group key1, for example.

        Raises:
            ValueError: If filepath is not a valid .zip file
        """
        if ".zip" not in filepath:
            raise ValueError(f"{filepath} not a valid .zip filepath!")
        # TODO: implement
        raise NotImplementedError

    def _get_client(self, zip_path: str, mode: str) -> ZipFile:
        """Get Zip client from path.

        Args:
            zip_path (str): Path to Zip file.
            mode (str): Mode to open the file in.

        Returns:
            ZipFile: the hdf5 file.
        """
        if zip_path not in self.db_cache:
            client = ZipFile(zip_path, mode)
            self.db_cache[zip_path] = (client, mode)
        else:
            client, current_mode = self.db_cache[zip_path]
            if current_mode != mode:
                client.close()
                client = ZipFile(zip_path, mode)
                self.db_cache[zip_path] = (client, mode)
        return client

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath as bytes.

        Args:
            filepath (str): The path to the file. It consists of an Zip path
                together with the relative path inside it, e.g.: "/path/to/
                file.zip/key/subkey/data". If no .zip given inside filepath,
                the function will search for the first .zip file present in
                the path, i.e. "/path/to/file/key/subkey/data" will also /key/
                subkey/data from /path/to/file.zip.

        Raises:
            ZipFileNotFoundError: If no suitable file exists.
            ValueError: If key not found inside hdf5 file.

        Returns:
            bytes: The file content in bytes
        """
        zip_path, keys = self._get_zip_path(filepath)

        if not os.path.exists(zip_path):
            raise IOError(f"Corresponding zip file not found:" f" {filepath}")
        file = self._get_client(zip_path, "r")
        url = "".join(reversed(keys))
        with file.open(url) as zfile:
            content = zfile.read()
        return bytes(content)
