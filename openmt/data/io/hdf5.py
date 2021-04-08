"""Hdf5 data backend."""
import os
from typing import Dict

from .base import BaseDataBackend, DataBackendConfig


class HDF5Backend(BaseDataBackend):
    """Backend for loading data from HDF5 files.

    This backend works with the same filepaths as the file backend. However,
    we assume that all dataset subfolders do not contain any other hdf5 files,
    so that this backend can back-track from each filename to the closest hdf5
    file in the path hierarchy in order to find its corresponding hdf5 blob,
    e.g.:
    /path/to/dataset/images/first_image.png
    will first search in /path/to/dataset for images.hdf5, next it will try
    /path/to/dataset.hdf5 and so on.
    "images.hdf5" should contain the binary image data of "first_image.png"
    at group first_image.png, column raw.
    You can use the provided script at tools/datasets_to_hdf5.py to convert
    your dataset to the expected hdf5 format before using this backend.
    """

    def __init__(self, cfg: DataBackendConfig):
        """Init."""
        super().__init__()
        try:
            import h5py  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                "Please install h5py to enable HDF5Backend."
            ) from e
        self.cfg = DataBackendConfig(**cfg.__dict__)
        self.h5_file_api = h5py.File
        self.db_cache: Dict[str, h5py.File] = dict()

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath as bytes."""
        db_path = os.path.dirname(filepath).strip("/")
        while not os.path.exists(db_path + ".hdf5"):
            db_path = os.path.dirname(db_path).strip("/")
            if db_path == ".hdf5":
                raise FileNotFoundError(
                    f"Corresponding HDF5 file not found:" f" {filepath}"
                )

        if not db_path in self.db_cache.keys():
            client = self.h5_file_api(db_path + ".hdf5", "r")
            self.db_cache[db_path] = client
        else:
            client = self.db_cache[db_path]

        key = filepath[len(db_path) :].strip("/")
        value_buf = client.get(key)
        if value_buf is not None:
            return bytes(value_buf["raw"][0])

        raise ValueError(f"Value {filepath} not found in {client.filename}!")

    def get_text(self, filepath: str) -> str:
        """Get values in hdf5 according to filepath as string."""
        raise NotImplementedError
