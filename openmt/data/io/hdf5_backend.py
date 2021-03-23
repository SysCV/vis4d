"""Hdf5 data backend."""
import os

import numpy as np

from .base_backend import BaseDataBackend, DataBackendConfig


class HDF5BackendConfig(DataBackendConfig):
    root_path: str


class HDF5Backend(BaseDataBackend):

    def __init__(self, cfg: DataBackendConfig):
        """Init."""
        super().__init__()
        try:
            import h5py
        except ImportError:
            raise ImportError('Please install h5py to enable HDF5Backend.')

        self.cfg = HDF5BackendConfig(**cfg.__dict__)
        self.h5_file_api = h5py.File
        self.db_cache = dict()

    def get(self, filepath):
        """Get values according to the filepath.
        Args:
            filepath (str | obj:`Path`): Here, filepath is the hdf5 key.
        """
        split, seq_token, column, row = filepath.split('/')
        if not split + '/' + seq_token in self.db_cache.keys():
            db_path = os.path.join(self.cfg.root_path, split, seq_token + '.hdf5')
            if os.path.exists(db_path):
                client = self.h5_file_api(db_path, 'r')
                self.db_cache[split + '/' + seq_token] = client
            else:
                return None
        else:
            client = self.db_cache[split + '/' + seq_token]

        value_buf = client.get(column)
        if value_buf is not None:
            value_buf = value_buf.get(row)
            if value_buf is not None:
                value_buf = np.array(client[column][row])
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError
