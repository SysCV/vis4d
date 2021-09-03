"""Testcases for data backends."""
import sys
import unittest

from vist.data.utils import str_decode

from ..utils import timeit
from .base import DataBackendConfig
from .file import FileBackend
from .hdf5 import HDF5Backend


class TestBackends(unittest.TestCase):
    """Testcases for ensuring equal output of each backend."""

    @timeit  # type: ignore
    def test_get(self) -> None:
        """Test image retrieval from different backends."""
        backend_file = FileBackend(DataBackendConfig(type="file"))
        backend_hdf5 = HDF5Backend(DataBackendConfig(type="hdf5"))
        sample_path = (
            "vist/engine/testcases/track/bdd100k-samples/images/"
            "00091078-875c1f73/00091078-875c1f73-0000166.jpg"
        )
        hdf5_path = (
            "vist/engine/testcases/track/bdd100k-samples/images_.hdf5/"
            "00091078-875c1f73/00091078-875c1f73-0000166.jpg"
        )

        out_file = backend_file.get(sample_path)
        out_hdf5 = backend_hdf5.get(hdf5_path)
        self.assertTrue(out_file == out_hdf5)

        # check db_cache
        backend_hdf5.get(hdf5_path)

        self.assertRaises(FileNotFoundError, backend_file.get, "invalid_path")
        self.assertRaises(FileNotFoundError, backend_hdf5.get, "invalid_path")

        invalid_hdf5_path = (
            "vist/engine/testcases/track/bdd100k-samples/images_.hdf5/"
            "000/000.jpg"
        )
        self.assertRaises(ValueError, backend_hdf5.get, invalid_hdf5_path)

    def test_str_decode(self) -> None:
        """Test str decode method in utils."""
        my_str = "Hello world!"
        gen_str = str_decode(my_str.encode(sys.getdefaultencoding()))
        self.assertTrue(my_str == gen_str)
