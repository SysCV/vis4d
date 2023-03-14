"""Testcases for data backends."""
import os
import sys
import unittest

from tests.util import get_test_data
from vis4d.data.io import FileBackend, HDF5Backend, ZipBackend
from vis4d.data.io.util import str_decode


class TestBackends(unittest.TestCase):
    """Testcases for ensuring equal output of each backend."""

    def test_get(self) -> None:
        """Test image retrieval from different backends."""
        backend_file = FileBackend()
        backend_hdf5 = HDF5Backend()
        backend_zip = ZipBackend()
        base_path = get_test_data("bdd100k_test/track/")
        img = "00091078-875c1f73"
        sample_path = f"{base_path}/images/{img}/{img}-0000166.jpg"
        hdf5_path = f"{base_path}/images_.hdf5/{img}/{img}-0000166.jpg"
        zip_path = f"{base_path}/images_.zip/{img}/{img}-0000166.jpg"

        # check get
        out_file = backend_file.get(sample_path)
        out_hdf5 = backend_hdf5.get(hdf5_path)
        out_zip = backend_zip.get(zip_path)
        self.assertTrue(out_file == out_hdf5)
        self.assertTrue(out_file == out_zip)

        # check exists
        self.assertFalse(backend_hdf5.exists("invalid_path"))
        invalid_inhdf5 = f"{base_path}/images_.hdf5/invalid_path"
        self.assertFalse(backend_hdf5.exists(invalid_inhdf5))
        self.assertTrue(backend_hdf5.exists(hdf5_path))

        self.assertFalse(backend_zip.exists("invalid_path"))
        invalid_inzip = f"{base_path}/images_.zip/invalid_path"
        self.assertFalse(backend_zip.exists(invalid_inzip))
        self.assertTrue(backend_zip.exists(zip_path))

        # check set
        os.makedirs("./unittests/", exist_ok=True)
        backend_file.set("./unittests/test_file.bin", bytes())
        self.assertTrue(os.path.exists("./unittests/test_file.bin"))

        # check db_cache
        backend_hdf5.get(hdf5_path)
        backend_zip.get(zip_path)

        self.assertRaises(FileNotFoundError, backend_file.get, "invalid_path")
        self.assertRaises(FileNotFoundError, backend_hdf5.get, "invalid_path")
        self.assertRaises(FileNotFoundError, backend_zip.get, "invalid_path")

        invalid_hdf5_path = f"{base_path}/images_.hdf5/000/000.jpg"
        invalid_zip_path = f"{base_path}/images_.zip/000/000.jpg"
        self.assertRaises(ValueError, backend_hdf5.get, invalid_hdf5_path)
        self.assertRaises(ValueError, backend_zip.get, invalid_zip_path)

    def test_str_decode(self) -> None:
        """Test str decode method in utils."""
        my_str = "Hello world!"
        gen_str = str_decode(my_str.encode(sys.getdefaultencoding()))
        self.assertTrue(my_str == gen_str)
