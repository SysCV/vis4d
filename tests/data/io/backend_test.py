"""Testcases for data backends."""

import os
import shutil
import sys
import tempfile
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

        # check isfile
        self.assertTrue(backend_file.isfile(sample_path))
        self.assertFalse(backend_file.isfile(os.path.dirname(sample_path)))
        self.assertTrue(backend_hdf5.isfile(hdf5_path))
        self.assertFalse(backend_hdf5.isfile(os.path.dirname(hdf5_path)))
        self.assertTrue(backend_zip.isfile(zip_path))
        self.assertFalse(backend_zip.isfile(os.path.dirname(zip_path)))

        # check listdir
        list_file = backend_file.listdir(os.path.dirname(sample_path))
        list_hdf5 = backend_hdf5.listdir(os.path.dirname(hdf5_path))
        list_zip = backend_zip.listdir(os.path.dirname(zip_path))
        self.assertTrue(len(list_file) == len(list_hdf5) == len(list_zip))
        self.assertTrue(list_file == list_hdf5 == list_zip)

        # check set
        test_dir = tempfile.mkdtemp()
        backend_file.set(f"{test_dir}/test_file.bin", bytes())

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

        # Remove test file
        shutil.rmtree(test_dir)

    def test_str_decode(self) -> None:
        """Test str decode method in utils."""
        my_str = "Hello world!"
        gen_str = str_decode(my_str.encode(sys.getdefaultencoding()))
        self.assertTrue(my_str == gen_str)

    def test_get_path(self) -> None:
        """Test file path parsing."""
        backend_hdf5 = HDF5Backend()
        backend_zip = ZipBackend()

        (
            path,
            keys,
        ) = backend_hdf5._get_hdf5_path(  # pylint: disable=protected-access
            "/usr/test.hdf5/test/test.jpg", allow_omitted_ext=False
        )
        self.assertEqual(path, "/usr/test.hdf5")
        self.assertEqual(keys, ["test.jpg", "test"])

        (
            path,
            keys,
        ) = backend_zip._get_zip_path(  # pylint: disable=protected-access
            "/usr/test.zip/test/test.jpg", allow_omitted_ext=False
        )
        self.assertEqual(path, "/usr/test.zip")
        self.assertEqual(keys, ["test.jpg", "test"])
