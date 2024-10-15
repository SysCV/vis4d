"""Testcases for hdf5 dataset conversion."""

import os
import unittest

from tests.util import get_test_data
from vis4d.data.io.to_hdf5 import convert_dataset


class TestHDF5(unittest.TestCase):
    """build Testcase class."""

    def test_convert_single_dataset(self) -> None:
        """Testcase for convert_single_dataset."""
        path = get_test_data("bdd100k_test/track/")
        convert_dataset(os.path.join(path, "images"))
        # check if file was created
        self.assertTrue(os.path.exists(os.path.join(path, "images.hdf5")))
        # allow up to 3 KByte deviation due to platform specifics
        self.assertAlmostEqual(
            os.stat(os.path.join(path, "images.hdf5")).st_size / 1024,
            os.stat(os.path.join(path, "images_.hdf5")).st_size / 1024,
            delta=3.0,
        )
        # check if clause aborting when file exists
        convert_dataset(os.path.join(path, "images"))
        # cleanup: remove file
        os.remove(os.path.join(path, "images.hdf5"))
