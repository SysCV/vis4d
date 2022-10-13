"""S3DIS dataset testing class."""
import unittest
from collections import defaultdict

from sympy import dsolve

from vis4d.common import COMMON_KEYS

from .s3dis import S3DIS


class S3DISTest(unittest.TestCase):
    """Test S3DIS dataloading."""

    ds = S3DIS(data_root="/data/Stanford3dDataset_v1.2")
    ds_test = S3DIS(
        data_root="/data/Stanford3dDataset_v1.2", split="testArea5"
    )

    def test_len(self):
        """Test if len of dataset correct."""
        self.assertEqual(len(self.ds), 204)

    def test_len_test_split(self):
        """Test if len of dataset correct."""
        self.assertEqual(len(self.ds_test), 68)

    def test_sample(self):
        """Test if sample loaded correctly."""
        self.assertEqual(
            tuple(self.ds[0].keys()),
            (
                COMMON_KEYS.points3d,
                COMMON_KEYS.colors3d,
                COMMON_KEYS.semantics3d,
                COMMON_KEYS.instances3d,
            ),
        )
