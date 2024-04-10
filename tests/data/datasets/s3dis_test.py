"""S3DIS dataset testing class."""

import unittest

from tests.util import get_test_data
from vis4d.data.const import CommonKeys
from vis4d.data.datasets.s3dis import S3DIS


class S3DISTest(unittest.TestCase):
    """Test S3DIS dataloading."""

    dataset = S3DIS(data_root=get_test_data("s3d_test"))
    dataset_test = S3DIS(
        data_root=get_test_data("s3d_test"), split="testArea5"
    )

    def test_len(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset), 2)

    def test_len_test_split(self) -> None:
        """Test if len of dataset correct."""
        self.assertEqual(len(self.dataset_test), 2)

    def test_sample(self) -> None:
        """Test if sample loaded correctly."""
        self.assertEqual(
            tuple(self.dataset[0].keys()),
            (
                CommonKeys.points3d,
                CommonKeys.colors3d,
                CommonKeys.semantics3d,
                CommonKeys.instances3d,
            ),
        )
