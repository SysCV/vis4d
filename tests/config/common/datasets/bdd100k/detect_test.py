"""Testcase for default data pipelines for detection."""
import unittest

from tests.util import get_test_data
from vis4d.config import instantiate_classes
from vis4d.config.common.datasets.bdd100k.detect import (
    get_test_dataloader,
    get_train_dataloader,
)
from vis4d.data.const import CommonKeys as K


class TestDetPreprocessing(unittest.TestCase):
    """Test BDD100K detection preprocessing pipelines."""

    def test_train_preprocessing(self):
        """Testcase for train preprocessing."""
        data_root = get_test_data("bdd100k_test/detect")
        dataloader = instantiate_classes(
            get_train_dataloader(
                f"{data_root}/images",
                f"{data_root}/labels/annotation.json",
                samples_per_gpu=1,
                workers_per_gpu=1,
            )
        )
        for data in dataloader:
            self.assertEqual(tuple(data[K.input_hw][0]), (720, 1280))
            self.assertEqual(data[K.images].shape, (1, 3, 736, 1280))
            self.assertEqual(data[K.boxes2d][0].shape, (10, 4))
            self.assertEqual(data[K.boxes2d_classes][0].shape, (10,))
            break

    def test_test_preprocessing(self):
        """Testcase for test preprocessing."""
        data_root = get_test_data("bdd100k_test/detect")
        dataloader = instantiate_classes(
            get_test_dataloader(
                f"{data_root}/images",
                f"{data_root}/labels/annotation.json",
                keys_to_load=(K.images, K.original_hw, K.boxes2d),
                samples_per_gpu=1,
                workers_per_gpu=1,
            )
        )

        for data in dataloader[0]:
            self.assertEqual(data[K.input_hw][0], (720, 1280))
            self.assertEqual(data[K.images].shape, (1, 3, 736, 1280))
            self.assertEqual(data[K.boxes2d][0].shape, (10, 4))
            self.assertEqual(data[K.boxes2d_classes][0].shape, (10,))
            break
