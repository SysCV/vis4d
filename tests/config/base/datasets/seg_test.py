"""Testcase for default data pipelines for segmentation."""
import unittest

from tests.util import get_test_data
from vis4d.config.base.datasets.bdd100k_segmentation import (
    get_test_dataloader,
    get_train_dataloader,
)
from vis4d.config.util import instantiate_classes
from vis4d.data.const import CommonKeys as K


class TestSegPreprocessing(unittest.TestCase):
    """Test segmentation preprocessing pipelines."""

    def test_train_preprocessing(self):
        """Testcase for train preprocessing."""
        data_root = get_test_data("bdd100k_test/segment")
        dataloader = instantiate_classes(
            get_train_dataloader(
                f"{data_root}/images",
                f"{data_root}/labels/annotation.json",
                samples_per_gpu=1,
                workers_per_gpu=1,
            )
        )
        for data in dataloader:
            self.assertEqual(data[K.input_hw][0], [512, 1024])
            self.assertEqual(data[K.images].shape, (1, 3, 512, 1024))
            self.assertEqual(data[K.seg_masks].shape, (1, 512, 1024))

            # import numpy as np
            # from PIL import Image
            # Image.fromarray(
            #     data[K.images][0].permute(1, 2, 0).numpy().astype(np.uint8)
            # ).save("test.jpg")
            # Image.fromarray(data[K.seg_masks][0].numpy()).save("test_mask.jpg")
            # breakpoint()

            break

    def test_test_preprocessing(self):
        """Testcase for test preprocessing."""
        data_root = get_test_data("bdd100k_test/segment")
        dataloader = instantiate_classes(
            get_test_dataloader(
                f"{data_root}/images",
                f"{data_root}/labels/annotation.json",
                samples_per_gpu=1,
                workers_per_gpu=1,
            )
        )

        for data in dataloader[0]:
            self.assertEqual(data[K.input_hw][0], (720, 1280))
            self.assertEqual(data[K.images].shape, (1, 3, 720, 1280))
            self.assertEqual(data[K.seg_masks].shape, (1, 720, 1280))
            break
