"""Test loader components."""
import unittest

import torch
from torch import Tensor

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets import COCO
from vis4d.data.loader import (
    build_inference_dataloaders,
    build_train_dataloader,
    default_collate,
)
from vis4d.data.transforms import compose, normalize, pad, resize, to_tensor


class DataLoaderTest(unittest.TestCase):
    """Test loader components."""

    def test_train_loader(self) -> None:
        """Test the data loading pipeline."""
        coco = COCO(data_root=get_test_data("coco_test"), split="train")
        batch_size = 2
        preprocess_fn = compose(
            [
                resize.GenResizeParameters((256, 256), keep_ratio=True),
                resize.ResizeImages(),
                normalize.NormalizeImages(),
            ]
        )
        batchprocess_fn = compose([pad.PadImages(), to_tensor.ToTensor()])

        datapipe = DataPipe(coco, preprocess_fn)
        train_loader = build_train_dataloader(
            datapipe,
            samples_per_gpu=batch_size,
            batchprocess_fn=batchprocess_fn,
        )

        sample = next(iter(train_loader))
        assert isinstance(sample[K.images], Tensor)
        assert batch_size == sample[K.images].size(0)
        assert batch_size == len(sample[K.boxes2d])
        assert sample[K.boxes2d][0].shape[1] == 4
        assert sample[K.images].shape[1] == 3

    def test_inference_loader(self) -> None:
        """Test the data loading pipeline."""
        coco = COCO(data_root=get_test_data("coco_test"), split="train")
        preprocess_fn = compose(
            [
                resize.GenResizeParameters((256, 256), keep_ratio=True),
                resize.ResizeImages(),
                normalize.NormalizeImages(),
            ]
        )
        batchprocess_fn = compose([pad.PadImages(), to_tensor.ToTensor()])

        datapipe = DataPipe(coco, preprocess_fn)
        test_loader = build_inference_dataloaders(
            datapipe, batchprocess_fn=batchprocess_fn
        )[0]

        sample = next(iter(test_loader))
        assert isinstance(sample[K.images], Tensor)
        assert sample[K.images].size(0) == 1
        assert len(sample[K.boxes2d]) == 1
        assert sample[K.boxes2d][0].shape == torch.Size([14, 4])
        assert sample[K.images].shape == torch.Size([1, 3, 192, 256])

    def test_default_collate(self) -> None:
        """Test the default collate."""
        t1, t2 = torch.rand(2, 3), torch.rand(2, 3)
        data = [{K.seg_masks: t1}, {K.seg_masks: t2}]
        col_data = default_collate(data)
        assert (col_data[K.seg_masks] == torch.stack([t1, t2])).all()

        col_data = default_collate([{"name": ["a"]}, {"name": ["b"]}])
        assert col_data["name"] == [["a"], ["b"]]

        with self.assertRaises(RuntimeError):
            col_data = default_collate(
                [
                    {K.seg_masks: torch.rand(1, 3)},
                    {K.seg_masks: torch.rand(2, 3)},
                ],
                collate_keys=[K.seg_masks],
            )
