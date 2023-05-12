"""SHIFT eval test cases."""
from __future__ import annotations

import os
import unittest
import zipfile

import torch

from tests.eval.utils import get_dataloader
from tests.util import get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.shift import SHIFT
from vis4d.eval.shift import SHIFTOnlineEvaluator


class TestOnlineEvaluator(unittest.TestCase):
    """Tests for SHIFTOnlineEvaluator."""

    base_dir = get_test_data("shift_test")
    online_evaluator = SHIFTOnlineEvaluator(
        output_dir=f"{base_dir}/submission", submission_file="test.zip"
    )
    dataset = SHIFT(
        data_root=base_dir,
        split="val",
        keys_to_load=[
            K.images,
            K.boxes2d,
            K.boxes2d_classes,
            K.boxes2d_track_ids,
            K.depth_maps,
            K.seg_masks,
            K.instance_masks,
        ],
        views_to_load=["front"],
    )
    test_loader = get_dataloader(dataset, 1, sensors=["front"])

    def test_shift_prediction(self) -> None:
        """Tests using shift data."""
        for batch in self.test_loader:
            self.online_evaluator.process_batch(
                frame_ids=batch[K.frame_ids],
                sample_names=batch[K.sample_names],
                sequence_names=batch[K.sequence_names],
                pred_sem_mask=batch["front"][K.seg_masks],
                pred_depth=batch["front"][K.depth_maps],
                pred_boxes2d=batch["front"][K.boxes2d],
                pred_boxes2d_classes=batch["front"][K.boxes2d_classes],
                pred_boxes2d_track_ids=batch["front"][K.boxes2d_track_ids],
                pred_boxes2d_scores=[
                    torch.ones_like(batch["front"][K.boxes2d_classes][0])
                ],
                pred_instance_masks=batch["front"][K.instance_masks],
            )

        self.online_evaluator.save("", "")

        assert os.path.exists(f"{self.base_dir}/submission/test.zip")
        assert zipfile.is_zipfile(f"{self.base_dir}/submission/test.zip")

        with zipfile.ZipFile(f"{self.base_dir}/submission/test.zip") as zf:
            files = [f.filename for f in zf.filelist]
        self.assertTrue("semseg/007b-4e72/00000100_semseg_front.png" in files)
        self.assertTrue("depth/007b-4e72/00000100_depth_front.png" in files)
        self.assertTrue("det_2d.json" in files)
