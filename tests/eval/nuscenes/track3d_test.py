"""Testcases for NuScenes evaluator."""
from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.data.loader import (
    VideoDataPipe,
    build_inference_dataloaders,
    multi_sensor_collate,
)
from vis4d.engine.connectors import data_key, get_multi_sensor_inputs, pred_key
from vis4d.eval.nuscenes import NuScenesEvaluator


def get_dataloader(datasets: Dataset, batch_size: int) -> DataLoader:
    """Get data loader for testing."""
    datapipe = VideoDataPipe(datasets)
    return build_inference_dataloaders(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
        collate_fn=multi_sensor_collate,
    )[0]


class TestNuScenesEvaluator(unittest.TestCase):
    """NuScenes evaluator testcase class."""

    CONN_BBOX_3D_TEST = {
        K.images: K.images,
        K.original_hw: "images_hw",
        K.intrinsics: K.intrinsics,
        K.extrinsics: K.extrinsics,
        K.frame_ids: K.frame_ids,
    }

    CONN_NUSC_EVAL = {
        "tokens": data_key("token"),
        "boxes_3d": pred_key("boxes_3d"),
        "class_ids": pred_key("class_ids"),
        "scores_3d": pred_key("scores_3d"),
        "track_ids": pred_key("track_ids"),
    }

    CAMERAS = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def test_nusc_eval(self) -> None:
        """Testcase for NuScenes evaluation."""
        batch_size = 1
        nusc_eval = NuScenesEvaluator()

        # test gt
        dataset = NuScenes(
            data_root=get_test_data("nuscenes_test"),
            version="v1.0-mini",
            split="mini_val",
            metadata=["use_camera"],
        )
        test_loader = get_dataloader(dataset, batch_size)

        output = {
            "boxes_3d": torch.zeros(batch_size, 12),
            "class_ids": torch.zeros(batch_size),
            "scores_3d": torch.zeros(batch_size),
            "track_ids": torch.zeros(batch_size),
        }

        batch = next(iter(test_loader))

        nusc_eval.process_batch(
            **get_multi_sensor_inputs(
                self.CONN_NUSC_EVAL, output, batch, self.CAMERAS
            )
        )

        _, _ = nusc_eval.evaluate("detect_3d")
        _, _ = nusc_eval.evaluate("track_3d")