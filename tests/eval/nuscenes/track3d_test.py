"""Testcases for NuScenes 3D tracking evaluator."""
from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.data.loader import build_inference_dataloaders, multi_sensor_collate
from vis4d.engine.connectors import data_key, get_multi_sensor_inputs, pred_key
from vis4d.eval.nuscenes import NuScenesTrack3DEvaluator


def get_dataloader(datasets: Dataset, batch_size: int) -> DataLoader:
    """Get data loader for testing."""
    return build_inference_dataloaders(
        datasets,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
        collate_fn=multi_sensor_collate,
    )[0]


class TestNuScenesTrack3DEvaluator(unittest.TestCase):
    """NuScenes evaluator testcase class."""

    CONN_NUSC_EVAL = {
        "tokens": data_key("token"),
        "boxes_3d": pred_key("boxes_3d"),
        "velocities": pred_key("velocities"),
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
        nusc_eval = NuScenesTrack3DEvaluator()

        # test gt
        dataset = NuScenes(
            data_root=get_test_data("nuscenes_test"),
            version="v1.0-mini",
            split="mini_val",
        )
        test_loader = get_dataloader(dataset, batch_size)

        output = {
            "boxes_3d": torch.zeros(batch_size, 10),
            "velocities": torch.zeros(batch_size, 3),
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

        _, _ = nusc_eval.evaluate("track_3d")
