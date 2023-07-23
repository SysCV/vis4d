"""Testcases for NuScenes 3D detection evaluator."""
from __future__ import annotations

import unittest

import torch

from tests.util import get_test_data
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.data.loader import build_inference_dataloaders, multi_sensor_collate
from vis4d.engine.connectors import data_key, get_multi_sensor_inputs, pred_key
from vis4d.eval.nuscenes import NuScenesDet3DEvaluator


class TestNuScenesDet3DEvaluator(unittest.TestCase):
    """NuScenes evaluator testcase class."""

    CONN_NUSC_EVAL = {
        "tokens": data_key("token"),
        "boxes_3d": pred_key("boxes_3d"),
        "velocities": pred_key("velocities"),
        "class_ids": pred_key("class_ids"),
        "scores_3d": pred_key("scores_3d"),
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
        data_root = get_test_data("nuscenes_test")

        nusc_eval = NuScenesDet3DEvaluator(
            data_root=data_root, version="v1.0-mini", split="mini_val"
        )

        # test gt
        dataset = NuScenes(
            data_root=data_root,
            version="v1.0-mini",
            split="mini_val",
        )
        test_loader = build_inference_dataloaders(
            dataset,
            samples_per_gpu=batch_size,
            workers_per_gpu=1,
            collate_fn=multi_sensor_collate,
        )[0]

        output = {
            "boxes_3d": torch.zeros(batch_size, 10),
            "velocities": torch.zeros(batch_size, 3),
            "class_ids": torch.zeros(batch_size),
            "scores_3d": torch.zeros(batch_size),
        }

        batch = next(iter(test_loader))

        nusc_eval.process_batch(
            **get_multi_sensor_inputs(
                self.CONN_NUSC_EVAL, output, batch, self.CAMERAS
            )
        )

        _, _ = nusc_eval.evaluate("detect_3d")
