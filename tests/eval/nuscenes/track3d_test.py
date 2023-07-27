"""Testcases for NuScenes 3D tracking evaluator."""
from __future__ import annotations

import unittest

import torch

from tests.eval.utils import get_dataloader
from tests.util import get_test_data
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.engine.connectors import (
    data_key,
    get_inputs_for_pred_and_data,
    pred_key,
)
from vis4d.eval.nuscenes import NuScenesTrack3DEvaluator


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

    def test_nusc_eval(self) -> None:
        """Testcase for NuScenes evaluation."""
        batch_size = 1
        data_root = get_test_data("nuscenes_test", absolute_path=False)
        nusc_eval = NuScenesTrack3DEvaluator()

        # test gt
        dataset = NuScenes(
            data_root=data_root,
            version="v1.0-mini",
            split="mini_val",
            cache_as_binary=True,
            cached_file_path=f"{data_root}/mini_val.pkl",
        )
        test_loader = get_dataloader(
            dataset, batch_size, sensors=NuScenes.CAMERAS
        )

        output = {
            "boxes_3d": [torch.zeros(batch_size, 10)],
            "velocities": [torch.zeros(batch_size, 3)],
            "class_ids": [torch.zeros(batch_size)],
            "scores_3d": [torch.zeros(batch_size)],
            "track_ids": [torch.zeros(batch_size)],
        }

        batch = next(iter(test_loader))

        nusc_eval.process_batch(
            **get_inputs_for_pred_and_data(self.CONN_NUSC_EVAL, output, batch)
        )

        _, _ = nusc_eval.evaluate("track_3d")
