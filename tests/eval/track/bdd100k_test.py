"""Testcases for NuScenes evaluator."""
from __future__ import annotations

import os.path as osp
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as CK
from vis4d.data.datasets.bdd100k import BDD100K
from vis4d.data.loader import VideoDataPipe, build_inference_dataloaders
from vis4d.engine.connectors import (
    DataConnectionInfo,
    StaticDataConnector,
    data_key,
    pred_key,
)
from vis4d.eval.track.bdd100k import BDD100KEvaluator


def get_dataloader(datasets: Dataset, batch_size: int) -> DataLoader:
    """Get data loader for testing."""
    datapipe = VideoDataPipe(datasets)
    return build_inference_dataloaders(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
    )[0]


class TestBDD100KEvaluator(unittest.TestCase):
    """BDD100K tracking evaluator testcase class."""

    CONN_BBOX_2D_TEST = {
        CK.images: CK.images,
        CK.input_hw: "images_hw",
        CK.frame_ids: CK.frame_ids,
    }

    CONN_BDD100K_EVAL = {
        "frame_ids": data_key("frame_ids"),
        "data_names": data_key("name"),
        "video_names": data_key("videoName"),
        "boxes_list": pred_key("boxes"),
        "class_ids_list": pred_key("class_ids"),
        "scores_list": pred_key("scores"),
        "track_ids_list": pred_key("track_ids"),
    }

    def test_nusc_eval(self) -> None:
        """Testcase for NuScenes evaluation."""
        batch_size = 2

        data_root = osp.join(get_test_data("bdd100k_test"), "track/images")
        annotations = osp.join(get_test_data("bdd100k_test"), "track/labels")
        config = osp.join(get_test_data("bdd100k_test"), "track/config.toml")

        scalabel_eval = BDD100KEvaluator(annotation_path=annotations)

        # test gt
        dataset = BDD100K(
            data_root=data_root,
            annotation_path=annotations,
            config_path=config,
        )
        test_loader = get_dataloader(dataset, batch_size)

        data_connection_info = DataConnectionInfo(
            test=self.CONN_BBOX_2D_TEST,
            callbacks={"bdd100k_eval_test": self.CONN_BDD100K_EVAL},
        )

        data_connector = StaticDataConnector(
            connections=data_connection_info,
        )

        output = {
            "boxes": [torch.zeros(batch_size, 4)],
            "class_ids": [torch.zeros(batch_size)],
            "scores": [torch.zeros(batch_size)],
            "track_ids": [torch.zeros(batch_size)],
        }

        batch = next(iter(test_loader))
        clbk_kwargs = data_connector.get_callback_input(
            "bdd100k_eval", output, batch, "test"
        )

        scalabel_eval.process(**clbk_kwargs)