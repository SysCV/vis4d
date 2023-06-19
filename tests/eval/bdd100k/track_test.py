"""Testcases for BDD100K tracking evaluator."""
from __future__ import annotations

import os.path as osp
import unittest

from torch.utils.data import DataLoader, Dataset

from tests.util import generate_boxes, get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets.bdd100k import BDD100K
from vis4d.data.loader import build_inference_dataloaders
from vis4d.engine.connectors import (
    data_key,
    get_inputs_for_pred_and_data,
    pred_key,
)
from vis4d.eval.bdd100k import BDD100KTrackEvaluator


def get_dataloader(datasets: Dataset, batch_size: int) -> DataLoader:
    """Get data loader for testing."""
    datapipe = DataPipe(datasets)
    return build_inference_dataloaders(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=0
    )[0]


class TestBDD100KTrackEvaluator(unittest.TestCase):
    """BDD100K tracking evaluator testcase class."""

    CONN_BDD100K_EVAL = {
        "frame_ids": data_key(K.frame_ids),
        "data_names": data_key(K.sample_names),
        "video_names": data_key(K.sequence_names),
        "boxes_list": pred_key("boxes"),
        "class_ids_list": pred_key("class_ids"),
        "scores_list": pred_key("scores"),
        "track_ids_list": pred_key("track_ids"),
    }

    def test_bdd_eval(self) -> None:
        """Testcase for BDD100K evaluation."""
        batch_size = 1

        data_root = osp.join(get_test_data("bdd100k_test"), "track/images")
        annotations = osp.join(get_test_data("bdd100k_test"), "track/labels")
        config = osp.join(get_test_data("bdd100k_test"), "track/config.toml")

        scalabel_eval = BDD100KTrackEvaluator(annotation_path=annotations)
        assert str(scalabel_eval) == "BDD100K Tracking Evaluator"
        assert scalabel_eval.metrics == ["track"]

        # test gt
        dataset = BDD100K(
            data_root=data_root,
            annotation_path=annotations,
            config_path=config,
        )
        test_loader = get_dataloader(dataset, batch_size)

        boxes, scores, classes, track_ids = generate_boxes(
            720, 1280, 4, batch_size, True
        )
        output = {
            "boxes": boxes,
            "scores": scores,
            "class_ids": classes,
            "track_ids": track_ids,
        }

        for batch in test_loader:
            scalabel_eval.process_batch(
                **get_inputs_for_pred_and_data(
                    self.CONN_BDD100K_EVAL, output, batch
                )
            )

        _, log_str = scalabel_eval.evaluate("track")
        assert log_str.count("\n") == 18
