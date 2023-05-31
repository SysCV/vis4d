"""Testcases for BDD100K segmentation evaluator."""
from __future__ import annotations

import os.path as osp
import unittest

from torch.utils.data import DataLoader, Dataset

from tests.util import generate_semantic_masks, get_test_data
from vis4d.data.datasets.bdd100k import BDD100K
from vis4d.data.loader import DataPipe, build_inference_dataloaders
from vis4d.engine.connectors import (
    data_key,
    get_inputs_for_pred_and_data,
    pred_key,
)
from vis4d.eval.bdd100k import BDD100KSegEvaluator


def get_dataloader(datasets: Dataset, batch_size: int) -> DataLoader:
    """Get data loader for testing."""
    datapipe = DataPipe(datasets)
    return build_inference_dataloaders(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=0
    )[0]


class TestBDD100KSegEvaluator(unittest.TestCase):
    """BDD100K segmentation evaluator testcase class."""

    CONN_BDD100K_EVAL = {
        "data_names": data_key("name"),
        "masks_list": pred_key("masks"),
    }

    def test_bdd_eval(self) -> None:
        """Testcase for BDD100K evaluation."""
        batch_size = 1

        data_root = osp.join(get_test_data("bdd100k_test"), "segment/images")
        annotations = osp.join(
            get_test_data("bdd100k_test"), "segment/labels/annotation.json"
        )

        scalabel_eval = BDD100KSegEvaluator(annotation_path=annotations)
        assert str(scalabel_eval) == "BDD100K Segmentation Evaluator"
        assert scalabel_eval.metrics == ["sem_seg"]

        # test gt
        dataset = BDD100K(
            data_root=data_root,
            annotation_path=annotations,
            config_path="sem_seg",
        )
        test_loader = get_dataloader(dataset, batch_size)

        masks = generate_semantic_masks(720, 1280, 19, batch_size)
        output = {"masks": masks}

        for batch in test_loader:
            scalabel_eval.process_batch(
                **get_inputs_for_pred_and_data(
                    self.CONN_BDD100K_EVAL, output, batch
                )
            )

        _, log_str = scalabel_eval.evaluate("sem_seg")
        assert log_str.count("\n") == 24
