"""Tracking training API."""

import logging
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluator,
    DatasetEvaluators,
)
from detectron2.utils.comm import is_main_process

from openmt.config import Config
from openmt.data import build_tracking_train_loader
from openmt.detect.config import default_setup, to_detectron2
from openmt.modeling.meta_arch import build_model

from .checkpointer import TrackingCheckpointer
from .evaluator import MOTAEvaluator, inference_on_dataset


class TrackingTrainer(DefaultTrainer):  # type: ignore
    """Trainer with COCOEvaluator for testing."""

    def __init__(self, cfg: Config, det2cfg: CfgNode):
        self.track_cfg = cfg
        super().__init__(det2cfg)
        # Assumes you want to save checkpoints together with logs/statistics
        self.checkpointer = TrackingCheckpointer(
            self._trainer.model,
            cfg.output_dir,
            optimizer=self._trainer.optimizer,
            scheduler=self.scheduler,
        )

    def build_model(self, cfg: CfgNode):
        """
        Returns:
            torch.nn.Module:
        """
        model = build_model(self.track_cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    def build_train_loader(self, cfg: CfgNode) -> Iterable[List]:
        """It calls :func:`openmt.data.build_tracking_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_tracking_train_loader(self.track_cfg.dataloader, cfg)

    @classmethod
    def build_evaluator(
        cls, cfg: CfgNode, dataset_name: str
    ) -> DatasetEvaluators:
        """Build evaluators for tracking and detection."""
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        track_eval = MOTAEvaluator(dataset_name, True, output_folder)
        det_eval = COCOEvaluator(dataset_name, cfg, True, output_folder)
        return DatasetEvaluators([track_eval, det_eval])

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(
                evaluators
            ), "{} != {}".format(len(cfg.DATASETS.TEST), len(evaluators))

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def train_func(
    det2cfg: CfgNode, cfg: Config
) -> Optional[Dict[str, Dict[str, float]]]:
    """Training function."""
    trainer = TrackingTrainer(cfg, det2cfg)
    trainer.resume_or_load(resume=cfg.launch.resume)
    return trainer.train()  # type: ignore


def train(cfg: Config) -> None:
    """Launcher for training."""

    detectron2cfg = to_detectron2(cfg)  # TODO refactor to d2
    default_setup(detectron2cfg, cfg.launch)

    launch(
        train_func,
        cfg.launch.num_gpus,
        num_machines=cfg.launch.num_machines,
        machine_rank=cfg.launch.machine_rank,
        dist_url=cfg.launch.dist_url,
        args=(detectron2cfg, cfg),
    )
