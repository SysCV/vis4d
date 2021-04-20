"""Tracking training API."""

import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
from bdd100k.eval.mot import EvalResults
from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.comm import is_main_process

from openmt.config import Config
from openmt.data import build_tracking_test_loader, build_tracking_train_loader
from openmt.detect.config import default_setup, to_detectron2
from openmt.model import build_model

from .checkpointer import TrackingCheckpointer
from .evaluator import ScalabelMOTAEvaluator, inference_on_dataset


class TrackingTrainer(DefaultTrainer):  # type: ignore
    """TrackingTrainer class."""

    def __init__(self, cfg: Config, det2cfg: CfgNode):
        """Init."""
        self.track_cfg = cfg
        super().__init__(det2cfg)
        # Assumes you want to save checkpoints together with logs/statistics
        self.checkpointer = TrackingCheckpointer(
            self._trainer.model,
            cfg.output_dir,
            optimizer=self._trainer.optimizer,
            scheduler=self.scheduler,
        )

    def build_model(self, cfg: CfgNode) -> torch.nn.Module:
        """Builds tracking detect."""
        model = build_model(self.track_cfg.model)
        model.to(torch.device(self.track_cfg.launch.device))
        if hasattr(model, "detector") and hasattr(model.detector, "d2_cfg"):
            cfg.MODEL.merge_from_other_cfg(model.detector.d2_cfg.MODEL)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n%s", model)
        return model

    def build_train_loader(self, cfg: CfgNode) -> torch.utils.data.DataLoader:
        """Calls :func:`openmt.data.build_tracking_train_loader`."""
        return build_tracking_train_loader(self.track_cfg.dataloader, cfg)

    def build_test_loader(
        self, cfg: CfgNode, dataset_name: str
    ) -> torch.utils.data.DataLoader:
        """Calls static version."""
        return self.build_test_loader_static(
            self.track_cfg, cfg, dataset_name
        )  # pragma: no cover # pylint: disable=line-too-long

    @classmethod
    def build_test_loader_static(
        cls, track_cfg: Config, cfg: CfgNode, dataset_name: str
    ) -> torch.utils.data.DataLoader:
        """Calls :func:`openmt.data.build_tracking_test_loader`."""
        return build_tracking_test_loader(
            track_cfg.dataloader, cfg, dataset_name
        )

    @classmethod
    def build_evaluator(
        cls, cfg: CfgNode, dataset_name: str
    ) -> DatasetEvaluator:
        """Build evaluators for tracking and detection."""
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return ScalabelMOTAEvaluator(dataset_name, True, output_folder)

    def test(
        self,
        cfg: CfgNode,
        model: torch.nn.Module,
        evaluators: Optional[List[DatasetEvaluator]] = None,
    ) -> Dict[str, EvalResults]:
        """Calls static test function."""
        return self.test_static(self.track_cfg, cfg, model, evaluators)

    @classmethod
    def test_static(
        cls,
        track_cfg: Config,
        cfg: CfgNode,
        model: torch.nn.Module,
        evaluators: Optional[List[DatasetEvaluator]] = None,
    ) -> Dict[str, EvalResults]:
        """Test detect with given evaluators."""
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]  # pragma: no cover
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(  # pragma: no cover
                evaluators
            ), "{} != {}".format(len(cfg.DATASETS.TEST), len(evaluators))

        results = OrderedDict()  # type: ignore
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader_static(
                track_cfg, cfg, dataset_name
            )
            # When evaluators are passed in as arguments, implicitly assume
            # that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]  # pragma: no cover
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:  # pragma: no cover
                    logger.warning(
                        "No evaluator found. Use `Trainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if is_main_process():
                assert isinstance(results_i, dict), (
                    "Evaluator must return a dict on the main process. Got {} "
                    "instead.".format(results_i)
                )

        if len(results) == 1:
            results = list(results.values())[0]  # type: ignore
        return results


def track_train_func(
    det2cfg: CfgNode, cfg: Config
) -> Optional[Dict[str, Dict[str, float]]]:
    """Training function."""
    trainer = TrackingTrainer(cfg, det2cfg)
    trainer.resume_or_load(resume=cfg.launch.resume)
    return trainer.train()  # type: ignore


def train(cfg: Config) -> None:
    """Launcher for training."""
    detectron2cfg = to_detectron2(cfg)
    default_setup(detectron2cfg, cfg.launch)

    launch(
        track_train_func,
        cfg.launch.num_gpus,
        num_machines=cfg.launch.num_machines,
        machine_rank=cfg.launch.machine_rank,
        dist_url=cfg.launch.dist_url,
        args=(detectron2cfg, cfg),
    )
