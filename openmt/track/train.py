"""Tracking training API."""

import logging
import os
from typing import Dict, Iterable, List, Optional

from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator

from openmt.config import Config
from openmt.data import build_tracking_train_loader
from openmt.detect.config import default_setup, to_detectron2
from openmt.modeling.meta_arch import build_model


class TrackingTrainer(DefaultTrainer):  # type: ignore
    """Trainer with COCOEvaluator for testing."""

    def __init__(self, cfg: Config, det2cfg: CfgNode):
        self.track_cfg = cfg
        super().__init__(det2cfg)

        # TODO needs new checkpointer (load pretrained detection weights into
        #  d2_detector, but save complete model incl tracking params,
        #  resume from complete params). Could also be handled via modifying
        #  loaded weights

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
    ) -> DatasetEvaluator:
        """Build COCOEvaluator."""
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


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
