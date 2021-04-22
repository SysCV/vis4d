"""Detection training API."""
import logging
import os
from typing import Dict, List, Optional, Union

import torch
from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator as D2COCOEvaluator
from detectron2.evaluation import DatasetEvaluator, coco_evaluation
from detectron2.structures import Boxes, Instances

from openmt.config import Config
from openmt.model import build_model
from openmt.struct import Boxes2D

from .config import default_setup, to_detectron2


class COCOEvaluator(D2COCOEvaluator):  # type: ignore
    """Detectron2 COCO Evaluation adapted to openMT output format."""

    def process(
        self,
        inputs: List[Dict[str, Union[torch.Tensor, int]]],
        outputs: List[Boxes2D],
    ) -> None:
        """D2COCOEvaluator override to be compatible with openMT out format."""
        for inp, out in zip(inputs, outputs):
            prediction = {"image_id": inp["image_id"]}
            boxes2d = out.to(self._cpu_device)
            fields = dict(
                pred_boxes=Boxes(boxes2d.boxes[:, :4]),
                scores=boxes2d.boxes[:, -1],
                pred_classes=boxes2d.class_ids,
            )
            instances = Instances((inp["height"], inp["width"]), **fields)
            prediction["instances"] = coco_evaluation.instances_to_coco_json(
                instances, inp["image_id"]
            )
            self._predictions.append(prediction)


class Trainer(DefaultTrainer):  # type: ignore
    """Trainer with COCOEvaluator for testing."""

    def __init__(self, cfg: Config, det2cfg: CfgNode):
        """Init."""
        self.track_cfg = cfg
        super().__init__(det2cfg)

    def build_model(self, cfg: CfgNode) -> torch.nn.Module:
        """Builds tracking detect."""
        model = build_model(self.track_cfg.model)
        assert hasattr(model, "detector")
        if hasattr(model, "detector") and hasattr(model.detector, "d2_cfg"):
            cfg.MODEL.merge_from_other_cfg(model.detector.d2_cfg.MODEL)
        model.to(torch.device(self.track_cfg.launch.device))
        logger = logging.getLogger(__name__)
        logger.info("Model:\n%s", model)
        return model

    @classmethod
    def build_evaluator(
        cls, cfg: CfgNode, dataset_name: str
    ) -> DatasetEvaluator:
        """Build COCOEvaluator."""
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def train(cfg: Config) -> Optional[Dict[str, Dict[str, float]]]:
    """Training function."""
    det2cfg = to_detectron2(cfg)
    default_setup(det2cfg, cfg.launch)
    trainer = Trainer(cfg, det2cfg)
    if cfg.launch.weights != "detectron2":
        trainer.cfg.MODEL.WEIGHTS = cfg.launch.weights
    trainer.resume_or_load(resume=cfg.launch.resume)
    return trainer.train()  # type: ignore
