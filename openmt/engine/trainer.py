"""DefaultTrainer for openMT."""
import logging
import os
import weakref
from collections import OrderedDict
from typing import Dict, List, Optional
from typing import OrderedDict as OrderedDictType

import torch
from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer as D2DefaultTrainer
from detectron2.engine import PeriodicCheckpointer, PeriodicWriter
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.comm import is_main_process

from openmt.config import Config
from openmt.data import build_test_loader, build_train_loader
from openmt.model import build_model
from openmt.struct import EvalResults
from openmt.vis import ScalabelVisualizer

from .checkpointer import Checkpointer
from .evaluator import ScalabelEvaluator, inference_on_dataset
from .utils import default_setup, register_directory, to_detectron2


class DefaultTrainer(D2DefaultTrainer):  # type: ignore
    """OpenMT DefaultTrainer class."""

    def __init__(self, cfg: Config, det2cfg: CfgNode):
        """Init."""
        self.openmt_cfg = cfg
        super().__init__(det2cfg)
        # Assumes you want to save checkpoints together with logs/statistics
        self.checkpointer = Checkpointer(
            self._trainer.model,
            cfg.launch.output_dir,
            trainer=weakref.proxy(self),
        )
        # Update hooks with custom parameters / objects
        logp = cfg.solver.log_period
        for hook in self._hooks:
            if isinstance(hook, PeriodicCheckpointer):
                hook.checkpointer = self.checkpointer
            if logp is not None and isinstance(hook, PeriodicWriter):
                hook._period = logp  # pylint: disable=protected-access

    def build_model(self, cfg: CfgNode) -> torch.nn.Module:
        """Builds tracking detect."""
        model = build_model(self.openmt_cfg.model)
        assert hasattr(model, "detector")
        if hasattr(model, "detector") and hasattr(model.detector, "d2_cfg"):
            cfg.MODEL.merge_from_other_cfg(model.detector.d2_cfg.MODEL)
        model.to(torch.device(self.openmt_cfg.launch.device))
        logger = logging.getLogger(__name__)
        logger.info("Model:\n%s", model)
        return model

    def build_train_loader(self, cfg: CfgNode) -> torch.utils.data.DataLoader:
        """Calls :func:`openmt.data.build_train_loader`."""
        return build_train_loader(self.openmt_cfg.dataloader, cfg)

    def build_test_loader(
        self, cfg: CfgNode, dataset_name: str
    ) -> torch.utils.data.DataLoader:
        """Calls static version."""
        return self.build_test_loader_static(
            self.openmt_cfg, cfg, dataset_name
        )  # pragma: no cover

    @classmethod
    def build_test_loader_static(
        cls, openmt_cfg: Config, cfg: CfgNode, dataset_name: str
    ) -> torch.utils.data.DataLoader:
        """Calls :func:`openmt.data.build_test_loader`."""
        return build_test_loader(openmt_cfg.dataloader, cfg, dataset_name)

    def build_evaluator(
        self, cfg: CfgNode, dataset_name: str
    ) -> DatasetEvaluator:
        """Build Scalabel evaluators."""
        return self.build_evaluator_static(
            self.openmt_cfg, cfg, dataset_name
        )  # pragma: no cover

    @classmethod
    def build_evaluator_static(
        cls, openmt_cfg: Config, cfg: CfgNode, dataset_name: str
    ) -> DatasetEvaluator:
        """Build evaluators for tracking and detection."""
        output_folder = os.path.join(cfg.OUTPUT_DIR, dataset_name)
        evaluator = ScalabelEvaluator(dataset_name, True, output_folder)
        evaluator.set_metrics(openmt_cfg.solver.eval_metrics)
        return evaluator

    def train(self) -> Dict[str, EvalResults]:
        """Run training."""
        super().train()
        return self._last_eval_results  # type: ignore

    def test(
        self,
        cfg: CfgNode,
        model: torch.nn.Module,
        evaluators: Optional[List[DatasetEvaluator]] = None,
    ) -> Dict[str, EvalResults]:
        """Calls static test function."""
        return self.test_static(self.openmt_cfg, cfg, model, evaluators)

    @classmethod
    def test_static(
        cls,
        openmt_cfg: Config,
        cfg: CfgNode,
        model: torch.nn.Module,
        evaluators: Optional[List[DatasetEvaluator]] = None,
    ) -> OrderedDictType[str, EvalResults]:
        """Test model with given evaluators."""
        logger = logging.getLogger(__name__)
        assert openmt_cfg.test is not None
        datasets = [ds.name for ds in openmt_cfg.test]
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]  # pragma: no cover
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(  # pragma: no cover
                evaluators
            ), "{} != {}".format(len(datasets), len(evaluators))

        results = OrderedDict()  # type: ignore
        for idx, dataset_name in enumerate(datasets):
            data_loader = cls.build_test_loader_static(
                openmt_cfg, cfg, dataset_name
            )
            # When evaluators are passed in as arguments, implicitly assume
            # that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]  # pragma: no cover
            else:
                try:
                    evaluator = cls.build_evaluator_static(
                        openmt_cfg, cfg, dataset_name
                    )
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

    @classmethod
    def predict(
        cls,
        openmt_cfg: Config,
        cfg: CfgNode,
        model: torch.nn.Module,
    ) -> None:
        """Test detect with given evaluators."""
        assert openmt_cfg.launch.output_dir is not None
        if openmt_cfg.launch.input_dir is not None:
            datasets = [register_directory(openmt_cfg.launch.input_dir)]
        else:
            assert openmt_cfg.test is not None
            datasets = [ds.name for ds in openmt_cfg.test]

        for dataset_name in datasets:
            data_loader = cls.build_test_loader_static(
                openmt_cfg, cfg, dataset_name
            )
            output_folder = os.path.join(
                openmt_cfg.launch.output_dir, dataset_name
            )
            visualizer = ScalabelVisualizer(
                dataset_name, output_folder, True, openmt_cfg.launch.visualize
            )
            inference_on_dataset(model, data_loader, visualizer)


def train(cfg: Config) -> Dict[str, EvalResults]:
    """Training function."""
    det2cfg = to_detectron2(cfg)
    default_setup(cfg, det2cfg, cfg.launch)

    trainer = DefaultTrainer(cfg, det2cfg)
    if cfg.launch.weights != "detectron2":
        trainer.cfg.MODEL.WEIGHTS = cfg.launch.weights  # pragma: no cover
    trainer.resume_or_load(resume=cfg.launch.resume)
    return trainer.train()


def test(cfg: Config) -> Dict[str, EvalResults]:
    """Test function."""
    det2cfg = to_detectron2(cfg)
    default_setup(cfg, det2cfg, cfg.launch)

    model = build_model(cfg.model)
    model.to(torch.device(cfg.launch.device))
    if hasattr(model, "detector") and hasattr(model.detector, "d2_cfg"):
        det2cfg.MODEL.merge_from_other_cfg(model.detector.d2_cfg.MODEL)
    if cfg.launch.weights != "detectron2":
        det2cfg.MODEL.WEIGHTS = cfg.launch.weights  # pragma: no cover
    Checkpointer(model, save_dir=det2cfg.OUTPUT_DIR).resume_or_load(
        det2cfg.MODEL.WEIGHTS, resume=cfg.launch.resume
    )
    return DefaultTrainer.test_static(cfg, det2cfg, model)


def predict(cfg: Config) -> None:
    """Prediction function."""
    det2cfg = to_detectron2(cfg)
    default_setup(cfg, det2cfg, cfg.launch)

    model = build_model(cfg.model)
    model.to(torch.device(cfg.launch.device))
    if hasattr(model, "detector") and hasattr(model.detector, "d2_cfg"):
        det2cfg.MODEL.merge_from_other_cfg(model.detector.d2_cfg.MODEL)
    if cfg.launch.weights != "detectron2":
        det2cfg.MODEL.WEIGHTS = cfg.launch.weights  # pragma: no cover
    Checkpointer(model, save_dir=det2cfg.OUTPUT_DIR).resume_or_load(
        det2cfg.MODEL.WEIGHTS, resume=cfg.launch.resume
    )
    DefaultTrainer.predict(cfg, det2cfg, model)
