"""DefaultTrainer for openMT."""
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional
from typing import OrderedDict as OrderedDictType

import torch
import pytorch_lightning as pl

from openmt.config import Config
from openmt.data import build_test_loader, build_train_loader
from openmt.model import build_model
from openmt.struct import EvalResults
from openmt.vis import ScalabelVisualizer

from .evaluator import ScalabelEvaluator, inference_on_dataset
from .utils import default_setup, register_directory, to_detectron2


class DefaultTrainer(D2DefaultTrainer):  # type: ignore
    """OpenMT DefaultTrainer class."""

    def __init__(self, cfg: Config, det2cfg: CfgNode):
        """Init."""
        self.openmt_cfg = cfg
        super().__init__(det2cfg)
        # Update hooks with custom parameters / objects
        logp = cfg.solver.log_period
        for hook in self._hooks:
            if logp is not None and isinstance(hook, PeriodicWriter):
                hook._period = logp  # pylint: disable=protected-access

    def build_model(self, cfg: CfgNode) -> torch.nn.Module:
        """Builds model."""
        model = build_model(self.openmt_cfg.model)
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
        sampling = "sequence_based"
        for ds in openmt_cfg.test:
            if ds.name == dataset_name:
                sampling = ds.inference_sampling
        return build_test_loader(
            openmt_cfg.dataloader, cfg, dataset_name, sampling
        )

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
        """Build evaluators."""
        output_folder = os.path.join(cfg.OUTPUT_DIR, dataset_name)
        metrics = [
            ds.eval_metrics
            for ds in openmt_cfg.test
            if ds.name == dataset_name
        ][0]
        evaluator = ScalabelEvaluator(
            dataset_name, metrics, True, output_folder
        )
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


def train(cfg: Config) -> None:
    """Training function."""
    # TODO setup to lightning

    model = build_model(cfg.model)

    # TODO
    # workers_per_gpu: int --> lightning

    dataloader_train = build_train_loader(cfg.train)

    dataloaders_test = [build_test_loader(data_cfg) for data_cfg in cfg.test]

    trainer = pl.Trainer()
    trainer.fit(model)


def test(cfg: Config) -> None:
    """Test function."""

    # TODO setup to lightning

    model = build_model(cfg.model)
    # TODO load weights

    trainer = pl.Trainer()
    trainer.test(model)


def predict(cfg: Config) -> None:
    """Prediction function."""
    # TODO setup to lightning

    model = build_model(cfg.model)

    trainer = pl.Trainer()
    trainer.predict(model)
