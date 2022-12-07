"""Vis4D tester."""
from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vis4d.common import DictStrAny
from vis4d.common.distributed import get_rank
from vis4d.data import DictData
from vis4d.eval import Evaluator
from vis4d.vis.base import Visualizer

from .util import move_data_to_device


class Tester:
    """Vis4D Tester."""

    def __init__(
        self,
        num_epochs: int = -1,
        test_every_nth_epoch: int = 1,
        vis_every_nth_epoch: int = 1,
    ) -> None:
        """Init."""
        self.num_epochs = num_epochs
        self.test_every_nth_epoch = test_every_nth_epoch
        self.vis_every_nth_epoch = vis_every_nth_epoch

        self.test_dataloader = self.setup_test_dataloaders()
        self.evaluators = self.setup_evaluators()
        self.visualizers = self.setup_visualizers()

    def setup_test_dataloaders(self) -> list[DataLoader]:
        """Set-up testing data loaders."""
        raise NotImplementedError

    def test_connector(self, data: DictData) -> DictData:
        """Connector between the test data and the model."""
        return data

    def setup_evaluators(self) -> list[Evaluator]:
        """Set-up evaluators."""
        raise NotImplementedError

    def evaluator_connector(
        self, data: DictData, output: DictStrAny
    ) -> DictStrAny:
        """Connector between the data and the evaluator."""
        # For now just wrap data connector to not break anything.
        return data

    def do_evaluation(self, epoch: int) -> bool:
        """Return whether to do evaluation for current epoch."""
        return (
            epoch == self.num_epochs - 1
            or epoch % self.test_every_nth_epoch
            == self.test_every_nth_epoch - 1
        )

    def setup_visualizers(self) -> list[Visualizer]:
        """Set-up visualizers."""
        raise NotImplementedError

    def do_visualization(self, epoch: int) -> bool:
        """Return whether to do visualization for current epoch."""
        return (
            epoch == self.num_epochs - 1
            or epoch % self.vis_every_nth_epoch == self.vis_every_nth_epoch - 1
        )

    @torch.no_grad()
    def test(self, model: nn.Module, metric: str, epoch: int) -> None:
        """Testing loop."""
        logger = logging.getLogger(__name__)

        model.eval()
        logger.info("Running validation...")
        for test_loader in self.test_dataloader:
            for _, data in enumerate(tqdm(test_loader)):
                # input data
                device = next(model.parameters()).device  # model device
                data = move_data_to_device(data, device)
                test_input = self.test_connector(data)

                # forward
                output = model(**test_input)

                if self.do_evaluation(epoch):
                    for test_eval in self.evaluators:
                        evaluator_kwargs = self.evaluator_connector(
                            data, output
                        )
                        test_eval.process(
                            *[
                                v.detach().cpu().numpy()
                                for k, v in evaluator_kwargs.items()
                            ]
                        )

                if self.do_visualization(epoch):
                    for vis in self.visualizers:
                        vis.process(data, output)

        if self.do_evaluation(epoch):
            for test_eval in self.evaluators:
                _, log_str = test_eval.evaluate(metric)
                logger.info(log_str)

        if self.do_visualization(epoch):
            for test_vis in self.visualizers:
                test_vis.visualize()
                # test_vis.clear()
