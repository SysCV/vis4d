"""Vis4D tester."""
from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vis4d.data import DictData
from vis4d.engine.connectors import DataConnector
from vis4d.eval import Evaluator
from vis4d.vis.base import Visualizer

from .util import move_data_to_device


class Tester:
    """Vis4D Tester."""

    def __init__(
        self,
        dataloaders: list[DataLoader[DictData]],
        data_connector: DataConnector,
        evaluators: dict[str, Evaluator] | None = None,
        visualizers: dict[str, Visualizer] | None = None,
        num_epochs: int = -1,
        test_every_nth_epoch: int = 1,
        vis_every_nth_epoch: int = 1,
    ) -> None:
        """Creates an instance of the class."""
        self.num_epochs = num_epochs
        self.test_every_nth_epoch = test_every_nth_epoch
        self.vis_every_nth_epoch = vis_every_nth_epoch

        self.test_dataloader = dataloaders
        self.data_connector = data_connector
        self.evaluators = evaluators if evaluators is not None else {}
        self.visualizers = visualizers if visualizers is not None else {}

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

    @torch.no_grad()  # type: ignore
    def test(
        self, model: nn.Module, metric: str, epoch: None | int = None
    ) -> None:
        """Testing loop."""
        logger = logging.getLogger(__name__)

        model.eval()
        logger.info("Running validation...")
        for test_loader in self.test_dataloader:
            for _, data in enumerate(tqdm(test_loader)):
                # input data
                device = next(model.parameters()).device  # model device
                data = move_data_to_device(data, device)
                test_input = self.data_connector.get_test_input(data)

                # forward
                output = model(**test_input)

                if not epoch or self.do_evaluation(epoch):
                    # TODO, this should be all numpy.
                    for name, test_eval in self.evaluators.items():
                        eval_kwargs = self.data_connector.get_evaluator_input(
                            name, output, data
                        )
                        test_eval.process(
                            **move_data_to_device(  # TODO, maybe
                                # move this to data connector?
                                eval_kwargs,
                                "cpu",
                                True,
                            )
                        )

                if not epoch or self.do_visualization(epoch):
                    for name, vis in self.visualizers.items():
                        eval_kwargs = self.data_connector.get_visualizer_input(
                            name, output, data
                        )
                        vis.process(**eval_kwargs)

        if not epoch or self.do_evaluation(epoch):
            for name, test_eval in self.evaluators.items():

                _, log_str = test_eval.evaluate(metric)
                logger.info(log_str)

        if not epoch or self.do_visualization(epoch):
            for name, test_vis in self.visualizers.items():
                test_vis.save_to_disk(".")
                # test_vis.clear()
