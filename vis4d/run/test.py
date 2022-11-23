"""Vis4D tester."""
from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vis4d.eval import Evaluator
from vis4d.vis.base import Visualizer

from .util import move_data_to_device


@torch.no_grad()
def testing_loop(
    test_dataloader: list[DataLoader],
    evaluators: list[Evaluator],
    metric: str,
    model: nn.Module,
    data_connector,
    eval_connector,  # TODO, discuss
    visualizers: None | list[Visualizer] = None,
) -> None:
    """Testing loop."""
    if visualizers is None:
        visualizers = []
    logger = logging.getLogger(__name__)

    model.eval()
    logger.info("Running validation...")
    for test_loader in test_dataloader:
        for _, data in enumerate(tqdm(test_loader)):
            # input data
            device = next(model.parameters()).device  # model device
            data = move_data_to_device(data, device)
            test_input = data_connector("test", data)

            # forward
            output = model(**test_input)

            for test_eval in evaluators:
                evaluator_kwargs = eval_connector("eval", data, output)
                test_eval.process(
                    *[
                        v.detach().cpu().numpy()
                        for k, v in evaluator_kwargs.items()
                    ]
                )
            for vis in visualizers:
                vis.process(data, output)
    for test_eval in evaluators:
        _, log_str = test_eval.evaluate(metric)
        logger.info(log_str)

    for test_vis in visualizers:
        test_vis.visualize()
        # test_vis.clear()
