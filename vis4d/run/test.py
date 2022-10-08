"""Vis4D tester."""
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vis4d.eval import Evaluator

from .util import move_data_to_device


@torch.no_grad()
def testing_loop(
    test_dataloader: List[DataLoader],
    evaluators: List[Evaluator],
    metric: str,
    model: nn.Module,
    model_test_keys: List[str],
) -> None:
    """Testing loop."""
    model.eval()
    print("Running validation...")
    for test_loader in test_dataloader:
        for _, data in enumerate(tqdm(test_loader)):
            # input data
            device = next(model.parameters()).device  # model device
            data = move_data_to_device(data, device)
            test_input = (data[key] for key in model_test_keys)

            # forward
            output = model.forward_test(*test_input)

            for test_eval in evaluators:
                test_eval.process(data, output)

    for test_eval in evaluators:
        _, log_str = test_eval.evaluate(metric)
        print(log_str)
