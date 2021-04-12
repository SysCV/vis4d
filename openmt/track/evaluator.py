"""Evaluation components for tracking."""

import datetime
import itertools
import logging
import os.path as osp
import time
from contextlib import ExitStack, contextmanager
from typing import Dict, Generator, List, Optional, Tuple

import detectron2.utils.comm as comm
import torch
from bdd100k.common.utils import group_and_sort
from bdd100k.eval.mot import EvalResults, acc_single_video_mot, evaluate_track
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from scalabel.label.typing import Frame

from openmt.data.datasets.scalabel_video import load_json
from openmt.struct import Boxes2D


@contextmanager
def inference_context(model: torch.nn.Module) -> Generator[None, None, None]:
    """Context for inference.

    The model is temporarily changed to eval mode and
    restored to previous mode afterwards.
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def inference_on_dataset(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    evaluator: DatasetEvaluator,
) -> EvalResults:
    """Run model on the data_loader and evaluate the metrics with evaluator.

    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and
            `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if
        you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        EvalResults: The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on %s images", len(data_loader))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])  # pragma: no cover
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0.0
    with ExitStack() as stack:
        if isinstance(model, torch.nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # pragma: no cover
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (
                    time.perf_counter() - start_time
                ) / iters_after_start
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization
    # barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, "
        "on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(
        datetime.timedelta(seconds=int(total_compute_time))
    )
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, "
        "on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream
    # code to handle
    if results is None:
        results = {}  # pragma: no cover
    return results


class ScalabelMOTAEvaluator(DatasetEvaluator):  # type: ignore
    """Evaluate tracking model using MOTA metrics.

    This class will accumulate information of the inputs/outputs (by
    :meth:`process`), and produce evaluation results in the end (by
    :meth:`evaluate`).
    """

    def __init__(
        self,
        dataset_name: str,
        distributed: bool = True,
        output_dir: Optional[str] = None,
    ) -> None:
        """Init."""
        self._distributed = distributed
        self._output_dir = output_dir
        self._metadata = MetadataCatalog.get(dataset_name)

        self.gts = load_json(
            self._metadata.json_path, self._metadata.image_root
        )
        self._predictions = []  # type: List[Frame]

    def reset(self) -> None:
        """Preparation for a new round of evaluation."""
        self._predictions = []

    def process(
        self, inputs: Tuple[Dict[str, torch.Tensor]], outputs: List[Boxes2D]
    ) -> None:
        """Process the pair of inputs and outputs.

        If they contain batches, the pairs can be consumed one-by-one using
        `zip`:
        .. code-block:: python
            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for inp, out in zip(inputs, outputs):
            prediction = dict(
                name=osp.basename(inp["file_name"]),
                video_name=inp["video_id"],
                frame_index=inp["frame_id"],
            )

            prediction["labels"] = out.to(torch.device("cpu")).to_scalabel(
                self._metadata.idx_to_class_mapping
            )

            self._predictions.append(Frame(**prediction))

    def evaluate(self) -> EvalResults:
        """Evaluate the performance after processing all input/output pairs."""
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}  # pragma: no cover
        else:
            predictions = self._predictions  # pragma: no cover

        return evaluate_track(
            acc_single_video_mot,
            group_and_sort(predictions),
            group_and_sort(self.gts),
        )
