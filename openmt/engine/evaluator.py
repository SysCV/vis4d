"""Evaluation components for tracking."""

import datetime
import itertools
import logging
import os
import time
from contextlib import ExitStack, contextmanager
from multiprocessing import cpu_count
from typing import Callable, Dict, Generator, List, Optional

import detectron2.utils.comm as comm
import torch
from bdd100k.common.utils import DEFAULT_COCO_CONFIG
from bdd100k.eval.detect import evaluate_det
from bdd100k.eval.mot import acc_single_video_mot, evaluate_track
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from scalabel.label.io import group_and_sort, load, save
from scalabel.label.typing import Frame

from openmt.struct import Boxes2D, EvalResult, EvalResults, InputSample

_eval_mapping = dict(
    detect=lambda pred, gt: evaluate_det(gt, pred, DEFAULT_COCO_CONFIG),
    track=lambda pred, gt: evaluate_track(
        acc_single_video_mot, group_and_sort(gt), group_and_sort(pred)
    ),
)  # type: Dict[str, Callable[[List[Frame], List[Frame]], EvalResult]]


@contextmanager
def inference_context(model: torch.nn.Module) -> Generator[None, None, None]:
    """Context for inference.

    The detect is temporarily changed to eval mode and
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
    """Run detect on the data_loader and evaluate the metrics with evaluator.

    Also benchmark the inference speed of `detect.__call__` accurately.
    The detect will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a detect in `training` mode instead, you
            wrap the given detect and override its behavior of `.eval()` and
            `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the detect.
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
            if idx >= num_warmup or seconds_per_img > 5:
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
    return results  # type: ignore


class ScalabelEvaluator(DatasetEvaluator):  # type: ignore
    """Evaluate tracking detect using MOTA metrics.

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
        self._metrics = list(_eval_mapping.keys())
        self._distributed = distributed
        self._output_dir = output_dir
        self._metadata = MetadataCatalog.get(dataset_name)
        self.gts = load(
            self._metadata.json_path, nprocs=cpu_count() // get_world_size()
        )
        self._predictions = []  # type: List[Frame]

    def reset(self) -> None:
        """Preparation for a new round of evaluation."""
        self._predictions = []

    def set_metrics(self, metrics: List[str]) -> None:
        """Set metrics to evaluate."""
        for metric in metrics:
            if metric not in _eval_mapping.keys():  # pragma: no cover
                raise KeyError(f"metric {metric} is not supported")
        self._metrics = metrics

    def process(
        self, inputs: List[List[InputSample]], outputs: List[Boxes2D]
    ) -> None:
        """Process the pair of inputs and outputs."""
        for inp, out in zip(inputs, outputs):
            prediction = inp[0].metadata  # no ref views during test
            prediction.labels = out.to(torch.device("cpu")).to_scalabel(
                self._metadata.idx_to_class_mapping
            )
            self._predictions.append(prediction)

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

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            file_path = os.path.join(self._output_dir, "predictions.json")
            save(file_path, predictions)

        results = {}
        for metric in self._metrics:
            results[metric] = _eval_mapping[metric](predictions, self.gts)

        return results
