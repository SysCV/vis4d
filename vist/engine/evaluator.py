"""Evaluation components for tracking."""
import copy
import datetime
import logging
import os
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from typing import Dict, Generator, List, Optional

import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators
from detectron2.utils import comm
from detectron2.utils.logger import log_every_n_seconds
from scalabel.eval.detect import evaluate_det
from scalabel.eval.mot import acc_single_video_mot, evaluate_track
from scalabel.label.io import group_and_sort, save
from scalabel.label.typing import Config, Frame

from vist.struct import (
    EvalResult,
    EvalResults,
    InputSample,
    LabelInstance,
    ModelOutput,
)

from ..common.utils.parallel import gather_predictions


def _detect(
    pred: List[Frame],
    gt: List[Frame],
    cfg: Config,
    ignore_unknown_cats: bool,  # pylint: disable=unused-argument
) -> EvalResult:
    """Wrapper for evaluate_det function."""
    return evaluate_det(gt, pred, cfg)


def _track(
    pred: List[Frame], gt: List[Frame], cfg: Config, ignore_unknown_cats: bool
) -> EvalResult:
    """Wrapper for evaluate_track function."""
    return evaluate_track(
        acc_single_video_mot,
        group_and_sort(gt),
        group_and_sort(pred),
        cfg,
        ignore_unknown_cats=ignore_unknown_cats,
    )


_eval_mapping = dict(detect=_detect, track=_track)


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
    """Runs model on the data_loader and evaluate the metrics with evaluator.

    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

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
    num_devices = comm.get_world_size()
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
    """Evaluate model using metrics supported in salabel (currently AP / MOTA).

    This class will accumulate information of the inputs/outputs (by
    :meth:`process`), and produce evaluation results in the end (by
    :meth:`evaluate`).
    """

    def __init__(
        self,
        dataset_name: str,
        metrics: List[str],
        distributed: bool = True,
        output_dir: Optional[str] = None,
        ignore_unknown_cats: bool = False,
    ) -> None:
        """Init."""
        self._metrics = list(_eval_mapping.keys())
        self._distributed = distributed
        self._output_dir = output_dir
        self._metadata = MetadataCatalog.get(dataset_name)
        self.gts = DatasetCatalog[dataset_name]()
        self._predictions = defaultdict(list)  # type: Dict[str, List[Frame]]
        self.set_metrics(metrics)
        self.ignore_unknown_cats = ignore_unknown_cats

    def reset(self) -> None:
        """Preparation for a new round of evaluation."""
        self._predictions = defaultdict(list)

    def set_metrics(self, metrics: List[str]) -> None:
        """Set metrics to evaluate."""
        for metric in metrics:
            if metric not in _eval_mapping.keys():  # pragma: no cover
                raise KeyError(f"metric {metric} is not supported")
        self._metrics = metrics

    def process(
        self, inputs: List[List[InputSample]], outputs: ModelOutput
    ) -> None:
        """Process the pair of inputs and outputs."""
        for key, output in outputs.items():
            for inp, out in zip(inputs, output):
                prediction = copy.deepcopy(inp[0].metadata)
                out_cpu = out.to(torch.device("cpu"))
                assert isinstance(out_cpu, LabelInstance)
                prediction.labels = out_cpu.to_scalabel(
                    self._metadata.idx_to_class_mapping
                )
                self._predictions[key].append(prediction)

    def evaluate(self) -> EvalResults:
        """Evaluate the performance after processing all input/output pairs."""
        if self._distributed:
            predictions_dict = gather_predictions(self._predictions)
            if not comm.is_main_process():
                return {}  # pragma: no cover
        else:
            predictions_dict = self._predictions  # pragma: no cover

        results = {}
        for key, predictions in predictions_dict.items():
            if self._output_dir:
                os.makedirs(self._output_dir, exist_ok=True)
                file_path = os.path.join(
                    self._output_dir, f"{key}_predictions.json"
                )
                save(file_path, predictions)

            if key in self._metrics:
                results[key] = _eval_mapping[key](
                    predictions,
                    self.gts,
                    self._metadata.metadata_cfg,
                    self.ignore_unknown_cats,
                )

        return results
