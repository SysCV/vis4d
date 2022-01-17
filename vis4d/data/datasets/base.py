"""Function for registering the datasets in Vis4D."""
import abc
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

from pytorch_lightning.utilities.distributed import rank_zero_info
from scalabel.eval.detect import evaluate_det
from scalabel.eval.ins_seg import evaluate_ins_seg
from scalabel.eval.mot import acc_single_video_mot, evaluate_track
from scalabel.eval.mots import acc_single_video_mots, evaluate_seg_track
from scalabel.eval.pan_seg import evaluate_pan_seg
from scalabel.eval.result import Result
from scalabel.eval.sem_seg import evaluate_sem_seg
from scalabel.label.io import group_and_sort
from scalabel.label.typing import Config, Dataset, Frame, FrameGroup

from vis4d.common.module import Vis4DModule
from vis4d.common.utils.time import Timer
from vis4d.struct import CategoryMap, MetricLogs


def _detect(
    pred: List[Frame],
    gt: List[Frame],
    cfg: Config,
    ignore_unknown_cats: bool,  # pylint: disable=unused-argument
) -> Result:
    """Wrapper for evaluate_det function."""
    return evaluate_det(gt, pred, cfg, nproc=1)


def _ins_seg(
    pred: List[Frame],
    gt: List[Frame],
    cfg: Config,
    ignore_unknown_cats: bool,  # pylint: disable=unused-argument
) -> Result:
    """Wrapper for evaluate_ins_seg function."""
    return evaluate_ins_seg(gt, pred, cfg, nproc=1)


def _track(
    pred: List[Frame], gt: List[Frame], cfg: Config, ignore_unknown_cats: bool
) -> Result:
    """Wrapper for evaluate_track function."""
    return evaluate_track(
        acc_single_video_mot,
        group_and_sort(gt),
        group_and_sort(pred),
        cfg,
        nproc=1,
        ignore_unknown_cats=ignore_unknown_cats,
    )


def _seg_track(
    pred: List[Frame], gt: List[Frame], cfg: Config, ignore_unknown_cats: bool
) -> Result:
    """Wrapper for evaluate_seg_track function."""
    return evaluate_seg_track(
        acc_single_video_mots,
        group_and_sort(gt),
        group_and_sort(pred),
        cfg,
        nproc=1,
        ignore_unknown_cats=ignore_unknown_cats,
    )


def _sem_seg(
    pred: List[Frame],
    gt: List[Frame],
    cfg: Config,
    ignore_unknown_cats: bool,  # pylint: disable=unused-argument
) -> Result:
    """Wrapper for evaluate_sem_seg function."""
    return evaluate_sem_seg(gt, pred, cfg, nproc=1)


def _pan_seg(
    pred: List[Frame],
    gt: List[Frame],
    cfg: Config,
    ignore_unknown_cats: bool,
) -> Result:
    """Wrapper for evaluate_pan_seg function."""
    return evaluate_pan_seg(
        gt, pred, cfg, nproc=1, ignore_unknown_cats=ignore_unknown_cats
    )


_eval_mapping = dict(
    detect=_detect,
    track=_track,
    ins_seg=_ins_seg,
    seg_track=_seg_track,
    sem_seg=_sem_seg,
    pan_seg=_pan_seg,
)


class BaseDatasetLoader(Vis4DModule[None, None]):
    """Interface for loading dataset to Scalabel format."""

    def __init__(
        self,
        name: str,
        data_root: str,
        annotations: Optional[str] = None,
        category_mapping: Optional[CategoryMap] = None,
        attributes: Optional[
            Dict[str, Union[bool, float, str, List[float], List[str]]]
        ] = None,
        config_path: Optional[str] = None,
        eval_metrics: Optional[List[str]] = None,
        ignore_unknown_cats: bool = False,
        cache_as_binary: bool = False,
        num_processes: int = 4,
        collect_device: str = "cpu",
        multi_sensor_inference: bool = True,
        compute_global_instance_ids: bool = False,
    ):
        """Init dataset loader."""
        super().__init__()
        self.name = name
        self.data_root = data_root
        self.annotations = annotations
        self.config_path = config_path
        self.eval_metrics = eval_metrics
        self.ignore_unknown_cats = ignore_unknown_cats
        self.collect_device = collect_device
        self.category_mapping = category_mapping
        self.cache_as_binary = cache_as_binary
        self.compute_global_instance_ids = compute_global_instance_ids
        self.attributes = attributes
        self.num_processes = num_processes
        self.multi_sensor_inference = multi_sensor_inference

        if self.eval_metrics is None:
            self.eval_metrics = []
        self._check_metrics()

        timer = Timer()
        if cache_as_binary:
            dataset = self.load_cached_dataset()
        else:
            dataset = self.load_dataset()

        assert dataset.config is not None
        add_data_path(data_root, dataset.frames)
        if dataset.groups is not None:
            add_data_path(data_root, dataset.groups)
        rank_zero_info(f"Loading {name} takes {timer.time():.2f} seconds.")
        self.metadata_cfg = dataset.config
        self.frames = dataset.frames
        self.groups = dataset.groups

    def load_cached_dataset(self) -> Dataset:
        """Load cached dataset from file."""
        assert self.annotations is not None
        cache_path = self.annotations.rstrip("/") + ".pkl"
        if not os.path.exists(cache_path):
            dataset = self.load_dataset()
            with open(cache_path, "wb") as file:
                file.write(pickle.dumps(dataset))
        else:
            with open(cache_path, "rb") as file:
                dataset = pickle.loads(file.read())
        return dataset

    @abc.abstractmethod
    def load_dataset(self) -> Dataset:
        """Load and possibly convert dataset to Scalabel format."""
        raise NotImplementedError

    def _check_metrics(self) -> None:
        """Check if evaluation metrics specified are valid."""
        assert self.eval_metrics is not None
        for metric in self.eval_metrics:
            if metric not in _eval_mapping:  # pragma: no cover
                raise KeyError(
                    f"metric {metric} is not supported in"
                    f" dataset {self.name}"
                )

    def evaluate(
        self, metric: str, predictions: List[Frame], gts: List[Frame]
    ) -> Tuple[MetricLogs, str]:
        """Convert predictions from Scalabel format and evaluate.

        Returns a dictionary of scores to log and a pretty printed string.
        """
        result = _eval_mapping[metric](
            predictions, gts, self.metadata_cfg, self.ignore_unknown_cats
        )
        log_dict = {f"{metric}/{k}": v for k, v in result.summary().items()}
        return log_dict, str(result)


def add_data_path(
    data_root: str, frames: Union[List[Frame], List[FrameGroup]]
) -> None:
    """Add filepath to frame using data_root."""
    for ann in frames:
        assert ann.name is not None
        if ann.url is None:
            if ann.videoName is not None:
                ann.url = os.path.join(data_root, ann.videoName, ann.name)
            else:
                ann.url = os.path.join(data_root, ann.name)
        else:
            ann.url = os.path.join(data_root, ann.url)
