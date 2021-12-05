"""Function for registering the datasets in Vis4D."""
import abc
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel
from pytorch_lightning.utilities.distributed import rank_zero_info
from scalabel.eval.detect import evaluate_det
from scalabel.eval.ins_seg import evaluate_ins_seg
from scalabel.eval.mot import acc_single_video_mot, evaluate_track
from scalabel.eval.mots import acc_single_video_mots, evaluate_seg_track
from scalabel.eval.result import Result
from scalabel.eval.sem_seg import evaluate_sem_seg
from scalabel.label.io import group_and_sort
from scalabel.label.typing import Config, Dataset, Frame

from vis4d.common.registry import RegistryHolder
from vis4d.common.utils.time import Timer
from vis4d.struct import MetricLogs

from ..mapper import SampleMapperConfig
from ..reference import ReferenceSamplerConfig


class BaseDatasetConfig(BaseModel, extra="allow"):
    """Config for training/evaluation datasets."""

    name: str
    type: str
    data_root: str
    sample_mapper: SampleMapperConfig = SampleMapperConfig()
    ref_sampler: ReferenceSamplerConfig = ReferenceSamplerConfig()
    annotations: Optional[str]
    attributes: Optional[
        Dict[str, Union[bool, float, str, List[float], List[str]]]
    ]
    config_path: Optional[str]
    eval_metrics: List[str] = []
    validate_frames: bool = False
    ignore_unkown_cats: bool = False
    cache_as_binary: bool = False
    num_processes: int = 4
    collect_device = "cpu"
    multi_sensor_inference: bool = True
    compute_global_instance_ids: bool = False


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


_eval_mapping = dict(
    detect=_detect,
    track=_track,
    ins_seg=_ins_seg,
    seg_track=_seg_track,
    sem_seg=_sem_seg,
)


class BaseDatasetLoader(metaclass=RegistryHolder):
    """Interface for loading dataset to scalabel format."""

    def __init__(self, cfg: BaseDatasetConfig):
        """Init dataset loader."""
        super().__init__()
        self.cfg = cfg
        self._check_metrics()

        timer = Timer()
        if self.cfg.cache_as_binary:
            assert self.cfg.annotations is not None
            if not os.path.exists(self.cfg.annotations.rstrip("/") + ".pkl"):
                dataset = self.load_dataset()
                with open(
                    self.cfg.annotations.rstrip("/") + ".pkl", "wb"
                ) as file:
                    file.write(pickle.dumps(dataset))
            else:
                with open(
                    self.cfg.annotations.rstrip("/") + ".pkl", "rb"
                ) as file:
                    dataset = pickle.loads(file.read())
        else:
            dataset = self.load_dataset()

        assert dataset.config is not None
        add_data_path(cfg.data_root, dataset.frames)
        rank_zero_info(f"Loading {cfg.name} takes {timer.time():.2f} seconds.")
        self.metadata_cfg = dataset.config
        self.frames = dataset.frames
        self.groups = dataset.groups

    @abc.abstractmethod
    def load_dataset(self) -> Dataset:
        """Load and possibly convert dataset to scalabel format."""
        raise NotImplementedError

    def _check_metrics(self) -> None:
        """Check if evaluation metrics specified are valid."""
        for metric in self.cfg.eval_metrics:
            if metric not in _eval_mapping:
                raise KeyError(
                    f"metric {metric} is not supported in {self.cfg.name}"
                )

    def evaluate(
        self, metric: str, predictions: List[Frame], gts: List[Frame]
    ) -> Tuple[MetricLogs, str]:
        """Convert predictions from scalabel format and evaluate.

        Returns a dictionary of scores to log and a pretty printed string.
        """
        result = _eval_mapping[metric](
            predictions, gts, self.metadata_cfg, self.cfg.ignore_unkown_cats
        )
        log_dict = {f"{metric}/{k}": v for k, v in result.summary().items()}
        return log_dict, str(result)


def build_dataset_loader(cfg: BaseDatasetConfig) -> BaseDatasetLoader:
    """Build a dataset loader."""
    registry = RegistryHolder.get_registry(BaseDatasetLoader)
    if cfg.type in registry:
        dataset_loader = registry[cfg.type](cfg)
        assert isinstance(dataset_loader, BaseDatasetLoader)
        return dataset_loader
    raise NotImplementedError(f"Dataset type {cfg.type} not found.")


def add_data_path(data_root: str, frames: List[Frame]) -> None:
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
