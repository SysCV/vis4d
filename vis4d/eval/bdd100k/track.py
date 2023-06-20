"""BDD100K tracking evaluator."""
from __future__ import annotations

from vis4d.common.imports import BDD100K_AVAILABLE, SCALABEL_AVAILABLE
from vis4d.common.typing import MetricLogs
from vis4d.data.datasets.bdd100k import bdd100k_track_map

from ..scalabel.track import ScalabelTrackEvaluator

if SCALABEL_AVAILABLE and BDD100K_AVAILABLE:
    from bdd100k.common.utils import load_bdd100k_config
    from bdd100k.label.to_scalabel import bdd100k_to_scalabel
    from scalabel.eval.mot import acc_single_video_mot, evaluate_track
    from scalabel.label.io import group_and_sort


class BDD100KTrackEvaluator(ScalabelTrackEvaluator):
    """BDD100K 2D tracking evaluation class."""

    METRICS_TRACK = "MOT"

    def __init__(
        self,
        annotation_path: str,
        config_path: str = "box_track",
        mask_threshold: float = 0.0,
    ) -> None:
        """Initialize the evaluator."""
        config = load_bdd100k_config(config_path)
        super().__init__(
            annotation_path=annotation_path,
            config=config.scalabel,
            mask_threshold=mask_threshold,
        )
        self.gt_frames = bdd100k_to_scalabel(self.gt_frames, config)
        self.inverse_cat_map = {v: k for k, v in bdd100k_track_map.items()}

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "BDD100K Tracking Evaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return [self.METRICS_TRACK]

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the dataset."""
        assert self.config is not None, "config is not set"

        if metric == self.METRICS_TRACK:
            results = evaluate_track(
                acc_single_video_mot,
                gts=group_and_sort(self.gt_frames),
                results=group_and_sort(self.frames),
                config=self.config,
                nproc=1,
            )
        else:
            raise NotImplementedError

        log_dict = {f"{k}": float(v) for k, v in results.summary().items()}

        return log_dict, str(results)  # type: ignore
