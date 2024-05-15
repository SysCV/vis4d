"""BDD100K tracking evaluator."""

from __future__ import annotations

from vis4d.common.imports import BDD100K_AVAILABLE, SCALABEL_AVAILABLE
from vis4d.common.typing import MetricLogs
from vis4d.data.datasets.bdd100k import bdd100k_track_map

from ..scalabel.track import ScalabelTrackEvaluator

if SCALABEL_AVAILABLE and BDD100K_AVAILABLE:
    from bdd100k.common.utils import load_bdd100k_config
    from bdd100k.label.to_scalabel import bdd100k_to_scalabel
    from scalabel.eval.detect import evaluate_det
    from scalabel.eval.mot import acc_single_video_mot, evaluate_track
    from scalabel.label.io import group_and_sort
else:
    raise ImportError("scalabel or bdd100k is not installed.")


class BDD100KTrackEvaluator(ScalabelTrackEvaluator):
    """BDD100K 2D tracking evaluation class."""

    METRICS_DET = "Det"
    METRICS_TRACK = "Track"

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
        return [self.METRICS_DET, self.METRICS_TRACK]

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the dataset."""
        assert self.config is not None, "BDD100K config is not loaded."
        metrics_log: MetricLogs = {}
        short_description = ""

        if metric == self.METRICS_DET:
            det_results = evaluate_det(
                self.gt_frames,
                self.frames,
                config=self.config,
                nproc=0,
            )
            for metric_name, metric_value in det_results.summary().items():
                metrics_log[metric_name] = metric_value
            short_description += str(det_results) + "\n"

        if metric == self.METRICS_TRACK:
            track_results = evaluate_track(
                acc_single_video_mot,
                gts=group_and_sort(self.gt_frames),
                results=group_and_sort(self.frames),
                config=self.config,
                nproc=1,
            )
            for metric_name, metric_value in track_results.summary().items():
                metrics_log[metric_name] = metric_value
            short_description += str(track_results) + "\n"

        return metrics_log, short_description
