"""BDD100K tracking evaluator."""
from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.imports import BDD100K_AVAILABLE, SCALABEL_AVAILABLE
from vis4d.common.typing import ArrayLike, MetricLogs
from vis4d.data.datasets.bdd100k import bdd100k_track_map

from ..base import Evaluator

if SCALABEL_AVAILABLE and BDD100K_AVAILABLE:
    from bdd100k.common.utils import load_bdd100k_config
    from bdd100k.label.to_scalabel import bdd100k_to_scalabel
    from scalabel.eval.detect import evaluate_det
    from scalabel.eval.mot import acc_single_video_mot, evaluate_track
    from scalabel.label.io import group_and_sort, load
    from scalabel.label.transforms import xyxy_to_box2d
    from scalabel.label.typing import Frame, Label


class BDD100KTrackEvaluator(Evaluator):
    """BDD100K 2D tracking evaluation class."""

    METRICS_DET = "Det"
    METRICS_TRACK = "Track"
    inverse_track_map = {v: k for k, v in bdd100k_track_map.items()}

    def __init__(self, annotation_path: str) -> None:
        """Initialize the evaluator."""
        super().__init__()
        self.annotation_path = annotation_path
        self.frames: list[Frame] = []

        bdd100k_anns = load(self.annotation_path)
        frames = bdd100k_anns.frames
        self.config = load_bdd100k_config("box_track")
        self.gt_frames = bdd100k_to_scalabel(frames, self.config)

        self.reset()

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "BDD100K Tracking Evaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return [self.METRICS_DET, self.METRICS_TRACK]

    def gather(  # type: ignore # pragma: no cover
        self, gather_func: Callable[[Any], Any]
    ) -> None:
        """Gather variables in case of distributed setting (if needed).

        Args:
            gather_func (Callable[[Any], Any]): Gather function.
        """
        all_preds = gather_func(self.frames)
        if all_preds is not None:
            self.frames = list(itertools.chain(*all_preds))

    def reset(self) -> None:
        """Reset the evaluator."""
        self.frames = []

    def process_batch(  # type: ignore # pylint: disable=arguments-differ
        self,
        frame_ids: list[int],
        sample_names: list[str],
        sequence_names: list[str],
        boxes_list: list[ArrayLike],
        class_ids_list: list[ArrayLike],
        scores_list: list[ArrayLike],
        track_ids_list: list[ArrayLike],
    ) -> None:
        """Process tracking results."""
        for (
            frame_id,
            data_name,
            video_name,
            boxes,
            scores,
            class_ids,
            track_ids,
        ) in zip(
            frame_ids,
            sample_names,
            sequence_names,
            boxes_list,
            scores_list,
            class_ids_list,
            track_ids_list,
        ):
            boxes = array_to_numpy(boxes, n_dims=None, dtype=np.float32)
            class_ids = array_to_numpy(class_ids, n_dims=None, dtype=np.int64)
            scores = array_to_numpy(scores, n_dims=None, dtype=np.float32)
            track_ids = array_to_numpy(track_ids, n_dims=None, dtype=np.int64)
            labels = []
            for box, score, class_id, track_id in zip(
                boxes, scores, class_ids, track_ids
            ):
                box2d = xyxy_to_box2d(*box.tolist())
                label = Label(
                    id=str(int(track_id)),
                    box2d=box2d,
                    category=self.inverse_track_map[int(class_id)],
                    score=float(score),
                )
                labels.append(label)
            frame = Frame(
                name=data_name,
                videoName=video_name,
                frameIndex=frame_id,
                labels=labels,
            )
            self.frames.append(frame)

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the dataset."""
        assert self.config is not None, "BDD100K config is not loaded."
        metrics_log: MetricLogs = {}
        short_description = ""

        if metric == self.METRICS_DET:
            det_results = evaluate_det(
                self.gt_frames,
                self.frames,
                config=self.config.scalabel,
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
                config=self.config.scalabel,
                nproc=0,
            )
            for metric_name, metric_value in track_results.summary().items():
                metrics_log[metric_name] = metric_value
            short_description += str(track_results) + "\n"

        return metrics_log, short_description
