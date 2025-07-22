"""Scalabel tracking evaluator."""

from __future__ import annotations

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.common.typing import MetricLogs, NDArrayNumber

from .base import ScalabelEvaluator

if SCALABEL_AVAILABLE:
    from scalabel.eval.mot import acc_single_video_mot, evaluate_track
    from scalabel.eval.mots import acc_single_video_mots, evaluate_seg_track
    from scalabel.label.io import group_and_sort
    from scalabel.label.transforms import mask_to_rle, xyxy_to_box2d
    from scalabel.label.typing import Config, Frame, Label
else:
    raise ImportError("scalabel is not installed.")


class ScalabelTrackEvaluator(ScalabelEvaluator):
    """Scalabel 2D tracking evaluation class."""

    METRICS_TRACK = "MOT"
    METRICS_SEG_TRACK = "MOTS"
    METRICS_ALL = "all"

    def __init__(
        self,
        annotation_path: str,
        config: Config | None = None,
        mask_threshold: float = 0.0,
    ) -> None:
        """Initialize the evaluator."""
        super().__init__(annotation_path=annotation_path, config=config)
        self.mask_threshold = mask_threshold

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "Scalabel Tracking Evaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return [self.METRICS_TRACK, self.METRICS_SEG_TRACK]

    def process_batch(
        self,
        frame_ids: list[int],
        sample_names: list[str],
        sequence_names: list[str],
        pred_boxes: list[NDArrayNumber],
        pred_classes: list[NDArrayNumber],
        pred_scores: list[NDArrayNumber],
        pred_track_ids: list[NDArrayNumber],
        pred_masks: list[NDArrayNumber] | None = None,
    ) -> None:
        """Process tracking results."""
        for i, (
            frame_id,
            sample_name,
            sequence_name,
            boxes,
            scores,
            class_ids,
            track_ids,
        ) in enumerate(
            zip(
                frame_ids,
                sample_names,
                sequence_names,
                pred_boxes,
                pred_scores,
                pred_classes,
                pred_track_ids,
            )
        ):
            boxes = array_to_numpy(boxes, n_dims=None, dtype=np.float32)
            class_ids = array_to_numpy(class_ids, n_dims=None, dtype=np.int64)
            scores = array_to_numpy(scores, n_dims=None, dtype=np.float32)
            if pred_masks:
                masks = array_to_numpy(
                    pred_masks[i], n_dims=None, dtype=np.float32
                )

            labels = []
            for label_id, (box, score, class_id, track_id) in enumerate(
                zip(boxes, scores, class_ids, track_ids)
            ):
                box2d = xyxy_to_box2d(*box.tolist())

                if pred_masks:
                    rle = mask_to_rle(
                        (masks[label_id] > self.mask_threshold).astype(
                            np.uint8
                        )
                    )
                else:
                    rle = None

                label = Label(
                    box2d=box2d,
                    category=(
                        self.inverse_cat_map[int(class_id)]
                        if self.inverse_cat_map != {}
                        else str(class_id)
                    ),
                    score=float(score),
                    id=str(int(track_id)),
                    rle=rle,
                )
                labels.append(label)
            frame = Frame(
                name=sample_name,
                videoName=sequence_name,
                frameIndex=frame_id,
                labels=labels,
            )
            self.frames.append(frame)

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the dataset."""
        assert self.config is not None, "config is not set"
        metrics_log: MetricLogs = {}
        short_description = ""

        if metric in [self.METRICS_TRACK, self.METRICS_ALL]:
            results = evaluate_track(
                acc_single_video_mot,
                gts=group_and_sort(self.gt_frames),
                results=group_and_sort(self.frames),
                config=self.config,
                nproc=0,
            )
            for metric_name, metric_value in results.summary().items():
                metrics_log[metric_name] = metric_value
            short_description += str(results) + "\n"

        if metric in [self.METRICS_SEG_TRACK, self.METRICS_ALL]:
            results = evaluate_seg_track(
                acc_single_video_mots,
                gts=group_and_sort(self.gt_frames),
                results=group_and_sort(self.frames),
                config=self.config,
                nproc=0,
            )
            for metric_name, metric_value in results.summary().items():
                metrics_log[metric_name] = metric_value
            short_description += str(results) + "\n"

        return metrics_log, short_description
