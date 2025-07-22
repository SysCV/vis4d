"""Scalabel detection evaluator."""

from __future__ import annotations

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.common.typing import ArrayLike, MetricLogs

from .base import ScalabelEvaluator

if SCALABEL_AVAILABLE:
    from scalabel.eval.detect import evaluate_det
    from scalabel.eval.ins_seg import evaluate_ins_seg
    from scalabel.label.transforms import mask_to_rle, xyxy_to_box2d
    from scalabel.label.typing import Config, Frame, Label
else:
    raise ImportError("scalabel is not installed.")


class ScalabelDetectEvaluator(ScalabelEvaluator):
    """Scalabel 2D detection evaluation class."""

    METRICS_DET = "Det"
    METRICS_INS_SEG = "InsSeg"

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
        return "Scalabel Detection Evaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return [self.METRICS_DET, self.METRICS_INS_SEG]

    def process_batch(
        self,
        frame_ids: list[int],
        sample_names: list[str],
        sequence_names: list[str],
        pred_boxes: list[ArrayLike],
        pred_classes: list[ArrayLike],
        pred_scores: list[ArrayLike],
        pred_masks: list[ArrayLike] | None = None,
    ) -> None:
        """Process tracking results."""
        for i, (
            frame_id,
            sample_name,
            sequence_name,
            boxes,
            class_ids,
            scores,
        ) in enumerate(
            zip(
                frame_ids,
                sample_names,
                sequence_names,
                pred_boxes,
                pred_classes,
                pred_scores,
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
            for label_id, (box, score, class_id) in enumerate(
                zip(boxes, scores, class_ids)
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
                    id=str(label_id),
                    box2d=box2d,
                    category=(
                        self.inverse_cat_map[int(class_id)]
                        if self.inverse_cat_map != {}
                        else str(class_id)
                    ),
                    score=float(score),
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
        assert self.config is not None, "Scalabel config is not loaded."
        metrics_log: MetricLogs = {}
        short_description = ""

        if metric == self.METRICS_DET:
            results = evaluate_det(
                self.gt_frames, self.frames, config=self.config, nproc=0
            )
            for metric_name, metric_value in results.summary().items():
                metrics_log[metric_name] = metric_value
            short_description += str(results) + "\n"

        if metric == self.METRICS_INS_SEG:
            results = evaluate_ins_seg(
                self.gt_frames, self.frames, config=self.config, nproc=0
            )
            for metric_name, metric_value in results.summary().items():
                metrics_log[metric_name] = metric_value
            short_description += str(results) + "\n"

        return metrics_log, short_description
