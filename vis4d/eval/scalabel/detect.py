"""Scalabel tracking evaluator."""
from __future__ import annotations

import numpy as np

from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.common.typing import MetricLogs, NDArrayNumber

from .base import ScalabelEvaluator

if SCALABEL_AVAILABLE:
    from scalabel.eval.detect import evaluate_det
    from scalabel.eval.ins_seg import evaluate_ins_seg
    from scalabel.label.transforms import mask_to_rle, xyxy_to_box2d
    from scalabel.label.typing import Frame, Label


class ScalabelDetectEvaluator(ScalabelEvaluator):
    """Scalabel 2D tracking evaluation class."""

    METRICS_DET = "det"
    METRICS_INS_SEG = "ins_seg"
    METRICS_ALL = "all"

    def __init__(
        self,
        annotation_path: str,
        mask_threshold: float = 0.0,
    ) -> None:
        """Initialize the evaluator."""
        super().__init__(annotation_path=annotation_path)
        self.mask_threshold = mask_threshold

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "Scalabel Tracking Evaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return [self.METRICS_DET, self.METRICS_INS_SEG]

    def process(  # type: ignore # pylint: disable=arguments-differ
        self,
        frame_ids: list[int],
        sample_names: list[str],
        sequence_names: list[str],
        pred_boxes: list[NDArrayNumber],
        pred_classes: list[NDArrayNumber],
        pred_scores: list[NDArrayNumber],
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
            labels = []
            for box, score, class_id in zip(boxes, scores, class_ids):
                box2d = xyxy_to_box2d(*box.tolist())
                label = Label(
                    box2d=box2d,
                    category=self.inverse_cat_map[int(class_id)]
                    if self.inverse_cat_map != {}
                    else str(class_id),
                    score=float(score),
                    rle=mask_to_rle(
                        (pred_masks[i][class_id] > self.mask_threshold).astype(
                            np.uint8
                        )
                    )
                    if pred_masks
                    else None,
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
        if metric in [self.METRICS_DET, self.METRICS_ALL]:
            results = evaluate_det(
                self.gt_frames,
                self.frames,
                config=self.config,
                nproc=0,
            )
        elif metric in [self.METRICS_INS_SEG, self.METRICS_ALL]:
            results = evaluate_ins_seg(
                self.gt_frames,
                self.frames,
                config=self.config,
                nproc=0,
            )
        else:
            raise NotImplementedError

        return results.summary(), str(results)  # type: ignore
