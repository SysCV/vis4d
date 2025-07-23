"""COCO evaluator."""

from __future__ import annotations

import contextlib
import copy
import io
import itertools

import numpy as np
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from vis4d.common.array import array_to_numpy
from vis4d.common.logging import rank_zero_warn
from vis4d.common.typing import (
    ArrayLike,
    DictStrAny,
    GenericFunc,
    MetricLogs,
    NDArrayF32,
    NDArrayI64,
)
from vis4d.data.datasets.coco import coco_det_map

from ..base import Evaluator


def xyxy_to_xywh(boxes: NDArrayF32) -> NDArrayF32:
    """Convert Tensor [N, 4] in xyxy format into xywh.

    Args:
        boxes (NDArrayF32): Bounding boxes in Vis4D format.

    Returns:
        NDArrayF32: COCO format bounding boxes.
    """
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    return boxes


class COCOevalV2(COCOeval):  # type: ignore
    """Subclass COCO eval for logging / printing."""

    def summarize(self) -> str:
        """Capture summary in string.

        Returns:
            str: Pretty printed string.
        """
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            super().summarize()
        summary_str = "\n" + f.getvalue()
        return summary_str


def predictions_to_coco(
    cat_map: dict[str, int],
    coco_id2name: dict[int, str],
    image_id: int,
    boxes: NDArrayF32,
    scores: NDArrayF32,
    classes: NDArrayI64,
    masks: None | NDArrayF32 = None,
) -> list[DictStrAny]:
    """Convert Vis4D format predictions to COCO format.

    Args:
        cat_map (dict[str, int]): COCO class name to class ID mapping.
        coco_id2name (dict[int, str]): COCO class ID to class name mapping.
        image_id (int): ID of image.
        boxes (NDArrayF32): Predicted bounding boxes.
        scores (NDArrayF32): Predicted scores for each box.
        classes (NDArrayI64): Predicted classes for each box.
        masks (None | NDArrayF32, optional): Predicted masks. Defaults to
            None.

    Returns:
        list[DictStrAny]: Predictions in COCO format.
    """
    predictions = []
    boxes_xyxy = copy.deepcopy(boxes)
    boxes_xywh = xyxy_to_xywh(boxes_xyxy)
    for i, (box, score, cls) in enumerate(zip(boxes_xywh, scores, classes)):
        mask = masks[i] if masks is not None else None
        xywh = box.tolist()
        area = float(xywh[2] * xywh[3])
        annotation = {
            "image_id": image_id,
            "bbox": xywh,
            "area": area,
            "score": float(score),
            "category_id": cat_map[coco_id2name[int(cls)]],
            "iscrowd": 0,
        }
        if mask is not None:
            annotation["segmentation"] = maskUtils.encode(
                np.array(mask, order="F", dtype="uint8")
            )
            annotation["segmentation"]["counts"] = annotation["segmentation"][
                "counts"
            ].decode()
        predictions.append(annotation)
    return predictions


class COCODetectEvaluator(Evaluator):
    """COCO detection evaluation class."""

    METRIC_DET = "Det"
    METRIC_INS_SEG = "InsSeg"

    def __init__(
        self,
        data_root: str,
        split: str = "val2017",
        per_class_eval: bool = False,
    ) -> None:
        """Creates an instance of the class.

        Args:
            data_root (str): Root directory of data.
            split (str, optional): COCO data split. Defaults to "val2017".
            per_class_eval (bool, optional): Per-class evaluation. Defaults to
                False.
        """
        super().__init__()
        self.per_class_eval = per_class_eval
        self.coco_id2name = {v: k for k, v in coco_det_map.items()}
        self.annotation_path = (
            f"{data_root}/annotations/instances_{split}.json"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_gt = COCO(self.annotation_path)
        coco_gt_cats = self._coco_gt.loadCats(self._coco_gt.getCatIds())
        self.cat_map = {c["name"]: c["id"] for c in coco_gt_cats}
        self._predictions: list[DictStrAny] = []

    @property
    def metrics(self) -> list[str]:
        """Supported metrics.

        Returns:
            list[str]: Metrics to evaluate.
        """
        return [self.METRIC_DET, self.METRIC_INS_SEG]

    def gather(self, gather_func: GenericFunc) -> None:
        """Accumulate predictions across processes."""
        all_preds = gather_func(self._predictions)
        if all_preds is not None:
            self._predictions = list(itertools.chain(*all_preds))

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        self._predictions = []

    def process_batch(
        self,
        coco_image_id: list[int],
        pred_boxes: list[ArrayLike],
        pred_scores: list[ArrayLike],
        pred_classes: list[ArrayLike],
        pred_masks: None | list[ArrayLike] = None,
    ) -> None:
        """Process sample and convert detections to coco format.

        coco_image_id (list[int]): COCO image ID.
        pred_boxes (list[ArrayLike]): Predicted bounding boxes.
        pred_scores (list[ArrayLike]): Predicted scores for each box.
        pred_classes (list[ArrayLike]): Predicted classes for each box.
        pred_masks (None | list[ArrayLike], optional): Predicted masks.
        """
        for i, (image_id, boxes, scores, classes) in enumerate(
            zip(coco_image_id, pred_boxes, pred_scores, pred_classes)
        ):
            boxes_np = array_to_numpy(boxes, n_dims=None, dtype=np.float32)
            scores_np = array_to_numpy(scores, n_dims=None, dtype=np.float32)
            classes_np = array_to_numpy(classes, n_dims=None, dtype=np.int64)

            if pred_masks is not None:
                masks_np = array_to_numpy(
                    pred_masks[i], n_dims=3, dtype=np.float32
                )
            else:
                masks_np = None

            coco_preds = predictions_to_coco(
                self.cat_map,
                self.coco_id2name,
                image_id,
                boxes_np,
                scores_np,
                classes_np,
                masks_np,
            )
            self._predictions.extend(coco_preds)

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate COCO predictions.

        Args:
            metric (str): Metric to evaluate. Should be "COCO_AP".

        Raises:
            NotImplementedError: Raised if metric is not "COCO_AP".
            RuntimeError: Raised if no predictions are available.

        Returns:
            tuple[MetricLogs, str]: Dictionary of scores to log and a pretty
                printed string.
        """
        if metric not in [self.METRIC_DET, self.METRIC_INS_SEG]:
            raise NotImplementedError(f"Metric {metric} not known!")

        if len(self._predictions) == 0:
            rank_zero_warn(
                "No predictions to evaluate. Make sure to process batch first!"
            )
            return {
                "AP": 0.0,
                "AP50": 0.0,
                "AP75": 0.0,
                "APs": 0.0,
                "APm": 0.0,
                "APl": 0.0,
            }, "No predictions to evaluate."

        if metric == self.METRIC_DET:
            iou_type = "bbox"
            _predictions = self._predictions
        else:
            # remove bbox for segm evaluation so cocoapi will use mask
            # area instead of box area
            iou_type = "segm"
            _predictions = copy.deepcopy(self._predictions)
            for pred in _predictions:
                pred.pop("bbox")
        coco_dt = self._coco_gt.loadRes(_predictions)

        with contextlib.redirect_stdout(io.StringIO()):
            assert coco_dt is not None
            evaluator = COCOevalV2(self._coco_gt, coco_dt, iouType=iou_type)
            evaluator.evaluate()
            evaluator.accumulate()

        log_str = evaluator.summarize()
        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
        score_dict = dict(zip(metrics, evaluator.stats))

        if self.per_class_eval:
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = evaluator.eval["precision"]
            # precision: (iou, recall, cls, area range, max dets)
            assert len(self._coco_gt.getCatIds()) == precisions.shape[2]

            results_per_category = []
            for idx, cat_id in enumerate(self._coco_gt.getCatIds()):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = self._coco_gt.loadCats(cat_id)[0]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision).item()
                else:
                    ap = float("nan")
                results_per_category.append((f'{nm["name"]}', f"{ap:0.3f}"))

            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            headers = ["category", "AP"] * (num_columns // 2)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::num_columns] for i in range(num_columns)]
            )
            table_data = [headers] + list(results_2d)
            table = AsciiTable(table_data)
            log_str = f"\n{table.table}\n{log_str}"

        return score_dict, log_str

    def __repr__(self) -> str:
        """Returns the string representation of the object."""
        return f"CocoEvaluator(annotation_path={self.annotation_path})"
