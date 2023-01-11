"""COCO evaluator."""
from __future__ import annotations

import contextlib
import copy
import io
import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable  # type: ignore

from vis4d.common import MetricLogs
from vis4d.common.typing import DictStrAny, NDArrayNumber
from vis4d.data.datasets.coco import coco_det_map

from ..base import Evaluator


def xyxy_to_xywh(boxes: NDArrayNumber) -> NDArrayNumber:
    """Convert Tensor [N, 4] in xyxy format into xywh.

    Args:
        boxes (NDArrayNumber): Bounding boxes in Vis4D format.

    Returns:
        NDArrayNumber: COCO format bounding boxes.
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
    image_id: str,
    boxes: NDArrayNumber,
    scores: NDArrayNumber,
    classes: NDArrayNumber,
    masks: None | NDArrayNumber = None,
) -> list[DictStrAny]:
    """Convert Vis4D format predictions to COCO format.

    Args:
        cat_map (dict[str, int]): COCO class name to class ID mapping.
        coco_id2name (dict[int, str]): COCO class ID to class name mapping.
        image_id (str): ID of image.
        boxes (NDArrayNumber): Predicted bounding boxes.
        scores (NDArrayNumber): Predicted scores for each box.
        classes (NDArrayNumber): Predicted classes for each box.
        masks (None | NDArrayNumber, optional): Predicted masks. Defaults to
            None.

    Returns:
        list[DictStrAny]: Predictions in COCO format.
    """
    predictions = []
    boxes = xyxy_to_xywh(boxes)
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        mask = masks[i] if masks is not None else None
        xywh = box.tolist()
        area = float(xywh[2] * xywh[3])
        annotation = dict(
            image_id=image_id,
            bbox=xywh,
            area=area,
            score=float(score),
            category_id=cat_map[coco_id2name[int(cls)]],
            iscrowd=0,
        )
        if mask is not None:
            annotation["segmentation"] = maskUtils.encode(
                np.array(mask.cpu(), order="F", dtype="uint8")
            )
            annotation["segmentation"]["counts"] = annotation["segmentation"][
                "counts"
            ].decode()
        predictions.append(annotation)
    return predictions


class COCOEvaluator(Evaluator):
    """COCO detection evaluation class."""

    def __init__(
        self,
        data_root: str,
        iou_type: str = "bbox",
        split: str = "val2017",
        per_class_eval: bool = False,
    ) -> None:
        """Creates an instance of the class.

        Args:
            data_root (str): Root directory of data.
            iou_type (str, optional): Type of IoU computation to use. Should be
                set to either "bbox" for bounding or "segm" for masks. Defaults
                to "bbox".
            split (str, optional): COCO data split. Defaults to "val2017".
            per_class_eval (bool, optional): Per-class evaluation. Defaults to
                False.
        """
        super().__init__()
        assert iou_type in {"bbox", "segm"}
        self.iou_type = iou_type
        self.per_class_eval = per_class_eval
        self.coco_id2name = {v: k for k, v in coco_det_map.items()}
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_gt = COCO(
                f"{data_root}/annotations/instances_{split}.json"
            )
        coco_gt_cats = self._coco_gt.loadCats(self._coco_gt.getCatIds())
        self.cat_map = {c["name"]: c["id"] for c in coco_gt_cats}
        self._predictions: list[DictStrAny] = []

    @property
    def metrics(self) -> list[str]:
        """Supported metrics.

        Returns:
            list[str]: Metrics to evaluate.
        """
        return ["COCO_AP"]

    def gather(
        self, gather_func: Callable[[Any], Any]
    ) -> None:  # type: ignore
        """Accumulate predictions across prcoesses."""
        all_preds = gather_func(self._predictions)
        if all_preds is not None:
            self._predictions = list(itertools.chain(*all_preds))

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        self._predictions = []

    def process(  # type: ignore # pylint: disable=arguments-differ
        self,
        coco_image_id: list[int],
        pred_boxes: list[NDArrayNumber],
        pred_scores: list[NDArrayNumber],
        pred_classes: list[NDArrayNumber],
        pred_masks: None | list[NDArrayNumber] = None,
    ) -> None:
        """Process sample and convert detections to coco format.

        coco_image_id (list[int]): COCO image ID.
        pred_boxes (list[NDArrayNumber]): Predicted bounding boxes.
        pred_scores (list[NDArrayNumber]): Predicted scores for each box.
        pred_classes (list[NDArrayNumber]): Predicted classes for each box.
        pred_masks (None | list[NDArrayNumber], optional): Predicted masks.
        """
        for i, (image_id, boxes, scores, classes) in enumerate(
            zip(coco_image_id, pred_boxes, pred_scores, pred_classes)
        ):
            masks = pred_masks[i] if pred_masks else None
            coco_preds = predictions_to_coco(
                self.cat_map,
                self.coco_id2name,
                image_id,
                boxes,
                scores,
                classes,
                masks,
            )

            self._predictions.extend(coco_preds)

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate COCO predictions.

        Args:
            metric (str): Metric to evaluate. Should be "COCO_AP".

        Raises:
            NotImplementedError: Raised if metric is not "COCO_AP".

        Returns:
            tuple[MetricLogs, str]: Dictionary of scores to log and a pretty
                printed string.
        """
        if metric != "COCO_AP":
            raise NotImplementedError(f"Metric {metric} not known!")

        with contextlib.redirect_stdout(io.StringIO()):
            if self.iou_type == "segm":
                # remove bbox for segm evaluation so cocoapi will use mask
                # area instead of box area
                _predictions = copy.deepcopy(self._predictions)
                for pred in _predictions:
                    pred.pop("bbox")
            else:
                _predictions = self._predictions
            coco_dt = self._coco_gt.loadRes(_predictions)
            evaluator = COCOevalV2(
                self._coco_gt, coco_dt, iouType=self.iou_type
            )
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
                    ap = np.mean(precision)
                else:
                    ap = float("nan")
                results_per_category.append(
                    (f'{nm["name"]}', f"{float(ap):0.3f}")
                )

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
