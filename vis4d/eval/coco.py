"""COCO evaluator."""
import contextlib
import copy
import io
import itertools
from typing import Any, Callable, List, Tuple

import numpy as np
import pycocotools.mask as maskUtils
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from vis4d.data.datasets.base import COMMON_KEYS, DictData
from vis4d.data.datasets.coco import coco_det_map
from vis4d.struct_to_revise import MetricLogs, ModelOutput

from .base import Evaluator


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert Tensor [N, 4] in xyxy format into xywh."""
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    return boxes


class COCOevalV2(COCOeval):
    """Subclass coco eval for logging / printing."""

    def summarize(self) -> Tuple[MetricLogs, str]:
        """Capture summary in string."""
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            super().summarize()
        summary_str = "\n" + f.getvalue()
        return {}, summary_str  # TODO summarize in metric logs


class COCOEvaluator(Evaluator):
    """COCO detection evaluation class."""

    def __init__(
        self, data_root: str, iou_type: str = "bbox", split: str = "val2017"
    ):
        """Init."""
        super().__init__()
        self.iou_type = iou_type
        self.coco_id2name = {v: k for k, v in coco_det_map.items()}
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_gt = COCO(
                f"{data_root}/annotations/instances_{split}.json"
            )
        coco_gt_cats = self._coco_gt.loadCats(self._coco_gt.getCatIds())
        self.cat_map = {c["name"]: c["id"] for c in coco_gt_cats}
        self.reset()

    @property
    def metrics(self) -> List[str]:
        """Supported metrics."""
        return ["COCO_AP"]

    def gather(self, gather_func: Callable[[Any], Any]) -> None:
        """Accumulate predictions across prcoesses."""
        all_preds = gather_func(self._predictions)
        if all_preds is not None:
            self._predictions = list(itertools.chain(*all_preds))

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        self._predictions = []

    def process(self, inputs: DictData, outputs: ModelOutput) -> None:
        """Process sample and convert detections to coco format."""
        for i, (image_id, boxes, scores, classes) in enumerate(
            zip(
                inputs[COMMON_KEYS.metadata]["coco_image_id"],
                outputs["boxes2d"],
                outputs["boxes2d_scores"],
                outputs["boxes2d_classes"],
            )
        ):
            masks = (
                outputs["masks"][i]
                if "masks" in outputs
                else [None for _ in range(len(boxes))]
            )
            annotations = []
            boxes = xyxy_to_xywh(boxes)
            for box, score, cls, mask in zip(boxes, scores, classes, masks):
                xywh = box.cpu().numpy().tolist()
                area = float(xywh[2] * xywh[3])
                annotation = dict(
                    image_id=image_id,
                    bbox=xywh,
                    area=area,
                    score=float(score),
                    category_id=self.cat_map[self.coco_id2name[int(cls)]],
                    iscrowd=0,
                )
                if mask is not None:
                    annotation["segmentation"] = maskUtils.encode(
                        np.array(mask.cpu(), order="F", dtype="uint8")
                    )
                    annotation["segmentation"]["counts"] = annotation[
                        "segmentation"
                    ]["counts"].decode()
                annotations.append(annotation)

            self._predictions.extend(annotations)

    def evaluate(self, metric: str) -> Tuple[MetricLogs, str]:
        """Evaluate predictions."""
        if metric == "COCO_AP":
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

            # TODO putting code here, need to organize
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = evaluator.eval["precision"]
            # precision: (iou, recall, cls, area range, max dets)
            assert len(self._coco_gt.getCatIds()) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(self._coco_gt.getCatIds()):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = self._coco_gt.loadCats(catId)[0]
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
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print("\n" + table.table)
            return evaluator.summarize()
        raise NotImplementedError(f"Metric {metric} not known!")
