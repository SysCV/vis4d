"""S3DIS evaluator."""
import itertools
from typing import Any, Callable, List, Tuple

import torch
from terminaltables import AsciiTable
from torchmetrics import ConfusionMatrix

from vis4d.common import MetricLogs, ModelOutput
from vis4d.common.typing import COMMON_KEYS
from vis4d.data.datasets.base import DictData
from vis4d.data.datasets.coco import coco_det_map

from .base import Evaluator


class S3DisEvaluator(Evaluator):
    """S3Dis Evaluation"""

    def __init__(
        self, num_classes=13, iou_type: str = "bbox", split: str = "val2017"
    ):
        """Init."""
        super().__init__()
        self.num_classes = num_classes
        self._confusion_matrix = ConfusionMatrix(num_classes=num_classes)
        self.reset()

    @property
    def metrics(self) -> List[str]:
        """Supported metrics."""
        return ["mIoU", "confusion_matrix"]

    def gather(self, gather_func: Callable[[Any], Any]) -> None:
        """Accumulate predictions across prcoesses."""
        all_preds = gather_func(self._predictions)
        if all_preds is not None:
            self._predictions = list(itertools.chain(*all_preds))

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        self._confusion_matrix.reset()

    def process(self, inputs: DictData, outputs: ModelOutput) -> None:
        """Process sample."""
        targets = inputs[COMMON_KEYS.semantics3d]
        predictions = outputs[COMMON_KEYS.semantics3d]
        if self._confusion_matrix.device != targets.device:
            self._confusion_matrix = self._confusion_matrix.to(targets.device)

        for i, (pred, gt_label) in enumerate(zip(predictions, targets)):
            self._confusion_matrix.update(pred, gt_label)

    def _get_miou(self):
        confusion = self._confusion_matrix.compute()
        tp = torch.diag(confusion)
        fp = torch.sum(confusion, dim=0) - tp
        fn = torch.sum(confusion, dim=1) - tp
        iou = tp / (tp + fn + fp) * 100
        mIoU = iou.mean().cpu().item()

        iou_class_str = ", ".join(f"{d:.3f}%" for d in iou)
        return {
            "IoU": list(iou.cpu().numpy()),
            "mIoU": mIoU,
        }, f"mIoU: {mIoU:.3f}%, [" + iou_class_str + "]"

    def _get_confusion_matrix(self):
        headers = ["Confusion"] + [
            f"Class_{i}" for i in range(self.num_classes)
        ]
        table_data = (
            self._confusion_matrix.compute()
            / (torch.sum(self._confusion_matrix.compute(), dim=1))
        ).cpu()
        data = list(
            [f"Class_{idx}"] + list(d.numpy())
            for idx, d in enumerate(table_data)
        )
        table = AsciiTable([headers] + data)
        return {}, table.table

    def evaluate(self, metric: str) -> Tuple[MetricLogs, str]:
        """Evaluate predictions."""
        metric_data, short_description = {}, ""
        if metric == "mIoU" or metric == "all":
            data, desc = self._get_miou()
            metric_data[metric] = data
            short_description += desc + "\n"
        if metric == "confusion_matrix" or metric == "all":
            data, desc = self._get_confusion_matrix()
            metric_data[metric] = data
            short_description += desc + "\n"
        return metric_data, short_description
