"""Common utils for segmentation."""
from __future__ import annotations

from functools import partial
from multiprocessing import Pool

import numpy as np

from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.common.typing import NDArrayInt, NDArrayUI8

if SCALABEL_AVAILABLE:
    from scalabel.eval.sem_seg import (
        fast_hist,
        freq_iou,
        per_class_acc,
        per_class_iou,
        whole_acc,
    )


PASCAL_LABEL = np.asarray(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]
)


def pascal_label_encode(color_mask: NDArrayInt) -> NDArrayInt:
    """Encode segmentation label images as pascal classes.

    Args:
        color_mask (NDArrayInt): raw segmentation label image of dimension
            (M, N, 3), in which the Pascal classes are encoded as colors.

    Returns:
        label_mask (NDArrayInt): class map with dimensions (M, N), where the
        value at a given location is the integer denoting the class index.
    """
    color_mask = np.asarray(color_mask)
    label_mask = np.zeros(color_mask.shape[:2], dtype=np.int64)
    for label_id, label in enumerate(PASCAL_LABEL):
        label_mask[
            np.where(np.all(color_mask == label, axis=-1))[:2]
        ] = label_id
    return label_mask


def pascal_label_decode(label_mask: NDArrayInt) -> NDArrayUI8:
    """Decode segmentation label images as pascal classes.

    Args:
        label_mask (NDArrayInt): segmentation label image of dimension
            (M, N), in which the Pascal classes are numerical indices.

    Returns:
        color_mask (NDArrayUI8): color map with dimensions (M, N, 3), where the
        value at a given location is the integer denoting the class index.
    """
    assert len(label_mask.shape) == 2
    color_mask = np.zeros(
        (label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8
    )
    for label_id, label in enumerate(PASCAL_LABEL):
        color_mask[np.where(label_mask == label_id)[:2]] = label
    return color_mask


def per_image_hist(
    target: NDArrayUI8,
    pred: NDArrayUI8,
    num_classes: int,
    ignore_label: int = 255,
) -> tuple[NDArrayInt, set[int]]:
    """Calculate per image hist.

    Args:
        target (NDArrayUI8): The ground truth.
        pred (NDArrayUI8): The prediction.
        num_classes (int): The number of classes.
        ignore_label (int): The class index that should be ignored.
            Defaults to 255.

    Returns:
        tuple[np.ndarray, set[int]]: The histogram and the set of ground truth
            ids.
    """
    num_classes = num_classes + 1
    assert num_classes >= 2
    assert num_classes <= ignore_label
    target = target.copy()
    target[target == ignore_label] = num_classes - 1
    gt_id_set = set(np.unique(target).tolist())

    # remove `ignored`
    if num_classes - 1 in gt_id_set:
        gt_id_set.remove(num_classes - 1)

    if len(pred) == 0:
        # empty mask
        pred = np.empty_like(target)
        pred.fill(ignore_label)
    hist = fast_hist(target.flatten(), pred.flatten(), num_classes)
    return hist, gt_id_set


def evaluate_sem_seg(
    ann_frames: list[NDArrayUI8],
    pred_frames: list[NDArrayUI8],
    num_classes: int,
    ignore_label: int = 255,
    nproc: int = 4,
) -> tuple[dict[str, float], set[int]]:
    """Evaluate segmentation result.

    Args:
        ann_frames (list[torch.Tensor]): The ground truth frames.
        pred_frames (list[torch.Tensor]): The prediction frames.
        num_classes (int): Metadata config.
        ignore_label (int): The value of ignored label. Defaults to 255.
        nproc (int): the number of process.

    Returns:
        res_dict (dict[str, float]): Dictionary of evaluation results.
        gt_id_set (set): Set of unique ground truth ids.
    """
    if nproc > 1:
        with Pool(nproc) as pool:
            hist_and_gt_id_sets = pool.starmap(
                partial(
                    per_image_hist,
                    num_classes=num_classes,
                    ignore_label=ignore_label,
                ),
                zip(ann_frames, pred_frames),
            )
    else:
        hist_and_gt_id_sets = [
            per_image_hist(
                ann_frame,
                pred_frame,
                num_classes=num_classes,
                ignore_label=ignore_label,
            )
            for ann_frame, pred_frame in zip(ann_frames, pred_frames)
        ]
    num_classes = num_classes + 1
    hist = np.zeros((num_classes, num_classes), dtype=np.int32)
    gt_id_set = set()
    for hist_, gt_id_set_ in hist_and_gt_id_sets:
        hist += hist_
        gt_id_set.update(gt_id_set_)

    ious = per_class_iou(hist)
    accs = per_class_acc(hist)
    res_dict = {
        "mIoU": np.multiply(np.mean(ious[list(gt_id_set)]), 100),
        "Acc": np.multiply(np.mean(accs[list(gt_id_set)]), 100),
        "fIoU": np.multiply(freq_iou(hist), 100),
        "pAcc": np.multiply(whole_acc(hist), 100),
        "IoUs": np.multiply(ious, 100),
        "Accs": np.multiply(accs, 100),
    }
    return res_dict, gt_id_set
