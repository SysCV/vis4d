from typing import Tuple, Set, Dict, List
import tqdm
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T

from scalabel.eval.sem_seg import (
    fast_hist,
    per_class_iou,
    per_class_acc,
    freq_iou,
    whole_acc,
)


class ResizeWithPadding:
    """Padding image to desired size."""

    def __init__(self, size: Tuple[int, int]):
        """Init.

        Args:
            size (Tuple[int, int]): The desired size of image, (height, width).
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        wp = self.size[1] - w
        hp = self.size[0] - h
        image = T.pad(image, (0, 0, wp, hp), 0, "constant")
        return image


def resize_feat(
    feat: torch.Tensor,
    resize: Tuple[int, int],
    align_corners: bool = False,
) -> torch.Tensor:
    """Resize the image features.

    Args:
        feat (torch.Tensor): Image features.
        resize (Tuple[int, int]): The shape that prediction maps will be
            resized to.
        align_corners (bool): Defaults to False.

    Returns:
        resized_feat (torch.Tensor): List of resized features.
    """
    resized_feat = F.interpolate(
        input=feat,
        size=resize,
        mode="bilinear",
        align_corners=align_corners,
    )
    return resized_feat


class PascalMaskEncoder:
    def __init__(self) -> None:
        self.pascal_label = np.asarray(
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

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
                (M, N, 3), in which the Pascal classes are encoded as colors.
        Returns:
            label_mask (np.ndarray): class map with dimensions (M, N), where the
            value at a given location is the integer denoting the class index.
        """
        mask = np.asarray(mask)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for label_id, label in enumerate(self.pascal_label):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = label_id
        label_mask = label_mask.astype(int)
        print(label_mask.shape, mask.shape)
        return label_mask


def per_image_hist(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_label: int = 255,
) -> Tuple[np.ndarray, Set[int]]:
    """Calculate per image hist."""
    assert num_classes >= 2
    # assert num_classes <= ignore_label
    gt = gt.copy()
    gt[gt == ignore_label] = num_classes - 1
    gt_id_set = set(np.unique(gt).tolist())

    # remove `ignored`
    if num_classes - 1 in gt_id_set:
        gt_id_set.remove(num_classes - 1)

    if len(pred) == 0:
        # empty mask
        pred = np.empty_like(gt)
        pred.fill(ignore_label)
    hist = fast_hist(gt.flatten(), pred.flatten(), num_classes)
    return hist, gt_id_set


def evaluate_sem_seg(
    ann_frames: List[torch.Tensor],
    pred_frames: List[torch.Tensor],
    num_classes: int,
    ignore_label: int = 255,
    nproc: int = 4,
) -> Tuple[Dict, Set]:
    """Evaluate segmentation with Scalabel format.

    Args:
        ann_frames: the ground truth frames.
        pred_frames: the prediction frames.
        num_classes: Metadata config.
        nproc: the number of process.

    Returns:
        SegResult: evaluation results.
    """
    if nproc > 1:
        with Pool(nproc) as pool:
            hist_and_gt_id_sets = pool.starmap(
                partial(
                    per_image_hist,
                    num_classes=num_classes,
                    ignore_label=ignore_label,
                ),
                tqdm.tqdm(zip(ann_frames, pred_frames), total=len(ann_frames)),
            )
    else:
        hist_and_gt_id_sets = [
            per_image_hist(
                ann_frame,
                pred_frame,
                num_classes=num_classes,
                ignore_label=ignore_label,
            )
            for ann_frame, pred_frame in tqdm(
                zip(ann_frames, pred_frames), total=len(ann_frames)
            )
        ]

    # num_classes = num_classes + 1
    hist = np.zeros((num_classes, num_classes), dtype=np.int32)
    gt_id_set = set()
    for (hist_, gt_id_set_) in hist_and_gt_id_sets:
        hist += hist_
        gt_id_set.update(gt_id_set_)

    ious = per_class_iou(hist)
    accs = per_class_acc(hist)
    res_dict = dict(
        mIoU=np.multiply(np.mean(ious[list(gt_id_set)]), 100),
        Acc=np.multiply(np.mean(accs[list(gt_id_set)]), 100),
        fIoU=np.multiply(freq_iou(hist), 100),
        pAcc=np.multiply(whole_acc(hist), 100),
    )
    return res_dict, gt_id_set
