import os
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
import tqdm
from PIL import Image
from scalabel.eval.sem_seg import (
    fast_hist,
    freq_iou,
    per_class_acc,
    per_class_iou,
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
        arr = np.asarray(image)
        w, h = image.size
        wp = self.size[1] - w
        hp = self.size[0] - h
        if len(arr.shape) == 3:
            image = T.pad(image, (0, 0, wp, hp), 0, "constant")
        else:
            image = T.pad(image, (0, 0, wp, hp), 255, "constant")
            print(np.unique(np.array(image)))
        return image


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


def pascal_label_encode(color_mask: np.ndarray) -> np.ndarray:
    """Encode segmentation label images as pascal classes
    Args:
        color_mask (np.ndarray): raw segmentation label image of dimension
            (M, N, 3), in which the Pascal classes are encoded as colors.
    Returns:
        label_mask (np.ndarray): class map with dimensions (M, N), where the
        value at a given location is the integer denoting the class index.
    """
    color_mask = np.asarray(color_mask)
    label_mask = np.zeros(color_mask.shape[:2], dtype=np.int64)
    for label_id, label in enumerate(PASCAL_LABEL):
        label_mask[
            np.where(np.all(color_mask == label, axis=-1))[:2]
        ] = label_id
    return label_mask


def pascal_label_decode(label_mask: np.ndarray) -> np.ndarray:
    """Decode segmentation label images as pascal classes
    Args:
        label_mask (np.ndarray): segmentation label image of dimension
            (M, N), in which the Pascal classes are numerical indices.
    Returns:
        color_mask (np.ndarray): color map with dimensions (M, N, 3), where the
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
        IoUs=np.multiply(ious, 100),
        Accs=np.multiply(accs, 100),
    )
    return res_dict, gt_id_set


def save_output_images(predictions, output_dir, colorize=True):
    """
    Saves a given tensor (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, prediction in enumerate(predictions):
        if len(prediction.shape) == 3:
            prediction = prediction.transpose((1, 2, 0))
        elif len(prediction.shape) == 2:
            if colorize:
                prediction = pascal_label_decode(prediction)
        im = Image.fromarray(prediction.astype(np.uint8))
        fn = os.path.join(output_dir, f"{i}.png")
        im.save(fn)


def read_output_images(image_dir):
    """
    Saves a given tensor (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    img_list = []
    for fn in sorted(list(os.listdir(image_dir))):
        if fn.endswith(".png") or fn.endswith(".jpg"):
            img = np.asarray(Image.open(os.path.join(image_dir, fn)))
        if len(img.shape) == 3:
            img = img.transpose((2, 0, 1))
        img_list.append(img)
    return img_list


def blend_images(images1, images2, alpha=0.6):
    img_list = []
    for img1, img2 in zip(images1, images2):
        print(img1.shape, img2.shape)
        if len(img1.shape) == 3:
            img1 = img1.transpose((1, 2, 0))
        if len(img2.shape) == 3:
            img2 = img2.transpose((1, 2, 0))
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img2 = img2.resize(img1.size)
        img = np.asarray(Image.blend(img1, img2, alpha))
        if len(img.shape) == 3:
            img = img.transpose((2, 0, 1))
        img_list.append(img)
    return img_list
