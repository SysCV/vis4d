# type: ignore
"""Calculate statistics."""
import json
import pickle
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from bdd100k.common.utils import load_bdd100k_config
from bdd100k.label.to_scalabel import bdd100k_to_scalabel
from scalabel.label.io import load
from scalabel.label.typing import Dataset
from tqdm import tqdm

import vist.data.datasets.base
import vist.vis.image as image  # .imshow_bboxes as imshow_bboxes
from vist.common.bbox.matchers import MaxIoUMatcher
from vist.common.bbox.matchers.base import MatcherConfig
from vist.common.bbox.utils import compute_iou
from vist.data.datasets.base import BaseDatasetLoader
from vist.struct import Boxes2D

# pylint: skip-file
Categories = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
idx_to_class_mapping = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
}
class_to_idx_mapping = {v: int(k) for k, v in idx_to_class_mapping.items()}
matcher = MaxIoUMatcher(
    MatcherConfig(
        type="MaxIoUMatcher",
        thresholds=[0.3, 0.5],
        labels=[0, -1, 1],
        allow_low_quality_matches=False,
    )
)


def tlbr_to_xyah(tlbr: torch.tensor):
    """Convert bounding box to format .

    from (tlx, tly. brx, bry) to (center x, center y, aspect ratio, height).
    aspect ratio is `width / height`.
    """
    ret = tlbr.copy()
    ret[:, :2] = (tlbr[:, :2] + tlbr[:, 2:]) / 2.0
    ret[:, 2] = (tlbr[:, 2] - tlbr[:, 0]) / (tlbr[:, 3] - tlbr[:, 1])
    ret[:, 3] = tlbr[:, 3] - tlbr[:, 1]
    return ret


class BDD100K(BaseDatasetLoader):
    """BDD100K dataloading class."""

    def load_dataset(self) -> Dataset:
        """Convert BDD100K annotations to Scalabel format and prepare them."""
        assert self.cfg.annotations is not None
        bdd100k_anns = load(
            self.cfg.annotations,
            validate_frames=self.cfg.validate_frames,
            nprocs=self.cfg.num_processes,
        )
        frames = bdd100k_anns.frames
        assert self.cfg.config_path is not None
        bdd100k_cfg = load_bdd100k_config(self.cfg.config_path)

        scalabel_frames = bdd100k_to_scalabel(frames, bdd100k_cfg)
        return Dataset(frames=scalabel_frames, config=bdd100k_cfg.scalabel)


def load_bdd100k_preds(pred_path: str):
    """Function for loading BDD100K predictions."""
    # per video, per frame, storing detections of every frame
    search_dict: Dict[str, Dict[int, Boxes2D]] = dict()
    given_predictions = json.load(
        open(
            pred_path,
            "r",
        )
    )

    frames = given_predictions["frames"]
    for frame in tqdm(frames):
        video_name = frame["videoName"]
        frame_index = frame["frameIndex"]
        if video_name not in search_dict:
            search_dict[video_name] = dict()

        boxes = torch.empty((0, 5))
        class_ids = torch.empty((0))
        track_ids = torch.empty((0))
        if "labels" not in frame:
            search_dict[video_name][frame_index] = Boxes2D(torch.empty((0, 5)))
        else:
            for label in frame["labels"]:
                cat = label["category"]
                if cat not in class_to_idx_mapping:
                    continue
                x1, y1, x2, y2, score = (
                    label["box2d"]["x1"],
                    label["box2d"]["y1"],
                    label["box2d"]["x2"],
                    label["box2d"]["y2"],
                    label["score"],
                )

                boxes = torch.cat(
                    (
                        boxes,
                        torch.tensor(
                            [x1, y1, x2, y2, score],
                        ).unsqueeze(0),
                    ),
                    dim=0,
                )

                class_ids = torch.cat(
                    (
                        class_ids,
                        torch.tensor(
                            [class_to_idx_mapping[label["category"]]]
                        ),
                    )
                )
                track_ids = torch.cat(
                    (track_ids, torch.tensor([int(label["id"])]))
                )

            search_dict[video_name][frame_index] = Boxes2D(
                boxes, class_ids, track_ids.int()
            )

    return search_dict


def load_from_gt(dataset_cfg):
    """Function for loading BDD100K groung truth."""
    bdd100k_loader = BDD100K(dataset_cfg)
    dataset = bdd100k_loader.load_dataset()
    frames = dataset.frames
    # per category, per video, per id, dict, storing trajectories for every instance
    instance_dict: Dict[str, Dict[str, Dict[str, np.array]]] = dict()
    for cat in Categories:
        instance_dict[cat] = dict()
    # per video, per frame, dict, storing ground truth labels
    gt_dict: Dict[str, Dict[int, Boxes2D]] = dict()
    print("calculating instance_dict and gt_dict")
    for frame in tqdm(frames):
        video_name = frame.video_name
        frame_index = frame.frame_index
        if video_name not in gt_dict:
            gt_dict[video_name] = dict()
        for cat in Categories:
            if video_name not in instance_dict[cat]:
                instance_dict[cat][video_name] = dict()
        boxes = torch.empty((0, 5))
        class_ids = torch.empty((0))
        for label in frame.labels:
            cat = label.category
            label_id = label.id
            x1, y1, x2, y2 = (
                label.box2d.x1,
                label.box2d.y1,
                label.box2d.x2,
                label.box2d.y2,
            )
            if y2 == y1:  # some labels are length 0 in one dimension
                continue
            boxes = torch.cat(
                (
                    boxes,
                    torch.tensor(
                        [
                            x1,
                            y1,
                            x2,
                            y2,
                            1.0,
                        ],
                    ).unsqueeze(0),
                ),
                dim=0,
            )
            class_ids = torch.cat(
                (
                    class_ids,
                    torch.tensor([class_to_idx_mapping[label.category]]),
                )
            )
            xyahf = [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                (x2 - x1) / (y2 - y1),
                y2 - y1,
                frame_index,
            ]
            if label_id not in instance_dict[cat][video_name]:
                instance_dict[cat][video_name][label_id] = np.array([xyahf])
            else:
                instance_dict[cat][video_name][label_id] = np.append(
                    instance_dict[cat][video_name][label_id],
                    [xyahf],
                    axis=0,
                )
        gt_dict[video_name][frame_index] = Boxes2D(boxes, class_ids)

    with open(
        "/home/yinjiang/systm/data/instance_dict.pickle", "wb"
    ) as handle:
        pickle.dump(instance_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return gt_dict, instance_dict


if __name__ == "__main__":
    train_cfg = vist.data.datasets.base.BaseDatasetConfig(
        name="bdd100k_train",
        type="BDD100K",
        # annotations="/home/yinjiang/systm/data/one_sequence/labels",
        # data_root="/home/yinjiang/systm/data/one_sequence/images",
        # annotations="/home/yinjiang/systm/data/bdd100k/labels/box_track_20/train/",
        # data_root="/home/yinjiang/systm/data/bdd100k/images/track/train/",
        data_root=(
            "/home/yinjiang/systm/data/bdd100k_train_samples/images/track/train"
        ),
        annotations=(
            "/home/yinjiang/systm/data/bdd100k_train_samples/labels/box_track_20/train"
        ),
        config_path="box_track",
    )

    gt_dict, instance_dict = load_from_gt(train_cfg)

    given_pred = load_bdd100k_preds(
        "/home/yinjiang/systm/given_predictions/track_predictions.json",
    )

    ###########################################################################
    print("calculating detection covariance R.")
    # detection covariance R, per cat,storing deviaiton between detection and ground truth
    detect_cov_R: Dict[str, Dict[str, np.array]] = dict()
    for cat in Categories:
        detect_cov_R[cat] = dict(
            detections=np.empty([0, 4]),
            gt=np.empty((0, 4)),
            deviation=np.empty((0, 4)),
        )

    for video_name in gt_dict:
        for frame_index in gt_dict[video_name]:
            gt = gt_dict[video_name][frame_index]
            if frame_index not in given_pred[video_name]:
                continue
            detections = given_pred[video_name][frame_index]
            # some frame has 0 zero detections
            if len(detections) == 0 or len(gt) == 0:
                continue
            match_result = matcher.match([detections], [gt])[0]
            assigned_gt_indices = match_result.assigned_gt_indices
            assigned_labels = match_result.assigned_labels
            # image_b = torch.ones((3, 720, 1280)) * 255
            # image.imshow_bboxes(image_b, detections)
            # image.imshow_bboxes(image_b, gt)
            positive_mask = assigned_labels == 1
            assigned_gt_indices_masked = assigned_gt_indices[positive_mask]
            matched_gt = gt.boxes[assigned_gt_indices_masked, :-1]
            matched_pred = detections.boxes[positive_mask, :-1]
            matched_class = gt.class_ids[assigned_gt_indices_masked]
            matched_gt_xyah = tlbr_to_xyah(matched_gt.numpy())
            matched_pred_xyah = tlbr_to_xyah(matched_pred.numpy())
            for class_id, pred_xyah, gt_xyah in zip(
                matched_class, matched_pred_xyah, matched_gt_xyah
            ):
                cat = idx_to_class_mapping[int(class_id.item())]
                detect_cov_R[cat]["detections"] = np.append(
                    detect_cov_R[cat]["detections"],
                    pred_xyah[np.newaxis, :],
                    axis=0,
                )
                detect_cov_R[cat]["gt"] = np.append(
                    detect_cov_R[cat]["gt"], gt_xyah[np.newaxis, :], axis=0
                )
                detect_cov_R[cat]["deviation"] = np.append(
                    detect_cov_R[cat]["deviation"],
                    pred_xyah[np.newaxis, :] - gt_xyah[np.newaxis, :],
                    axis=0,
                )
    with open("/home/yinjiang/systm/data/det_cov_R.pickle", "wb") as handle:
        pickle.dump(detect_cov_R, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """
    ###########################################################################
    # motion covariance Q, calculated from instance_dict
    print("calculate motion covariance")

    def calculate_motion_cov_Q():
        with open(
            "/home/yinjiang/systm/data/instance_dict.pickle", "rb"
        ) as handle:
            instance_dict = pickle.load(handle)
        # per cat, storing motion deviation of all categories
        motion_cov_Q = dict()
        # per cat, storing xyah of all instances
        position_dict = dict()
        # per cat, storing velocity of all instances
        velocity_dict = dict()
        for cat in Categories:
            if cat not in motion_cov_Q:
                motion_cov_Q[cat] = dict()
                motion_cov_Q[cat]["height"] = np.empty((0))
                motion_cov_Q[cat]["deviation"] = np.empty((0, 8))
            if cat not in position_dict:
                position_dict[cat] = np.empty((0, 4))
            if cat not in velocity_dict:
                velocity_dict[cat] = np.empty((0, 4))
        for cat in Categories:
            print("processing ", cat)
            for video in tqdm(instance_dict[cat]):
                for instance_id in instance_dict[cat][video]:
                    record = instance_dict[cat][video][instance_id]
                    record_length = len(record)
                    if record_length < 1:
                        continue
                    elif record_length == 1:
                        position_dict[cat] = np.append(
                            position_dict[cat],
                            record[0][:-1][np.newaxis, :],
                            axis=0,
                        )
                    elif record_length == 2:
                        for rec in record:
                            position_dict[cat] = np.append(
                                position_dict[cat],
                                rec[:-1][np.newaxis, :],
                                axis=0,
                            )
                        if record[0][-1] + 1 == record[1][-1]:
                            vel = vel = record[1][:-1] - record[0][:-1]
                            velocity_dict[cat] = np.append(
                                velocity_dict[cat], vel[np.newaxis, :], axis=0
                            )
                    else:
                        f1, f2, f3 = 0, 1, 2
                        while f3 < record_length:
                            position_dict[cat] = np.append(
                                position_dict[cat],
                                record[f1][:-1][np.newaxis, :],
                                axis=0,
                            )
                            if record[f1][-1] + 1 == record[f2][-1]:
                                vel = record[f2][:-1] - record[f1][:-1]
                                velocity_dict[cat] = np.append(
                                    velocity_dict[cat],
                                    vel[np.newaxis, :],
                                    axis=0,
                                )
                            if (
                                record[f1][-1] + 1 == record[f2][-1]
                                and record[f2][-1] + 1 == record[f3][-1]
                            ):
                                vel = record[f2][:-1] - record[f1][:-1]
                                deviation = (
                                    record[f3][:-1] - record[f2][:-1] - vel
                                )
                                vel_2 = (
                                    record[f3][:-1]
                                    - 2 * record[f2][:-1]
                                    + record[f1][:-1]
                                )
                                deviation = np.append(deviation, vel_2)
                                motion_cov_Q[cat]["deviation"] = np.append(
                                    motion_cov_Q[cat]["deviation"],
                                    deviation[np.newaxis, :],
                                    axis=0,
                                )
                                motion_cov_Q[cat]["height"] = np.append(
                                    motion_cov_Q[cat]["height"], record[f3][3]
                                )
                            f1, f2, f3 = f1 + 1, f2 + 1, f3 + 1
                        f1, f2, f3 = f1 - 1, f2 - 1, f3 - 1
                        position_dict[cat] = np.append(
                            position_dict[cat],
                            record[f2][:-1][np.newaxis, :],
                            axis=0,
                        )
                        position_dict[cat] = np.append(
                            position_dict[cat],
                            record[f3][:-1][np.newaxis, :],
                            axis=0,
                        )
                        if record[f2][-1] + 1 == record[f3][-1]:
                            vel = vel = record[f3][:-1] - record[f2][:-1]
                            velocity_dict[cat] = np.append(
                                velocity_dict[cat], vel[np.newaxis, :], axis=0
                            )
        return motion_cov_Q, position_dict, velocity_dict

    motion_cov_Q, position_dict, velocity_dict = calculate_motion_cov_Q()

    print("saving")
    with open("/home/yinjiang/systm/data/motion_cov_Q.pickle", "wb") as handle:
        pickle.dump(motion_cov_Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
        "/home/yinjiang/systm/data/position_dict.pickle", "wb"
    ) as handle:
        pickle.dump(position_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
        "/home/yinjiang/systm/data/velocity_dict.pickle", "wb"
    ) as handle:
        pickle.dump(velocity_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """
    print("done")
