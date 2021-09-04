"""Functions for loading given predictions."""
import glob
import json
import os
from typing import Dict

import numpy as np
import torch

# from detectron2.data import MetadataCatalog
from vist.struct import Boxes2D, NDArrayF64


def load_bdd100k_preds(pred_path: str) -> Dict[str, Dict[int, Boxes2D]]:
    """Function for loading BDD100K predictions."""
    search_dict: Dict[str, Dict[int, Boxes2D]] = dict()
    given_predictions = json.load(
        open(
            pred_path,
            "r",
        )
    )
    # idx_to_class_mapping = MetadataCatalog.get(
    #     "bdd100k_val"
    # ).idx_to_class_mapping
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
    class_to_idx_mapping = {v: k for k, v in idx_to_class_mapping.items()}
    given_predictions = given_predictions["frames"]
    for prediction in given_predictions:
        video_name = prediction["videoName"]
        frame_index = prediction["frameIndex"]
        if video_name not in search_dict:
            search_dict[video_name] = dict()
        boxes = torch.empty((0, 5))
        class_ids = torch.empty((0))
        track_ids = torch.empty((0))
        if "labels" not in prediction:
            search_dict[video_name][frame_index] = Boxes2D(torch.empty((0, 5)))
        else:
            for label in prediction["labels"]:
                if label["category"] not in class_to_idx_mapping:
                    continue
                boxes = torch.cat(
                    (
                        boxes,
                        torch.tensor(
                            [
                                label["box2d"]["x1"],
                                label["box2d"]["y1"],
                                label["box2d"]["x2"],
                                label["box2d"]["y2"],
                                label["score"],
                            ],
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


def load_mot16_preds(pred_path: str) -> Dict[str, Dict[int, Boxes2D]]:
    """Function for loading MOT16 predictions."""
    search_dict: Dict[str, Dict[int, Boxes2D]] = dict()
    video_names = glob.glob(os.path.join(pred_path, "MOT16-*_det.txt"))
    for v in video_names:
        video_name, _ = os.path.splitext(os.path.split(v)[1])
        video_name = video_name[:-4]
        search_dict[video_name] = dict()
        detections = np.loadtxt(v, delimiter=",")
        detections[:, 2:6] = tlwh_to_xyxy(detections[:, 2:6])
        frames = np.unique(detections[:, 0])
        for f_id in frames:
            frame_data = detections[detections[:, 0] == f_id]
            boxes = torch.from_numpy(frame_data[:, 2:7]).float()
            class_ids = torch.zeros(boxes.shape[0])
            search_dict[video_name][f_id - 1] = Boxes2D(boxes, class_ids)

    return search_dict


def tlwh_to_xyxy(tlwh: NDArrayF64) -> NDArrayF64:
    """Convert tlwh boxes to xyxy.

    tlwh: shape(n x 4), where axis 1 is [x1, y1, w, h]
    """
    x1 = tlwh[:, [0]]
    x2 = tlwh[:, [0]] + tlwh[:, [2]]
    y1 = tlwh[:, [1]]
    y2 = tlwh[:, [1]] + tlwh[:, [3]]
    xyxy = np.concatenate([x1, y1, x2, y2], axis=1)
    return xyxy


def load_predictions(
    dataset_name: str, pred_path: str
) -> Dict[str, Dict[int, Boxes2D]]:
    """Function for calling specific prediction loader."""
    if dataset_name == "BDD100K":
        return load_bdd100k_preds(pred_path)
    if dataset_name == "MOT16":
        return load_mot16_preds(pred_path)
    else:
        raise NotImplementedError("not implemented dataset")


if __name__ == "__main__":
    # mot16_preds = load_predictions("MOT16", "weight/MOT16_det_feat")
    bdd100K = load_predictions("BDD100K", "weight/predictions.json")
    THRESHOLD = 0.3
    MIN_CONF = 1.0
    COUNT = 0
    for video_id, video in bdd100K.items():
        for frame_id, frame in video.items():
            if len(frame.boxes) == 0:
                continue
            if torch.min(frame.boxes[:, -1]) < MIN_CONF:
                MIN_CONF = torch.min(frame.boxes[:, -1])
            threshold_conf = frame.boxes[:, -1] < THRESHOLD
            COUNT += torch.sum(threshold_conf)
    print("COUNT: ", COUNT)
    print("MIN_CONF", MIN_CONF)
