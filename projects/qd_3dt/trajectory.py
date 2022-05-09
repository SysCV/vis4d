"""Class for processing Trajectory datasets."""
import json
import os
import pickle
from typing import Dict, List

import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from scalabel.label.io import load
from scalabel.label.typing import Box3D
from scalabel.label.typing import Dataset as ScalabelDataset
from scalabel.label.typing import Label
from scalabel.label.utils import rotation_y_to_alpha
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from vis4d.common.utils.time import Timer
from vis4d.struct import Boxes3D, InputSample


class TrajectoryData(Dataset):  # type: ignore
    """Trajectory dataloading class."""

    def __init__(
        self,
        pure_detection_path: str,
        annotations_path: str,
        cache_path: str,
        cats_name2id: Dict[str, int],
        min_seq_len: int = 10,
    ) -> None:
        """Init dataset."""
        super().__init__()
        self.min_seq_len = min_seq_len

        # Load labels from existing file
        t = Timer()
        if os.path.exists(cache_path):
            with open(cache_path, mode="rb") as f:
                trajectories = pickle.load(f)
        else:
            rank_zero_info("No cache file found, generating trajectories...")
            trajectories = self.generate_trajectories(
                pure_detection_path, annotations_path, cache_path
            )

        rank_zero_info(
            f"Load {len(trajectories)} trajectory takes "
            f"{t.time():.2f} seconds."
        )

        self.trajectories = trajectories
        self.cats_name2id = cats_name2id

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> InputSample:
        """Return the item at the given index."""
        trajectory = self.trajectories[idx]
        traj_len = len(trajectory["gt"])

        if traj_len > self.min_seq_len:
            first_frame = np.random.randint(traj_len - self.min_seq_len)
        else:
            first_frame = 0

        gt_traj = trajectory["gt"][
            first_frame : first_frame + self.min_seq_len
        ]
        pred_traj = trajectory["pred"][
            first_frame : first_frame + self.min_seq_len
        ]

        gt_boxes3d = Boxes3D.from_scalabel(gt_traj, self.cats_name2id)
        pred_boxes3d = Boxes3D.from_scalabel(pred_traj, self.cats_name2id)

        input_data = InputSample([None, None])
        input_data.targets.boxes3d = [pred_boxes3d, gt_boxes3d]

        return input_data

    @staticmethod
    def gt_pred_matching(gt_world: Label, predictions: List[Label]) -> Label:
        """Matching gt and pred."""
        if predictions is not None:
            same_class_preds = [
                pred
                for pred in predictions
                if pred.category == gt_world.category
            ]

            if len(same_class_preds) > 0:
                preds_center = [
                    pred.box3d.location for pred in same_class_preds
                ]
                distance_matrix = cdist(
                    np.array(gt_world.box3d.location).reshape(1, 3),
                    np.array(preds_center).reshape(-1, 3),
                )[0]

                if distance_matrix[distance_matrix.argmin()] < 2:
                    return Label(
                        id=gt_world.id,
                        category=same_class_preds[
                            distance_matrix.argmin()
                        ].category,
                        box3d=same_class_preds[distance_matrix.argmin()].box3d,
                        score=same_class_preds[distance_matrix.argmin()].score,
                    )

        return gt_world

    def generate_trajectories(
        self, pure_detection_path: str, annotations_path: str, cache_path: str
    ) -> List[Dict[str, List[Label]]]:
        """Generate trajectories data."""
        with open(pure_detection_path, "rb") as f:
            pure_detection_result = json.load(f)

        dataset = load(annotations_path, nprocs=4)
        predictions = ScalabelDataset(frames=pure_detection_result["frames"])

        group_idx = [
            i for i, f in enumerate(dataset.groups) if f.frameIndex == 0
        ]

        total_traj = []
        for (i, _) in enumerate(group_idx):
            first_frame = group_idx[i]
            if i == len(group_idx) - 1:
                last_frame = len(dataset.groups)
            else:
                last_frame = group_idx[i + 1]

            frames = dataset.groups[first_frame:last_frame]
            preds = predictions.frames[first_frame:last_frame]

            local_traj = {}
            for j, frame in enumerate(frames):
                if frame.labels is not None:
                    assert frame.extrinsics is not None
                    sensor2global = R.from_euler(
                        "xyz", frame.extrinsics.rotation
                    ).as_matrix()
                    for label in frame.labels:
                        if label.id not in local_traj:
                            local_traj[label.id] = {"gt": [], "pred": []}

                        assert label.box3d is not None
                        translation = np.dot(
                            sensor2global, label.box3d.location
                        ) + np.array(frame.extrinsics.location)

                        # Using extrinsic rotation here to align with Pytorch3D
                        rotation = R.from_euler(
                            "XYZ", label.box3d.orientation
                        ).as_matrix()
                        rotation = np.dot(sensor2global, rotation)
                        rotation = R.from_matrix(rotation).as_euler("XYZ")

                        xyz = tuple(translation.tolist())
                        rotx, roty, rotz = rotation.tolist()

                        gt_world = Label(
                            id=label.id,
                            category=label.category,
                            box3d=Box3D(
                                location=xyz,
                                dimension=label.box3d.dimension,
                                orientation=(rotx, roty, rotz),
                                alpha=rotation_y_to_alpha(roty, xyz),
                            ),
                            score=1.0,
                        )
                        local_traj[label.id]["gt"].append(gt_world)

                        matched_pred = self.gt_pred_matching(
                            gt_world, preds[j].labels
                        )
                        local_traj[label.id]["pred"].append(matched_pred)
                else:
                    tid_list = list(local_traj.keys())
                    for tid in tid_list:
                        if len(local_traj[tid]["gt"]) >= self.min_seq_len:
                            total_traj.append(local_traj[tid])
                        local_traj.pop(tid)

            tid_list = list(local_traj.keys())
            for tid in tid_list:
                if len(local_traj[tid]["gt"]) >= self.min_seq_len:
                    total_traj.append(local_traj[tid])
                local_traj.pop(tid)

        with open(cache_path, "wb") as f:
            pickle.dump(total_traj, f)

        return total_traj
