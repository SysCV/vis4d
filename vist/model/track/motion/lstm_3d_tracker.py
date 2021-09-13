"""LSTM motion model tracker."""
import numpy as np
import torch

from .base import BaseMotionTracker, MotionTrackerConfig


class LSTM3DMotionTrackerConfig(MotionTrackerConfig):
    """VeloLSTM 3D motion model tracker config."""

    init_flag: bool = True


class LSTM3DMotionTracker(BaseMotionTracker):
    """LSTM 3D motion model tracker."""

    def __init__(self, device, cfg, detections_3d):
        """
        Initialises a motion model tracker using initial bounding box.

        Args:
            device: cpu / cuda.
            lstm: lstm model.
            detections_3d: x, y, z, h, w, l, ry, depth uncertainty
        """
        self.cfg = LSTM3DMotionTrackerConfig(**cfg.dict())

        # Init VeloLSTM
        self.device = device
        self.lstm = self.cfg.lstm.to(device)
        self.lstm.eval()

        self.loc_dim = self.cfg.loc_dim
        self.id = LSTM3DMotionTracker.count
        LSTM3DMotionTracker.count += 1
        self.nfr = 5
        self.hits = 1
        self.hit_streak = 0
        self.time_since_update = 0
        self.init_flag = True
        self.age = 0

        bbox3D = detections_3d[: self.loc_dim]
        info = detections_3d[self.loc_dim :]

        self.obj_state = np.hstack([bbox3D.reshape((7,)), np.zeros((3,))])
        self.history = np.tile(np.zeros_like(bbox3D), (self.nfr, 1))
        self.ref_history = np.tile(bbox3D, (self.nfr + 1, 1))
        self.avg_angle = bbox3D[6]
        self.avg_dim = np.array(bbox3D[3:6])
        self.prev_obs = bbox3D.copy()
        self.prev_ref = bbox3D.copy()
        self.info = info
        self.hidden_pred = self.lstm.init_hidden(self.device)
        self.hidden_ref = self.lstm.init_hidden(self.device)

    @staticmethod
    def fix_alpha(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def update_array(
        origin_array: np.ndarray, input_array: np.ndarray
    ) -> np.ndarray:
        new_array = origin_array.copy()
        new_array[:-1] = origin_array[1:]
        new_array[-1:] = input_array
        return new_array

    def _update_history(self, bbox3D):
        self.ref_history = self.update_array(self.ref_history, bbox3D)
        self.history = self.update_array(
            self.history, self.ref_history[-1] - self.ref_history[-2]
        )
        # align orientation history
        self.history[:, 3] = self.history[-1, 3]
        self.prev_ref[: self.loc_dim] = self.obj_state[: self.loc_dim]
        if self.loc_dim > 3:
            self.avg_angle = self.fix_alpha(self.ref_history[:, 3]).mean(
                axis=0
            )
            self.avg_dim = self.ref_history.mean(axis=0)[4:]
        else:
            self.avg_angle = self.prev_obs[3]
            self.avg_dim = np.array(self.prev_obs[4:])

    def _init_history(self, bbox3D):
        self.ref_history = self.update_array(self.ref_history, bbox3D)
        self.history = np.tile(
            [self.ref_history[-1] - self.ref_history[-2]], (self.nfr, 1)
        )
        self.prev_ref[: self.loc_dim] = self.obj_state[: self.loc_dim]
        if self.loc_dim > 3:
            self.avg_angle = self.fix_alpha(self.ref_history[:, 3]).mean(
                axis=0
            )
            self.avg_dim = self.ref_history.mean(axis=0)[4:]
        else:
            self.avg_angle = self.prev_obs[3]
            self.avg_dim = np.array(self.prev_obs[4:])

    def update(self, detections_3d):
        """
        Updates the state vector with observed bbox.
        """
        bbox3D = detections_3d[: self.loc_dim]
        info = detections_3d[self.loc_dim :]

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        if self.age == 1:
            self.obj_state[: self.loc_dim] = bbox3D.copy()

        if self.loc_dim > 3:
            # orientation correction
            self.obj_state[6] = self.fix_alpha(self.obj_state[6])
            bbox3D[6] = self.fix_alpha(bbox3D[6])

            # if the angle of two theta is not acute angle
            # make the theta still in the range
            curr_yaw = bbox3D[6]
            if (
                np.pi / 2.0
                < abs(curr_yaw - self.obj_state[6])
                < np.pi * 3 / 2.0
            ):
                self.obj_state[6] += np.pi
                if self.obj_state[6] > np.pi:
                    self.obj_state[6] -= np.pi * 2
                if self.obj_state[6] < -np.pi:
                    self.obj_state[6] += np.pi * 2

            # now the angle is acute: < 90 or > 270,
            # convert the case of > 270 to < 90
            if abs(curr_yaw - self.obj_state[6]) >= np.pi * 3 / 2.0:
                if curr_yaw > 0:
                    self.obj_state[6] += np.pi * 2
                else:
                    self.obj_state[6] -= np.pi * 2

        with torch.no_grad():
            refined_loc, self.hidden_ref = self.lstm.refine(
                torch.from_numpy(self.obj_state[: self.loc_dim])
                .view(1, self.loc_dim)
                .float()
                .to(self.device),
                torch.from_numpy(bbox3D)
                .view(1, self.loc_dim)
                .float()
                .to(self.device),
                torch.from_numpy(self.prev_ref)
                .view(1, self.loc_dim)
                .float()
                .to(self.device),
                torch.from_numpy(info).view(1, 1).float().to(self.device),
                self.hidden_ref,
            )

        refined_obj = refined_loc.cpu().numpy().flatten()
        if self.loc_dim > 3:
            refined_obj[6] = self.fix_alpha(refined_obj[6])

        self.obj_state[: self.loc_dim] = refined_obj
        self.prev_obs = bbox3D

        if np.pi / 2.0 < abs(bbox3D[6] - self.avg_angle) < np.pi * 3 / 2.0:
            for r_indx in range(len(self.ref_history)):
                self.ref_history[r_indx][6] = self.fix_alpha(
                    self.ref_history[r_indx][6] + np.pi
                )

        if self.init_flag:
            self._init_history(refined_obj)
            self.init_flag = False
        else:
            self._update_history(refined_obj)

        self.info = info

    def predict(self, update_state: bool = True):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        with torch.no_grad():
            pred_loc, hidden_pred = self.lstm.predict(
                torch.from_numpy(self.history[..., : self.loc_dim])
                .view(self.nfr, -1, self.loc_dim)
                .float()
                .to(self.device),
                torch.from_numpy(self.obj_state[: self.loc_dim])
                .view(-1, self.loc_dim)
                .float()
                .to(self.device),
                self.hidden_pred,
            )

        pred_state = self.obj_state.copy()
        pred_state[: self.loc_dim] = pred_loc.cpu().numpy().flatten()
        pred_state[7:] = pred_state[:3] - self.prev_ref[:3]
        if self.loc_dim > 3:
            pred_state[6] = self.fix_alpha(pred_state[6])

        if update_state:
            self.hidden_pred = hidden_pred
            self.obj_state = pred_state

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return pred_state.flatten()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.obj_state.flatten()

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history
