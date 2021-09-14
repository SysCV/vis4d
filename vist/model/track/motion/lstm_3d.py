"""LSTM 3D motion model."""
import torch
import numpy as np

from .base import BaseMotionModel, MotionModelConfig


class LSTM3DMotionModelConfig(MotionModelConfig):
    """VeloLSTM 3D motion model config."""

    lstm_ckpt_name: str
    init_flag: bool = True


class LSTM3DMotionModel(BaseMotionModel):
    """LSTM 3D motion model."""

    def __init__(self, cfg, detections_3d):
        """
        Initialises a motion model tracker using initial bounding box.

        Args:
            cfg: motion tracker config.
            detections_3d: x, y, z, h, w, l, ry, depth uncertainty
        """
        self.cfg = LSTM3DMotionModelConfig(**cfg.dict())

        self.device = detections_3d.device
        self.lstm = self.cfg.lstm.to(self.device)
        self.lstm.eval()

        self.pi = torch.tensor(np.pi).to(self.device)

        bbox3D = detections_3d[: self.cfg.motion_dims]
        info = detections_3d[self.cfg.motion_dims :]

        self.obj_state = torch.cat([bbox3D, bbox3D.new_zeros(3)])
        self.history = bbox3D.new_zeros(
            self.cfg.num_frames, self.cfg.motion_dims
        )
        self.ref_history = torch.cat(
            [bbox3D.view(1, self.cfg.motion_dims)] * (self.cfg.num_frames + 1)
        )
        self.avg_angle = bbox3D[6]
        self.avg_dim = bbox3D[3:6]
        self.prev_obs = bbox3D.clone()
        self.prev_ref = bbox3D.clone()
        self.info = info
        self.hidden_pred = self.lstm.init_hidden(self.device)
        self.hidden_ref = self.lstm.init_hidden(self.device)

    def fix_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Fix the angle value."""
        return (angle + self.pi) % (2 * self.pi) - self.pi

    def _update_history(self, bbox3D):
        self.ref_history = self.update_array(self.ref_history, bbox3D)
        self.history = self.update_array(
            self.history, self.ref_history[-1] - self.ref_history[-2]
        )
        # align orientation history
        self.history[:, 3] = self.history[-1, 3]
        self.prev_ref[: self.cfg.motion_dims] = self.obj_state[
            : self.cfg.motion_dims
        ]
        if self.cfg.motion_dims > 3:
            self.avg_angle = self.fix_angle(self.ref_history[:, 3]).mean(
                axis=0
            )
            self.avg_dim = self.ref_history.mean(axis=0)[3:6]
        else:
            self.avg_angle = self.prev_obs[6]
            self.avg_dim = self.prev_obs[3:6]

    def _init_history(self, bbox3D):
        self.ref_history = self.update_array(self.ref_history, bbox3D)
        self.history = torch.cat(
            [
                (self.ref_history[-1] - self.ref_history[-2]).view(
                    1, self.cfg.motion_dims
                )
            ]
            * self.cfg.num_frames
        )
        self.prev_ref[: self.cfg.motion_dims] = self.obj_state[
            : self.cfg.motion_dims
        ]
        if self.cfg.motion_dims > 3:
            self.avg_angle = self.fix_angle(self.ref_history[:, 3]).mean(
                axis=0
            )
            self.avg_dim = self.ref_history.mean(axis=0)[3:6]
        else:
            self.avg_angle = self.prev_obs[6]
            self.avg_dim = self.prev_obs[3:6]

    def update(self, detections_3d):
        """
        Updates the state vector with observed bbox.
        """
        bbox3D = detections_3d[: self.cfg.motion_dims]
        info = detections_3d[self.cfg.motion_dims :]

        self.cfg.time_since_update = 0
        self.cfg.hits += 1
        self.cfg.hit_streak += 1

        if self.cfg.age == 1:
            self.obj_state[: self.cfg.motion_dims] = bbox3D.clone()

        if self.cfg.motion_dims > 3:
            # orientation correction
            self.obj_state[6] = self.fix_angle(self.obj_state[6])
            bbox3D[6] = self.fix_angle(bbox3D[6])

            # if the angle of two theta is not acute angle
            # make the theta still in the range
            curr_yaw = bbox3D[6]
            if (
                self.pi / 2.0
                < abs(curr_yaw - self.obj_state[6])
                < self.pi * 3 / 2.0
            ):
                self.obj_state[6] += self.pi
                if self.obj_state[6] > self.pi:
                    self.obj_state[6] -= self.pi * 2
                if self.obj_state[6] < -self.pi:
                    self.obj_state[6] += self.pi * 2

            # now the angle is acute: < 90 or > 270,
            # convert the case of > 270 to < 90
            if abs(curr_yaw - self.obj_state[6]) >= self.pi * 3 / 2.0:
                if curr_yaw > 0:
                    self.obj_state[6] += self.pi * 2
                else:
                    self.obj_state[6] -= self.pi * 2

        with torch.no_grad():
            refined_loc, self.hidden_ref = self.lstm.refine(
                self.obj_state[: self.cfg.motion_dims].view(
                    1, self.cfg.motion_dims
                ),
                bbox3D.view(1, self.cfg.motion_dims),
                self.prev_ref.view(1, self.cfg.motion_dims),
                info.view(1, 1),
                self.hidden_ref,
            )

        refined_obj = refined_loc.view(self.cfg.motion_dims)
        if self.cfg.motion_dims > 3:
            refined_obj[6] = self.fix_angle(refined_obj[6])

        self.obj_state[: self.cfg.motion_dims] = refined_obj
        self.prev_obs = bbox3D

        if self.pi / 2.0 < abs(bbox3D[6] - self.avg_angle) < self.pi * 3 / 2.0:
            for r_indx in range(len(self.ref_history)):
                self.ref_history[r_indx][6] = self.fix_angle(
                    self.ref_history[r_indx][6] + self.pi
                )

        if self.cfg.init_flag:
            self._init_history(refined_obj)
            self.cfg.init_flag = False
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
                self.history[..., : self.cfg.motion_dims].view(
                    self.cfg.num_frames, -1, self.cfg.motion_dims
                ),
                self.obj_state[: self.cfg.motion_dims],
                self.hidden_pred,
            )

        pred_state = self.obj_state.clone()
        pred_state[: self.cfg.motion_dims] = pred_loc.view(
            self.cfg.motion_dims
        )
        pred_state[7:] = pred_state[:3] - self.prev_ref[:3]
        if self.cfg.motion_dims > 3:
            pred_state[6] = self.fix_angle(pred_state[6])

        if update_state:
            self.hidden_pred = hidden_pred
            self.obj_state = pred_state

        self.cfg.age += 1
        if self.cfg.time_since_update > 0:
            self.cfg.hit_streak = 0
        self.cfg.time_since_update += 1

        return pred_state

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.obj_state

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history
