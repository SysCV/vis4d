"""LSTM 3D motion model."""
import numpy as np
import torch
from torch import nn

from vis4d.struct import ArgsType

from .base import BaseMotionModel


class LSTM3DMotionModel(BaseMotionModel):
    """LSTM 3D motion model."""

    def __init__(
        self,
        lstm: nn.Module,
        detections_3d: torch.Tensor,
        *args: ArgsType,
        init_flag: bool = True,
        **kwargs: ArgsType,
    ) -> None:
        """Initialize a motion model using initial bounding box."""
        super().__init__(*args, **kwargs)
        self.init_flag = init_flag

        self.device = detections_3d.device
        self.lstm = lstm.to(self.device)
        self.lstm.eval()

        self.pi = torch.tensor(np.pi).to(self.device)

        bbox_3d = detections_3d[: self.motion_dims]
        info = detections_3d[self.motion_dims :]

        self.obj_state = torch.cat([bbox_3d, bbox_3d.new_zeros(3)])
        self.history = bbox_3d.new_zeros(self.num_frames, self.motion_dims)
        self.ref_history = torch.cat(
            [bbox_3d.view(1, self.motion_dims)] * (self.num_frames + 1)
        )
        self.avg_angle = bbox_3d[6]
        self.avg_dim = bbox_3d[3:6]
        self.prev_obs = bbox_3d.clone()
        self.prev_ref = bbox_3d.clone()
        self.info = info
        self.hidden_pred = self.lstm.init_hidden(self.device)
        self.hidden_ref = self.lstm.init_hidden(self.device)

    def fix_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Fix the angle value."""
        return (angle + self.pi) % (2 * self.pi) - self.pi

    def _update_history(self, bbox_3d: torch.Tensor) -> None:
        """Update velocity history."""
        self.ref_history = self.update_array(self.ref_history, bbox_3d)
        self.history = self.update_array(
            self.history, self.ref_history[-1] - self.ref_history[-2]
        )
        # align orientation history
        self.history[:, 3] = self.history[-1, 3]
        self.prev_ref[: self.motion_dims] = self.obj_state[: self.motion_dims]
        if self.motion_dims > 3:
            self.avg_angle = self.fix_angle(self.ref_history[:, 3]).mean(
                axis=0
            )
            self.avg_dim = self.ref_history.mean(axis=0)[3:6]
        else:
            self.avg_angle = self.prev_obs[6]
            self.avg_dim = self.prev_obs[3:6]

    def _init_history(self, bbox_3d: torch.Tensor) -> None:
        """Initialize velocity history."""
        self.ref_history = self.update_array(self.ref_history, bbox_3d)
        self.history = torch.cat(
            [
                (self.ref_history[-1] - self.ref_history[-2]).view(
                    1, self.motion_dims
                )
            ]
            * self.num_frames
        )
        self.prev_ref[: self.motion_dims] = self.obj_state[: self.motion_dims]
        if self.motion_dims > 3:
            self.avg_angle = self.fix_angle(self.ref_history[:, 3]).mean(
                axis=0
            )
            self.avg_dim = self.ref_history.mean(axis=0)[3:6]
        else:
            self.avg_angle = self.prev_obs[6]
            self.avg_dim = self.prev_obs[3:6]

    def update(self, obs_3d: torch.Tensor) -> None:  # type: ignore
        """Updates the state vector with observed bbox."""
        bbox_3d = obs_3d[: self.motion_dims]
        info = obs_3d[self.motion_dims :]

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        if self.age == 1:
            self.obj_state[: self.motion_dims] = bbox_3d.clone()

        if self.motion_dims > 3:
            # orientation correction
            self.obj_state[6] = self.fix_angle(self.obj_state[6])
            bbox_3d[6] = self.fix_angle(bbox_3d[6])

            # if the angle of two theta is not acute angle
            # make the theta still in the range
            curr_yaw = bbox_3d[6]
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
                self.obj_state[: self.motion_dims].view(1, self.motion_dims),
                bbox_3d.view(1, self.motion_dims),
                self.prev_ref.view(1, self.motion_dims),
                info.view(1, 1),
                self.hidden_ref,
            )

        refined_obj = refined_loc.view(self.motion_dims)
        if self.motion_dims > 3:
            refined_obj[6] = self.fix_angle(refined_obj[6])

        self.obj_state[: self.motion_dims] = refined_obj
        self.prev_obs = bbox_3d

        if (
            self.pi / 2.0
            < abs(bbox_3d[6] - self.avg_angle)
            < self.pi * 3 / 2.0
        ):
            for r_indx, _ in enumerate(self.ref_history):
                self.ref_history[r_indx][6] = self.fix_angle(
                    self.ref_history[r_indx][6] + self.pi
                )

        if self.init_flag:
            self._init_history(refined_obj)
            self.init_flag = False
        else:
            self._update_history(refined_obj)

        self.info = info

    def predict(self, update_state: bool = True) -> torch.Tensor:  # type: ignore # pylint: disable=line-too-long
        """Advances the state vector and returns the predicted bounding box."""
        with torch.no_grad():
            pred_loc, hidden_pred = self.lstm.predict(
                self.history[..., : self.motion_dims].view(
                    self.num_frames, -1, self.motion_dims
                ),
                self.obj_state[: self.motion_dims],
                self.hidden_pred,
            )

        pred_state = self.obj_state.clone()
        pred_state[: self.motion_dims] = pred_loc.view(self.motion_dims)
        pred_state[7:] = pred_state[:3] - self.prev_ref[:3]
        if self.motion_dims > 3:
            pred_state[6] = self.fix_angle(pred_state[6])

        if update_state:
            self.hidden_pred = hidden_pred
            self.obj_state = pred_state

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return pred_state

    def get_state(self) -> torch.Tensor:  # type: ignore
        """Returns the current bounding box estimate."""
        return self.obj_state

    def get_history(self) -> torch.Tensor:  # type: ignore
        """Returns the history of estimates."""
        return self.history
