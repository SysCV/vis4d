"""LSTM 3D motion model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vis4d.common import ArgsType
from vis4d.model.motion.velo_lstm import VeloLSTM
from vis4d.op.geometry.rotation import acute_angle, normalize_angle

from .base import BaseMotionModel, update_array


class LSTM3DMotionModel(BaseMotionModel):
    """LSTM 3D motion model."""

    def __init__(
        self,
        *args: ArgsType,
        lstm_model: nn.Module,
        obs_3d: Tensor,
        init_flag: bool = True,
        **kwargs: ArgsType,
    ) -> None:
        """Initialize a motion model using initial bounding box."""
        super().__init__(*args, **kwargs)
        self.init_flag = init_flag
        self.device = obs_3d.device

        assert isinstance(
            lstm_model, VeloLSTM
        ), "Currently only support VeloLSTM motion model!"
        self.lstm_model = lstm_model
        self.lstm_model.to(self.device)
        self.lstm_model.eval()

        self.obj_state = torch.cat([obs_3d, obs_3d.new_zeros(3)])
        self.history = obs_3d.new_zeros(self.num_frames, self.motion_dims)
        self.ref_history = torch.cat(
            [obs_3d.view(1, self.motion_dims)] * (self.num_frames + 1)
        )
        self.prev_ref = obs_3d.clone()
        self.hidden_pred = self.lstm_model.init_hidden(
            self.device, batch_size=1
        )
        self.hidden_ref = self.lstm_model.init_hidden(
            self.device, batch_size=1
        )

    def _update_history(self, bbox_3d: Tensor) -> None:
        """Update velocity history."""
        self.ref_history = update_array(self.ref_history, bbox_3d)
        self.history = update_array(
            self.history, self.ref_history[-1] - self.ref_history[-2]
        )
        self.prev_ref[: self.motion_dims] = self.obj_state[: self.motion_dims]

    def _init_history(self, bbox_3d: Tensor) -> None:
        """Initialize velocity history."""
        self.ref_history = update_array(self.ref_history, bbox_3d)
        self.history = torch.cat(
            [
                (self.ref_history[-1] - self.ref_history[-2]).view(
                    1, self.motion_dims
                )
            ]
            * self.num_frames
        )
        self.prev_ref[: self.motion_dims] = self.obj_state[: self.motion_dims]

    def update(self, obs_3d: Tensor, info: Tensor) -> None:
        """Updates the state vector with observed bbox."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        if self.age == 1:
            self.obj_state[: self.motion_dims] = obs_3d.clone()

        self.obj_state[6] = normalize_angle(self.obj_state[6])
        obs_3d[6] = normalize_angle(obs_3d[6])

        # acute angle
        self.obj_state[6] = acute_angle(self.obj_state[6], obs_3d[6])

        with torch.no_grad():
            refined_loc, self.hidden_ref = self.lstm_model.refine(
                self.obj_state[: self.motion_dims].unsqueeze(0),
                obs_3d.unsqueeze(0),
                self.prev_ref.unsqueeze(0),
                info.unsqueeze(0).unsqueeze(0),
                self.hidden_ref,
            )

        refined_obj = refined_loc.view(self.motion_dims)
        refined_obj[6] = normalize_angle(refined_obj[6])

        self.obj_state[: self.motion_dims] = refined_obj

        if self.init_flag:
            self._init_history(refined_obj)
            self.init_flag = False
        else:
            self._update_history(refined_obj)

    def predict_velocity(self) -> Tensor:
        """Predict velocity."""
        with torch.no_grad():
            pred_loc, _ = self.lstm_model.predict(
                self.history[..., : self.motion_dims].view(
                    self.num_frames, -1, self.motion_dims
                ),
                self.obj_state[: self.motion_dims],
                self.hidden_pred,
            )
        return (pred_loc[0][:3] - self.prev_ref[:3]) * self.fps

    def predict(self, update_state: bool = True) -> Tensor:
        """Advances the state vector and returns the predicted bounding box."""
        with torch.no_grad():
            pred_loc, hidden_pred = self.lstm_model.predict(
                self.history[..., : self.motion_dims].view(
                    self.num_frames, -1, self.motion_dims
                ),
                self.obj_state[: self.motion_dims],
                self.hidden_pred,
            )

        pred_state = self.obj_state.clone()
        pred_state[: self.motion_dims] = pred_loc.view(self.motion_dims)
        pred_state[self.motion_dims :] = pred_state[:3] - self.prev_ref[:3]

        pred_state[6] = normalize_angle(pred_state[6])

        if update_state:
            self.hidden_pred = hidden_pred
            self.obj_state = pred_state

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return pred_state

    def get_state(self) -> Tensor:
        """Returns the current bounding box estimate."""
        return self.obj_state
