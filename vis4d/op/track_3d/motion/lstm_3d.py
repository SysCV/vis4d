"""LSTM 3D motion model."""
from typing import Tuple

import numpy as np
import torch
from torch import nn

from vis4d.op.geometry.rotation import normalize_angle, acute_angle
from vis4d.common import ArgsType

from .base import BaseMotionModel


class LSTM3DMotionModel(BaseMotionModel):
    """LSTM 3D motion model."""

    def __init__(
        self,
        lstm_model: nn.Module,
        detections_3d: torch.Tensor,
        *args: ArgsType,
        init_flag: bool = True,
        **kwargs: ArgsType,
    ) -> None:
        """Initialize a motion model using initial bounding box."""
        super().__init__(*args, **kwargs)
        self.init_flag = init_flag

        self.device = detections_3d.device
        self.lstm_model = lstm_model.to(self.device)
        self.lstm_model.eval()

        bbox_3d = detections_3d[: self.motion_dims]
        info = detections_3d[self.motion_dims :]

        self.obj_state = torch.cat([bbox_3d, bbox_3d.new_zeros(3)])
        self.history = bbox_3d.new_zeros(self.num_frames, self.motion_dims)
        self.ref_history = torch.cat(
            [bbox_3d.view(1, self.motion_dims)] * (self.num_frames + 1)
        )
        self.prev_ref = bbox_3d.clone()
        self.info = info
        self.hidden_pred = self.lstm_model.init_hidden(
            self.device, batch_size=1
        )
        self.hidden_ref = self.lstm_model.init_hidden(
            self.device, batch_size=1
        )

    def _update_history(self, bbox_3d: torch.Tensor) -> None:
        """Update velocity history."""
        self.ref_history = self.update_array(self.ref_history, bbox_3d)
        self.history = self.update_array(
            self.history, self.ref_history[-1] - self.ref_history[-2]
        )
        self.prev_ref[: self.motion_dims] = self.obj_state[: self.motion_dims]

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

    def update(self, obs_3d: torch.Tensor) -> None:  # type: ignore
        """Updates the state vector with observed bbox."""
        bbox_3d = obs_3d[: self.motion_dims]
        info = obs_3d[self.motion_dims :]

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        if self.age == 1:
            self.obj_state[: self.motion_dims] = bbox_3d.clone()

        self.obj_state[6] = normalize_angle(self.obj_state[6])
        bbox_3d[6] = normalize_angle(bbox_3d[6])

        # acute angle
        self.obj_state[6] = acute_angle(self.obj_state[6], bbox_3d[6])

        with torch.no_grad():
            refined_loc, self.hidden_ref = self.lstm_model.refine(
                self.obj_state[: self.motion_dims].view(1, self.motion_dims),
                bbox_3d.view(1, self.motion_dims),
                self.prev_ref.view(1, self.motion_dims),
                info.view(1, 1),
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

        self.info = info

    def predict_velocity(self) -> torch.Tensor:  # type: ignore
        """Predict velocity."""
        with torch.no_grad():
            pred_loc, _ = self.lstm_model.predict(
                self.history[..., : self.motion_dims].view(
                    self.num_frames, -1, self.motion_dims
                ),
                self.obj_state[: self.motion_dims],
                self.hidden_pred,
            )
        return pred_loc[0][:3] - self.prev_ref[:3]

    def predict(self, update_state: bool = True) -> torch.Tensor:  # type: ignore # pylint: disable=line-too-long
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

    def get_state(self) -> torch.Tensor:  # type: ignore
        """Returns the current bounding box estimate."""
        return self.obj_state


class VeloLSTM(nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Estimating object location in world coordinates.

    Prediction LSTM:
        Input: 5 frames velocity
        Output: Next frame location
    Updating LSTM:
        Input: predicted location and observed location
        Output: Refined location
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        num_layers: int,
        loc_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Init."""
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loc_dim = loc_dim

        self.vel2feat = nn.Linear(
            loc_dim,
            feature_dim,
        )

        self.pred_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.pred2atten = nn.Linear(
            hidden_size,
            loc_dim,
            bias=False,
        )

        self.conf2feat = nn.Linear(
            1,
            feature_dim,
            bias=False,
        )

        self.refine_lstm = nn.LSTM(
            input_size=3 * feature_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.conf2atten = nn.Linear(
            hidden_size,
            loc_dim,
            bias=False,
        )

        self._init_param()

    def init_hidden(
        self, device: str, batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializae hidden state.

        The axes semantics are (num_layers, minibatch_size, hidden_dim)
        """
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                device
            ),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                device
            ),
        )

    def _init_param(self) -> None:
        """Initialize parameters."""
        init_module(self.vel2feat)
        init_module(self.pred2atten)
        init_module(self.conf2feat)
        init_module(self.conf2atten)
        init_lstm_module(self.pred_lstm)
        init_lstm_module(self.refine_lstm)

    def refine(
        self,
        location: torch.Tensor,
        observation: torch.Tensor,
        prev_location: torch.Tensor,
        confidence: torch.Tensor,
        hc_0: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Refine predicted location using single frame estimation at t+1.

        Input:
            location: (num_batch x loc_dim), location from prediction
            observation: (num_batch x loc_dim), location from single frame
            estimation
            prev_location: (num_batch x loc_dim), refined location
            confidence: (num_batch X 1), depth estimation confidence
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and
            cell
        Middle:
            loc_embed: (1, num_batch x feature_dim), predicted location feature
            obs_embed: (1, num_batch x feature_dim), single frame location
            feature
            conf_embed: (1, num_batch x feature_dim), depth estimation
            confidence feature
            embed: (1, num_batch x 2*feature_dim), location feature
            out: (1 x num_batch x hidden_size), lstm output
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated
            hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        """
        num_batch = location.shape[0]

        pred_vel = location - prev_location
        obsv_vel = observation - prev_location

        # Embed feature to hidden_size
        loc_embed = self.vel2feat(pred_vel).view(num_batch, self.feature_dim)
        obs_embed = self.vel2feat(obsv_vel).view(num_batch, self.feature_dim)
        conf_embed = self.conf2feat(confidence).view(
            num_batch, self.feature_dim
        )
        embed = torch.cat(
            [
                loc_embed,
                obs_embed,
                conf_embed,
            ],
            dim=1,
        ).view(1, num_batch, 3 * self.feature_dim)

        out, (h_n, c_n) = self.refine_lstm(embed, hc_0)

        delta_vel_atten = torch.sigmoid(self.conf2atten(out)).view(
            num_batch, self.loc_dim
        )

        output_pred = (
            delta_vel_atten * obsv_vel
            + (1.0 - delta_vel_atten) * pred_vel
            + prev_location
        )

        return output_pred, (h_n, c_n)

    def predict(
        self,
        vel_history: torch.Tensor,
        location: torch.Tensor,
        hc_0: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict location at t+1 using updated location at t.

        Input:
            vel_history: (num_seq, num_batch, loc_dim), velocity from previous
            num_seq updates
            location: (num_batch, loc_dim), location from previous update
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and
            cell
        Middle:
            embed: (num_seq, num_batch x feature_dim), location feature
            out: (num_seq x num_batch x hidden_size), lstm output
            attention_logit: (num_seq x num_batch x loc_dim), the predicted
            residual
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated
            hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        """
        num_seq, num_batch, _ = vel_history.shape

        # Embed feature to hidden_size
        embed = self.vel2feat(vel_history).view(
            num_seq, num_batch, self.feature_dim
        )

        out, (h_n, c_n) = self.pred_lstm(embed, hc_0)

        attention_logit = self.pred2atten(out).view(
            num_seq, num_batch, self.loc_dim
        )
        attention = torch.softmax(attention_logit, dim=0)

        output_pred = torch.sum(attention * vel_history, dim=0) + location

        return output_pred, (h_n, c_n)


def init_module(layer: nn.Module) -> None:
    """Initialize modules weights and biases."""
    for m in layer.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


def init_lstm_module(layer: nn.Module) -> None:
    """Initialize LSTM weights and biases."""
    for name, param in layer.named_parameters():
        if "weight_ih" in name:
            torch.nn.init.xavier_uniform_(param.data)
        elif "weight_hh" in name:
            torch.nn.init.orthogonal_(param.data)
        elif "bias" in name:
            param.data.fill_(0)
