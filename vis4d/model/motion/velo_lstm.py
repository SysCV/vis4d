"""VeloLSTM 3D motion model."""
from __future__ import annotations

import torch
from torch import nn, Tensor

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.op.layer.weight_init import xavier_init


class VeloLSTM(nn.Module):
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
        feature_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        loc_dim: int = 7,
        dropout: float = 0.1,
        weights: str | None = None,
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

        self._init_weights()

        if weights is not None:
            load_model_checkpoint(self, weights, map_location="cpu")

    def _init_weights(self) -> None:
        """Initialize model weights."""
        xavier_init(self.vel2feat)
        xavier_init(self.pred2atten)
        xavier_init(self.conf2feat)
        xavier_init(self.conf2atten)
        init_lstm_module(self.pred_lstm)
        init_lstm_module(self.refine_lstm)

    def init_hidden(
        self, device: torch.device, batch_size: int = 1
    ) -> tuple[Tensor, Tensor]:
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

    def refine(
        self,
        location: Tensor,
        observation: Tensor,
        prev_location: Tensor,
        confidence: Tensor,
        hc_0: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
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
        vel_history: Tensor,
        location: Tensor,
        hc_0: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
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


def init_lstm_module(layer: nn.Module) -> None:
    """Initialize LSTM weights and biases."""
    for name, param in layer.named_parameters():
        if "weight_ih" in name:
            torch.nn.init.xavier_uniform_(param.data)
        elif "weight_hh" in name:
            torch.nn.init.orthogonal_(param.data)
        elif "bias" in name:
            param.data.fill_(0)
