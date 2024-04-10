"""Exponential Moving Average (EMA) for PyTorch models."""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn

from vis4d.common.logging import rank_zero_info


class ModelEMAAdapter(nn.Module):
    """Torch module with Exponential Moving Average (EMA).

    Args:
        model (nn.Module): Model to apply EMA.
        decay (float): Decay factor for EMA. Defaults to 0.9998.
        use_ema_during_test (bool): Use EMA model during testing. Defaults to
            True.
        device (torch.device | None): Device to use. Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9998,
        use_ema_during_test: bool = True,
        device: torch.device | None = None,
    ):
        """Init ModelEMAAdapter class."""
        super().__init__()
        self.model = model
        self.ema_model = deepcopy(self.model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.use_ema_during_test = use_ema_during_test
        self.device = device
        if self.device is not None:
            self.ema_model.to(device=device)
        rank_zero_info("Using model EMA with decay rate %f", self.decay)

    def _update(
        self, model: nn.Module, update_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> None:
        """Update model params."""
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.ema_model.state_dict().values(),
                model.state_dict().values(),
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, steps: int) -> None:  # pylint: disable=unused-argument
        """Update the internal EMA model."""
        self._update(
            self.model,
            update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m,
        )

    def set(self, model: nn.Module) -> None:
        """Copy model params into the internal EMA."""
        self._update(model, update_fn=lambda e, m: m)

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore
        """Forward pass with original model."""
        if self.training or not self.use_ema_during_test:
            return self.model(*args, **kwargs)
        return self.ema_model(*args, **kwargs)


class ModelExpEMAAdapter(ModelEMAAdapter):
    """Exponential Moving Average (EMA) with exponential decay strategy.

    Used by YOLOX.

    Args:
        model (nn.Module): Model to apply EMA.
        decay (float): Decay factor for EMA. Defaults to 0.9998.
        warmup_steps (int): Number of warmup steps for decay. Use a smaller
            decay early in training and gradually anneal to the set decay value
            to update the EMA model smoothly.
        use_ema_during_test (bool): Use EMA model during testing. Defaults to
            True.
        device (torch.device | None): Device to use. Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9998,
        warmup_steps: int = 2000,
        use_ema_during_test: bool = True,
        device: torch.device | None = None,
    ):
        """Init ModelEMAAdapter class."""
        super().__init__(model, decay, use_ema_during_test, device)
        assert (
            warmup_steps > 0
        ), f"warmup_steps must be greater than 0, got {warmup_steps}"
        self.warmup_steps = warmup_steps

    def update(self, steps: int) -> None:
        """Update the internal EMA model."""
        decay = self.decay * (
            1 - math.exp(-float(1 + steps) / self.warmup_steps)
        )
        self._update(
            self.model,
            update_fn=lambda e, m: decay * e + (1.0 - decay) * m,
        )
