"""Config definitions."""

from typing import List

from pydantic import BaseModel


class Solver(BaseModel):
    """Config for solver."""

    images_per_batch: int
    lr_policy: str
    base_lr: float
    steps: List[int]
    max_iters: int


class Detection(BaseModel):
    """Config for detection model training."""

    model_name: str
    solver: Solver
