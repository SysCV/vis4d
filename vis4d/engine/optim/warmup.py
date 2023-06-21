"""Learning rate warmup schedules."""


class BaseLRWarmup:
    """Base class for LR warmup."""

    def __init__(self, warmup_ratio: float, warmup_steps: int):
        """Creates an instance of the class."""
        if not warmup_steps > 0:
            raise ValueError("warmup_steps must be a positive integer")
        if not 0 < warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be in range (0,1]")
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, cur_steps: int, regular_lr: float) -> float:
        """Compute learning rate according to current warmup schedule."""
        raise NotImplementedError


class ConstantLRWarmup(BaseLRWarmup):
    """Constant LR warmup."""

    def __call__(self, cur_steps: int, regular_lr: float) -> float:
        """Compute learning rate according to constant warmup schedule."""
        warmup_lr = regular_lr * self.warmup_ratio
        return warmup_lr


class LinearLRWarmup(BaseLRWarmup):
    """Linear LR warmup."""

    def __call__(self, cur_steps: int, regular_lr: float) -> float:
        """Compute learning rate according to linear warmup schedule."""
        k = (1 - cur_steps / self.warmup_steps) * (1 - self.warmup_ratio)
        warmup_lr = regular_lr * (1 - k)
        return warmup_lr


class ExponentialLRWarmup(BaseLRWarmup):
    """Exponential LR warmup."""

    def __call__(self, cur_steps: int, regular_lr: float) -> float:
        """Compute learning rate according to exponential warmup schedule."""
        k = self.warmup_ratio ** (1 - cur_steps / self.warmup_steps)
        warmup_lr = regular_lr * k
        return warmup_lr


class QuadraticLRWarmup(BaseLRWarmup):
    """Quadratic LR warmup."""

    def __call__(self, cur_steps: int, regular_lr: float) -> float:
        """Compute learning rate according to quadratic warmup schedule."""
        k = self.warmup_ratio * (cur_steps * 2 + 1) / self.warmup_steps**2
        warmup_lr = regular_lr * k
        return warmup_lr
