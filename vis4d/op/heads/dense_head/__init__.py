"""Dense heads."""
from .mmseg import MMSegDecodeHead
from .rpn import RPNHead

__all__ = [
    "MMSegDecodeHead",
    "RPNHead",
]
