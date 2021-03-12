"""Embedding head for quasi-dense instance similarity learning."""
import torch.nn as nn
from detectron2.config import CfgNode
from typing import Dict, Any

class EmbeddingHead(nn.Module):

    def __init__(self):
        """init."""
        # TODO add arguments

    @classmethod
    def from_config(cls, cfg: CfgNode, input_shape) -> Dict[Any]:
        # TODO read config
        pass


    def forward(self, x):

        # TODO implement forward pass
        pass