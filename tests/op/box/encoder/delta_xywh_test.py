"""DeltaXYWHBBoxEncoder test file."""

import torch

from vis4d.op.box.encoder.delta_xywh import (
    DeltaXYWHBBoxDecoder,
    DeltaXYWHBBoxEncoder,
)


def test_delta_xywh_encode() -> None:
    """Test DeltaXYWHBBoxEncoder encode."""
    encoder = DeltaXYWHBBoxEncoder()
    boxes = torch.tensor([[0, 0, 10, 10], [2, 5, 6, 7]])
    targets = torch.tensor([[1, 1, 11, 11], [2.5, 5.3, 6.5, 7.3]])
    enc = encoder(boxes, targets)
    assert enc.shape == (2, 4)
    assert torch.isclose(enc[0, 0], (targets[0, 0] - boxes[0, 0]) / 10)
    assert torch.isclose(enc[0, 1], (targets[0, 1] - boxes[0, 1]) / 10)
    assert torch.isclose(enc[1, 0], (targets[1, 0] - boxes[1, 0]) / 4)
    assert torch.isclose(enc[1, 1], (targets[1, 1] - boxes[1, 1]) / 2)


def test_delta_xywh_decode() -> None:
    """Test DeltaXYWHBBoxDecoder decode."""
    decoder = DeltaXYWHBBoxDecoder()
    boxes = torch.tensor([[0, 0, 10, 10], [2, 5, 6, 7]])
    deltas = torch.tensor([[0.1, 0.1, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    dec = decoder(boxes, deltas)
    assert dec.shape == (2, 4)
    assert torch.isclose(dec[0, 0], boxes[0, 0] + 0.1 * 10)
    assert torch.isclose(dec[0, 1], boxes[0, 1] + 0.1 * 10)
    assert torch.isclose(dec[1, 0], boxes[1, 0] + 1.0 * 4)
    assert torch.isclose(dec[1, 1], boxes[1, 1] + 1.0 * 2)
