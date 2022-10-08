"""Test cases for normalize transform."""
import torch

from .normalize import normalize_image


def test_normalize():
    """Image normalize testcase."""
    transform = normalize_image()
    x = torch.zeros((1, 3, 12, 12))
    x = transform({"images": x})
    assert torch.isclose(
        x["images"].view(3, -1).mean(dim=-1),
        torch.tensor([-2.1179, -2.0357, -1.8044]),
        rtol=0.0001,
    ).all()
