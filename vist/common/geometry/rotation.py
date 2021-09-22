"""Rotation utilities."""
import numpy as np
import torch


def yaw2alpha(rot_y: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Get alpha by rotation_y - theta.

    Args:
        rot_y: Rotation around Y-axis in camera coordinates [-pi..pi]
        center: 3D object center in camera coordinates

    Returns:
        alpha: Observation angle of object, ranging [-pi..pi]
    """
    torch_pi = rot_y.new_tensor([np.pi])
    alpha = rot_y - torch.atan2(center[..., 0], center[..., 2])
    alpha = (alpha + torch_pi) % (2 * torch_pi) - torch_pi
    return alpha


def alpha2yaw(alpha: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Get rotation_y by alpha + theta.

    Args:
        alpha: Observation angle of object, ranging [-pi..pi]
        center: 3D object center in camera coordinates

    Returns:
        rot_y: Rotation around Y-axis in camera coordinates [-pi..pi]
    """
    torch_pi = alpha.new_tensor([np.pi])
    rot_y = alpha + torch.atan2(center[..., 0], center[..., 2])
    rot_y = (rot_y + torch_pi) % (2 * torch_pi) - torch_pi
    return rot_y


def rotation_output_to_alpha(output: torch.Tensor) -> torch.Tensor:
    """Get alpha from bin-based regression output.

    Uses method described in (with two bins):
    See: 3D Bounding Box Estimation Using Deep Learning and Geometry,
     Mousavian et al., CVPR'17
    """
    idx = (output[:, 0] > output[:, 1]).float()
    alpha1 = torch.atan(output[:, 2] / output[:, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(output[:, 4] / output[:, 5]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def generate_rotation_output(pred: torch.Tensor) -> torch.Tensor:
    """Convert output to bin confidence and cos / sin of residual.

    The viewpoint (alpha) prediction (N, 6) consists of:
    bin confidences (N, 2): softmax logits for bin probability.
        1st entry is probability for orientation being in bin 1,
        2nd entry is probability for orientation being in bin 2.
    bin1 residual (N, 2): angle residual w.r.t. bin1 orientation,
        represented as sin and cos values.
    bin2 residual (N, 2): like above for bin2.

    See: 3D Bounding Box Estimation Using Deep Learning and Geometry,
     Mousavian et al., CVPR'17
    """
    pred = pred.view(pred.size(0), -1, 6)
    bin_logits = pred[..., 0:2]

    # bin 1 residuals
    norm1 = pred[..., 2:4].norm(dim=-1, keepdim=True)
    b1sin = pred[..., 2:3] / norm1
    b1cos = pred[..., 3:4] / norm1

    # bin 2 residuals
    norm2 = pred[..., 4:6].norm(dim=-1, keepdim=True)
    b2sin = pred[..., 4:5] / norm2
    b2cos = pred[..., 5:6] / norm2

    rot = torch.cat([bin_logits, b1sin, b1cos, b2sin, b2cos], -1)
    return rot
