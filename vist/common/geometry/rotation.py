"""Rotation utilities."""
import numpy as np
import torch


def yaw2alpha(rot_y: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Get alpha by rotation_y - theta.

    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    alpha : Observation angle of object, ranging [-pi..pi]
    """
    torch_pi = rot_y.new_tensor([np.pi])
    alpha = rot_y - torch.atan2(center[..., 0], center[..., 2])
    alpha = (alpha + torch_pi) % (2 * torch_pi) - torch_pi
    return alpha


def get_alpha(rot: torch.Tensor) -> torch.Tensor:
    """Get alpha from rotation y and bins."""
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    idx = (rot[:, 1] > rot[:, 5]).float()
    alpha1 = torch.atan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(rot[:, 6] / rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)