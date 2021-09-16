"""Rotation utilities."""
import numpy as np
import torch


def yaw2alpha(rot_y: torch.Tensor, center: torch.Tensor):
    """
    Get alpha by rotation_y - theta.

    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    alpha : Observation angle of object, ranging [-pi..pi]
    """
    torch_pi = rot_y.new_tensor([np.pi])
    alpha = rot_y - torch.atan2(center[..., 0], center[..., 2])
    alpha = (alpha + torch_pi) % (2 * torch_pi) - torch_pi
    return alpha


def get_alpha(rot):
    """Get alpha from rotation y and bins."""
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    idx = (rot[:, 1] > rot[:, 5]).float()
    alpha1 = torch.atan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(rot[:, 6] / rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def gen_bin_rot(orientation):
    """Transform rotation with bins."""
    # bin 1
    divider1 = torch.sqrt(orientation[:, 2:3] ** 2 + orientation[:, 3:4] ** 2)
    b1sin = orientation[:, 2:3] / divider1
    b1cos = orientation[:, 3:4] / divider1

    # bin 2
    divider2 = torch.sqrt(orientation[:, 6:7] ** 2 + orientation[:, 7:8] ** 2)
    b2sin = orientation[:, 6:7] / divider2
    b2cos = orientation[:, 7:8] / divider2

    rot = torch.cat(
        [
            orientation[:, 0:2],
            b1sin,
            b1cos,
            orientation[:, 4:6],
            b2sin,
            b2cos,
        ],
        1,
    )

    return rot
