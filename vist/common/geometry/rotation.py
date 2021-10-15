"""Rotation utilities."""
import functools

import numpy as np
import torch


def normalize_angle(input_angles: torch.Tensor) -> torch.Tensor:
    """Normalize content of input_angles to range [-pi, pi]."""
    return (input_angles + np.pi) % (2 * np.pi) - np.pi


def yaw2alpha(rot_y: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Get alpha by rotation_y - theta.

    Args:
        rot_y: Rotation around Y-axis in camera coordinates [-pi..pi]
        center: 3D object center in camera coordinates

    Returns:
        alpha: Observation angle of object, ranging [-pi..pi]
    """
    alpha = rot_y - torch.atan2(center[..., 0], center[..., 2])
    return normalize_angle(alpha)


def alpha2yaw(alpha: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Get rotation_y by alpha + theta.

    Args:
        alpha: Observation angle of object, ranging [-pi..pi]
        center: 3D object center in camera coordinates

    Returns:
        rot_y: Rotation around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + torch.atan2(center[..., 0], center[..., 2])
    return normalize_angle(rot_y)


def rotation_output_to_alpha(
    output: torch.Tensor, num_bins: int = 2
) -> torch.Tensor:
    """Get alpha from bin-based regression output.

    Uses method described in (with two bins):
    See: 3D Bounding Box Estimation Using Deep Learning and Geometry,
     Mousavian et al., CVPR'17
    """
    out_range = torch.tensor(list(range(len(output))), device=output.device)
    bin_idx = output[:, :num_bins].argmax(dim=-1)
    res_idx = num_bins + 2 * bin_idx
    bin_centers = torch.arange(
        -np.pi, np.pi, 2 * np.pi / num_bins, device=output.device
    )
    bin_centers += np.pi / num_bins
    alpha = (
        torch.atan(output[out_range, res_idx] / output[out_range, res_idx + 1])
        + bin_centers[bin_idx]
    )
    return alpha


def generate_rotation_output(
    pred: torch.Tensor, num_bins: int = 2
) -> torch.Tensor:
    """Convert output to bin confidence and cos / sin of residual.

    The viewpoint (alpha) prediction (N, num_bins + 2 * num_bins) consists of:
    bin confidences (N, num_bins): softmax logits for bin probability.
        1st entry is probability for orientation being in bin 1,
        2nd entry is probability for orientation being in bin 2,
        and so on.
    bin residual (N, num_bins * 2): angle residual w.r.t. bin N orientation,
        represented as sin and cos values.

    See: 3D Bounding Box Estimation Using Deep Learning and Geometry,
     Mousavian et al., CVPR'17
    """
    pred = pred.view(pred.size(0), -1, 3 * num_bins)
    bin_logits = pred[..., :num_bins]

    bin_residuals = []
    for i in range(num_bins):
        res_idx = num_bins + 2 * i
        norm = pred[..., res_idx : res_idx + 2].norm(dim=-1, keepdim=True)
        bsin = pred[..., res_idx : res_idx + 1] / norm
        bcos = pred[..., res_idx + 1 : res_idx + 2] / norm
        bin_residuals.append(bsin)
        bin_residuals.append(bcos)

    rot = torch.cat([bin_logits, *bin_residuals], -1)
    return rot


# Rotation conversion functions borrowed from:
# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Get rotation matrix for an angle around an axis.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        rot_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        rot_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        rot_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(rot_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(
    euler_angles: torch.Tensor, convention: str = "XYZ"
) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    Raises:
        ValueError: if convention string is not a combination of XYZ
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(
        _axis_angle_rotation, convention, torch.unbind(euler_angles, -1)
    )
    return functools.reduce(torch.matmul, matrices)


def _index_from_letter(letter: str) -> int:
    """Retunr index from letter."""
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter not valid!")


def _angle_from_tan(
    axis: str,
    other_axis: str,
    data: torch.Tensor,
    horizontal: bool,
    tait_bryan: bool,
) -> torch.Tensor:
    """Helper function for matrix_to_euler_angles.

    Extracts the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(
    matrix: torch.Tensor, convention: str = "XYZ"
) -> torch.Tensor:
    """Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).

    Raises:
        ValueError: if convention string is not a combination of XYZ
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)
