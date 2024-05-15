"""Rotation utilities."""

import functools

import torch
import torch.nn.functional as F
from torch import Tensor

from vis4d.data.const import AxisMode


def normalize_angle(input_angles: Tensor) -> Tensor:
    """Normalize content of input_angles to range [-pi, pi].

    Args:
        input_angles: (Tensor) tensor of any shape containing
                       unnormalized angles.

    Returns:
        Tensor with angles normalized to +/- pi
    """
    return torch.sub((input_angles + torch.pi) % (2 * torch.pi), torch.pi)


def acute_angle(theta_1: Tensor, theta_2: Tensor) -> Tensor:
    """Update theta_1 to mkae the agnle between two thetas is acute."""
    # Make sure the angle between two thetas is acute
    if torch.pi / 2.0 < abs(theta_2 - theta_1) < torch.pi * 3 / 2.0:
        theta_1 += torch.pi
        if theta_1 > torch.pi:
            theta_1 -= torch.pi * 2
        if theta_1 < -torch.pi:
            theta_1 += torch.pi * 2

    # Convert the case of > 270 to < 90
    if abs(theta_2 - theta_1) >= torch.pi * 3 / 2.0:
        if theta_2 > 0:
            theta_1 += torch.pi * 2
        else:
            theta_1 -= torch.pi * 2
    return theta_1


def yaw2alpha(rot_y: Tensor, center: Tensor) -> Tensor:
    """Get alpha by vertical rotation - theta.

    Args:
        rot_y: Rotation around Y-axis in camera coordinates [-pi..pi]
        center: 3D object center in camera coordinates

    Returns:
        alpha: Observation angle of object, ranging [-pi..pi]
    """
    alpha = rot_y - torch.atan2(center[..., 0], center[..., 2])
    return normalize_angle(alpha)


def alpha2yaw(alpha: Tensor, center: Tensor) -> Tensor:
    """Get vertical rotation by alpha + theta.

    Args:
        alpha: Observation angle of object, ranging [-pi..pi]
        center: 3D object center in camera coordinates

    Returns:
        rot_y: Vertical rotation in camera coordinates [-pi..pi]
    """
    rot_y = alpha + torch.atan2(center[..., 0], center[..., 2])
    return normalize_angle(rot_y)


def rotation_output_to_alpha(output: Tensor, num_bins: int = 2) -> Tensor:
    """Get alpha from bin-based regression output.

    Uses method described in (with two bins):
    See: 3D Bounding Box Estimation Using Deep Learning and Geometry,
    Mousavian et al., CVPR'17

    Args:
        output: (Tensor) bin based regressed output.
        num_bins: (int) number of bins to use

    Returns:
        Tensor containing the angle from the bin-based regression output
    """
    out_range = torch.tensor(list(range(len(output))), device=output.device)
    bin_idx = output[:, :num_bins].argmax(dim=-1)
    res_idx = num_bins + 2 * bin_idx
    bin_centers = torch.arange(
        -torch.pi, torch.pi, 2 * torch.pi / num_bins, device=output.device
    )
    bin_centers += torch.pi / num_bins
    alpha = (
        torch.atan(output[out_range, res_idx] / output[out_range, res_idx + 1])
        + bin_centers[bin_idx]
    )
    return alpha


def generate_rotation_output(pred: Tensor, num_bins: int = 2) -> Tensor:
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


# Rotation conversion functions adapted from:
# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def _axis_angle_rotation(axis: str, angle: Tensor) -> Tensor:
    """Get rotation matrix for an angle around an axis.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    assert axis in {"X", "Y", "Z"}, f"Invalid axis {axis}."
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        rot_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        rot_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    else:
        rot_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(rot_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(
    euler_angles: Tensor, convention: str = "XYZ"
) -> Tensor:
    """Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
        "X", "Y", and "Z".

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
    matrices = [
        _axis_angle_rotation(c, a)
        for c, a in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return functools.reduce(torch.matmul, matrices)


def _index_from_letter(letter: str) -> int:  # pragma: no cover
    """Return index from letter.

    Args:
        letter: (str) letter in [X,Y,Z]

    Returns:
        int mapping of the corresponding letter [0,1,2]

    Raises:
        ValueError: if the given letter is not valid
    """
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
    data: Tensor,
    horizontal: bool,
    tait_bryan: bool,
) -> Tensor:
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
    even = axis + other_axis in {"XY", "YZ", "ZX"}
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: Tensor, convention: str = "XYZ") -> Tensor:
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
        rads = matrix[..., i0, i2]
        # safety for nan
        rads[torch.where(rads > 1.0)] = rads.new_tensor([1.0]).to(rads.device)
        rads[torch.where(rads < -1.0)] = rads.new_tensor([-1.0]).to(
            rads.device
        )
        central_angle = torch.asin(
            rads * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
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


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(quat: Tensor) -> Tensor:
    """Returns sqrt(max(0, x)) but with a zero subgradient where x is 0."""
    ret = torch.zeros_like(quat)
    positive_mask = quat > 0
    ret[positive_mask] = torch.sqrt(quat[positive_mask])
    return ret


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).

    Raises:
        ValueError: If shape of input matrix is not correct.
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is
    # small, the candidate won't be picked.
    quat_candidates = quat_by_rijk / (
        2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1))
    )

    # if not for numerical problems, quat_candidates[i] should be same
    # (up to a sign), forall i; we pick the best-conditioned one
    # (with the largest denominator)

    return quat_candidates[
        F.one_hot(  # pylint: disable=not-callable
            q_abs.argmax(dim=-1), num_classes=4
        )
        > 0.5,
        :,  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)


def standardize_quaternion(quaternions: Tensor) -> Tensor:
    """Convert a unit quaternion to a standard form.

    Standard form: One in which the real part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(quat1: Tensor, quat2: Tensor) -> Tensor:
    """Multiply two quaternions.

    Usual torch rules for broadcasting apply.

    Args:
        quat1: Quaternions as tensor of shape (..., 4), real part first.
        quat2: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of quat1 and quat2, tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(quat1, -1)
    bw, bx, by, bz = torch.unbind(quat2, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(quat1: Tensor, quat2: Tensor) -> Tensor:
    """Multiply two quaternions representing rotations.

    Returns the quaternion representing their composition, i.e. the version
    with nonnegative real part. Usual torch rules for broadcasting apply.

    Args:
        quat1: Quaternions as tensor of shape (..., 4), real part first.
        quat2: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of quat1 and quat2, tensor of quaternions shape (..., 4).
    """
    return standardize_quaternion(quaternion_raw_multiply(quat1, quat2))


def quaternion_invert(quaternion: Tensor) -> Tensor:
    """Return quaternion that represents inverse rotation.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(quaternion: Tensor, points: Tensor) -> Tensor:
    """Apply the rotation given by a quaternion to a 3D point.

    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        points: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).

    Raises:
        ValueError: If points is not a valid 3D point set.
    """
    if points.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {points.shape}.")
    real_parts = points.new_zeros(points.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, points), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def rotation_matrix_yaw(
    rotation_matrix: Tensor, axis_mode: AxisMode
) -> Tensor:
    """Get yaw of 3D boxes in euler angle under given axis mode.

    Args:
        rotation_matrix (Tensor): [N, 3, 3] Rotation matrix of the object.
        axis_mode (AxisMode): Coordinate system convention.

    Returns:
        orientation (Tensor): [N, 3] Yaw in euler angle.
    """
    orientation = rotation_matrix.new_zeros(rotation_matrix.shape[0], 3)

    if axis_mode == AxisMode.OPENCV:
        orientation[:, 1] = matrix_to_euler_angles(rotation_matrix, "YZX")[
            :, 0
        ]
    else:
        orientation[:, 2] = matrix_to_euler_angles(rotation_matrix, "ZYX")[
            :, 0
        ]
    return orientation


def rotate_orientation(
    orientation: Tensor, extrinsics: Tensor, axis_mode: AxisMode = AxisMode.ROS
) -> Tensor:
    """Rotate the orientation of the object in different coordinate.

    Args:
        orientation (Tensor): [N, 3] Orientation of the object in euler angles.
        extrinsics (Tensor): [4, 4] Extrinsic matrix of the object.
        axis_mode (AxisMode): Coordinate system convention. Default:
            AxisMode.ROS
    """
    rot = extrinsics[:3, :3] @ euler_angles_to_matrix(orientation)
    return rotation_matrix_yaw(rot, axis_mode)


def rotate_velocities(velocities: Tensor, extrinsics: Tensor) -> Tensor:
    """Rotate the velocities of the object in different coordinate."""
    return (extrinsics[:3, :3] @ velocities.unsqueeze(-1)).squeeze(-1)
