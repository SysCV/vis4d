"""Rotation utilities."""
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
    res_idx = num_bins * (bin_idx + 1)
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
        2nd entry is probability for orientation being in bin 2.
    bin residual (N, num_bins * 2): angle residual w.r.t. bin N orientation,
        represented as sin and cos values.

    See: 3D Bounding Box Estimation Using Deep Learning and Geometry,
     Mousavian et al., CVPR'17
    """
    pred = pred.view(pred.size(0), -1, num_bins + 2 * num_bins)
    bin_logits = pred[..., :num_bins]

    bin_residuals = []
    for i in range(num_bins):
        res_idx = num_bins * (i + 1)
        norm = pred[..., res_idx : res_idx + 2].norm(dim=-1, keepdim=True)
        bsin = pred[..., res_idx : res_idx + 1] / norm
        bcos = pred[..., res_idx + 1 : res_idx + 2] / norm
        bin_residuals.append(bsin)
        bin_residuals.append(bcos)

    rot = torch.cat([bin_logits, *bin_residuals], -1)
    return rot
