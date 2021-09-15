"""Utility functions for bounding boxes."""
import math

import numpy as np
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from vist.struct import Boxes2D


def bbox_intersection(boxes1: Boxes2D, boxes2: Boxes2D) -> torch.Tensor:
    """Given two lists of boxes of size N and M, compute N x M intersection.

    Args:
        boxes1: N 2D boxes in format (x1, y1, x2, y2, Optional[score])
        boxes2: M 2D boxes in format (x1, y1, x2, y2, Optional[score])

    Returns:
        Tensor: intersection (N, M).
    """
    boxes1, boxes2 = boxes1.boxes[:, :4], boxes2.boxes[:, :4]
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )

    width_height.clamp_(min=0)
    intersection = width_height.prod(dim=2)
    return intersection


def bbox_iou(boxes1: Boxes2D, boxes2: Boxes2D) -> torch.Tensor:
    """Compute IoU between all pairs of boxes.

    Args:
        boxes1: N 2D boxes in format (x1, y1, x2, y2, Optional[score])
        boxes2: M 2D boxes in format (x1, y1, x2, y2, Optional[score])

    Returns:
        Tensor: IoU (N, M).
    """
    area1 = boxes1.area()
    area2 = boxes2.area()
    inter = bbox_intersection(boxes1, boxes2)

    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def random_choice(tensor: torch.Tensor, sample_size: int) -> torch.Tensor:
    """Randomly choose elements from a tensor."""
    perm = torch.randperm(len(tensor), device=tensor.device)[:sample_size]
    return tensor[perm]


def non_intersection(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Get the elements of t1 that are not present in t2."""
    compareview = t2.repeat(t1.shape[0], 1).T
    return t1[(compareview != t1).T.prod(1) == 1]


def project_points_to_image(
    points: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """Project Nx3 points to Nx2 pixel coordinates with 3x3 intrinsics."""
    hom_cam_coords = points / points[:, 2:3]
    pts_2d = hom_cam_coords.mm(intrinsics.t())
    return pts_2d[:, :2]  # type: ignore


def yaw2alpha(rot_y: torch.Tensor, x_loc: torch.Tensor, z_loc: torch.Tensor):
    """
    Get alpha by rotation_y - theta.

    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    alpha : Observation angle of object, ranging [-pi..pi]
    """
    torch_pi = rot_y.new_tensor([np.pi])
    alpha = rot_y - torch.atan2(x_loc, z_loc)
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


def quaternion_to_euler(quat_w, quat_x, quat_y, quat_z):
    """Transform quaternion into euler."""
    t0 = +2.0 * (quat_w * quat_x + quat_y * quat_z)
    t1 = +1.0 - 2.0 * (quat_x * quat_x + quat_y * quat_y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (quat_w * quat_y - quat_z * quat_x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (quat_w * quat_z + quat_x * quat_y)
    t4 = +1.0 - 2.0 * (quat_y * quat_y + quat_z * quat_z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def euler_to_quaternion(roll, pitch, yaw):
    """Transform euler into quaternion."""
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qw, qx, qy, qz]


def get_yaw_world(det_yaws, cam_extrinsics):
    """Transfer yaw in cam to yaw in world."""
    r_camera_to_world = R.from_matrix(cam_extrinsics[:3, :3]).as_matrix()
    cam_rot_quat = Quaternion(matrix=r_camera_to_world)

    quat_det_yaws_world = {"roll_pitch": [], "yaw_world": []}

    for det_yaw in det_yaws:
        yaw_quat = Quaternion(axis=[0, 1, 0], radians=det_yaw.cpu().numpy())
        rotation_world = cam_rot_quat * yaw_quat
        if rotation_world.z < 0:
            rotation_world *= -1
        roll_world, pitch_world, yaw_world = quaternion_to_euler(
            rotation_world.w,
            rotation_world.x,
            rotation_world.y,
            rotation_world.z,
        )
        quat_det_yaws_world["roll_pitch"].append([roll_world, pitch_world])
        quat_det_yaws_world["yaw_world"].append(yaw_world)

    quat_det_yaws_world["yaw_world"] = np.array(
        quat_det_yaws_world["yaw_world"]
    )

    return quat_det_yaws_world


def get_yaw_cam(yaws_world, cam_extrinsics, quat_det_yaws_world):
    """Transfer yaw in world to yaw in cam."""
    r_camera_to_world = R.from_matrix(cam_extrinsics[:3, :3]).as_matrix()
    cam_rot_quat = Quaternion(matrix=r_camera_to_world)

    yaws_cam = []
    for yaw_world, (roll_world, pitch_world) in zip(
        yaws_world, quat_det_yaws_world["roll_pitch"]
    ):
        rotation_cam = cam_rot_quat.inverse * Quaternion(
            euler_to_quaternion(
                roll_world, pitch_world, yaw_world.cpu().numpy()
            )
        )
        vtrans = np.dot(rotation_cam.rotation_matrix, np.array([1, 0, 0]))
        yaws_cam.append(-np.arctan2(vtrans[2], vtrans[0]))

    return np.array(yaws_cam)
