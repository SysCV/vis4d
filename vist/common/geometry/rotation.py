"""Rotation utilities."""
import math

import numpy as np
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform.rotation import Rotation as R


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