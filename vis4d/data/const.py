"""Defines data related constants.

While the datasets can hold arbitrary data types and formats, this file
provides some constants that are used to define a common data format which is
helpful to use for better data transformation.
"""
from dataclasses import dataclass
from enum import Enum


class AxisMode(Enum):
    """Enum for choosing among different coordinate frame conventions.

    ROS: The coordinate frame aligns with the right hand rule:
        x axis points forward
        y axis points left
        z axis points up
    See also: https://www.ros.org/reps/rep-0103.html#axis-orientation

    OpenCV: The coordinate frame aligns with a camera coordinate system:
        x axis points right
        y axis points down
        z axis points forward
    See also: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    """

    ROS = 0
    OPENCV = 1


@dataclass
class CommonKeys:
    """Common supported keys for DictData.

    While DictData can hold arbitrary keys of data, we define a common set of
    keys where we expect a pre-defined format to enable the usage of common
    data pre-processing operations among different datasets.

    images (NDArrayF32): Image of shape [1, H, W, C].
    original_hw (Tuple[int, int]): Original shape of image in (height, width).
    input_hw (Tuple[int, int]): Shape of image in (height, width) after
        transformations.
    frame_ids (int): If the dataset contains videos, this field indicates the
        temporal frame index of the current image / sample.

    boxes2d (NDArrayF32): 2D bounding boxes of shape [N, 4] in xyxy format.
    boxes2d_classes (NDArrayI32): Semantic classes of 2D bounding boxes, shape
        [N,].
    boxes2d_track_ids (NDArrayI32): Tracking IDs of 2D bounding boxes,
        shape [N,].
    instance_masks (NDArrayU8): Instance segmentation masks of shape [N, H, W].
    segmentation_masks (NDArrayU8): Semantic segmentation masks [H, W].
    depth_maps (NDArrayF32): Depth maps of shape [H, W].
    optical_flows (NDArrayF32): Optical flow maps of shape [H, W, 2].

    categories (NDArrayI32): Image categories of shape [N,].
    smooth_categories (NDArrayF32): Smoothed image categories of shape [N, C].

    intrinsics (NDArrayF32): Intrinsic sensor calibration. Shape [3, 3].
    extrinsics (NDArrayF32): Extrinsic sensor calibration, transformation of
        sensor to world coordinate frame. Shape [4, 4].
    axis_mode (AxisMode): Coordinate convention of the current sensor.
    timestamp (int): Sensor timestamp in Unix format.

    points3d (NDArrayF32): 3D pointcloud data, assumed to be [N, 3] and in
        sensor frame.
    colors3d (NDArrayF32): Associated color values for each point, [N, 3].

    semantics3d:  TODO complete
    instances3d:  TODO complete
    boxes3d (NDArrayF32): [N, 10], each row consists of center (XYZ),
        dimensions (WLH), and orientation quaternion (WXYZ).
    boxes3d_classes (NDArrayI32): Associated semantic classes of 3D bounding
        boxes, [N,].
    """

    # image based inputs
    images = "images"
    original_hw = "original_hw"
    input_hw = "input_hw"
    frame_ids = "frame_ids"

    # 2D annotations
    boxes2d = "boxes2d"
    boxes2d_classes = "boxes2d_classes"
    boxes2d_track_ids = "boxes2d_track_ids"
    instance_masks = "instance_masks"
    segmentation_masks = "segmentation_masks"
    depth_maps = "depth_maps"
    optical_flows = "optical_flows"

    # Image Classification
    categories = "categories"
    smooth_categories = "smooth_categories"

    # sensor calibration
    intrinsics = "intrinsics"
    extrinsics = "extrinsics"
    axis_mode = "axis_mode"
    timestamp = "timestamp"

    # 3D data
    points3d = "points3d"
    colors3d = "colors3d"

    # 3D annotation
    semantics3d = "semantics3d"
    instances3d = "instances3d"
    boxes3d = "boxes3d"
    boxes3d_classes = "boxes3d_classes"
    boxes3d_track_ids = "boxes3d_track_ids"
    occupancy3d = "occupancy3d"
