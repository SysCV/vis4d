"""Constants used in vis4d.data."""
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

    transform_params: DictStrAny
    batch_transform_params: DictStrAny

    images (Tensor): Image of shape [1, C, H, W].
    original_hw (Tuple[int, int]): Original shape of image in (height, width).
    input_hw (Tuple[int, int]): Shape of image in (height, width) after
        transformations.

    boxes2d (Tensor): 2D bounding boxes of shape [N, 4]
    boxes2d_classes (Tensor): Semantic classes of 2D bounding boxes, shape
        [N,].
    boxes2d_track_ids (Tensor): Tracking IDs of 2D bounding boxes, shape [N,].
    masks (Tensor): Instance segmentation masks of shape [N, H, W].
    segmentation_masks (Tensor):

    intrinsics (Tensor): Intrinsic sensor calibration. Shape [3, 3].
    extrinsics (Tensor): Extrinsic sensor calibration, transformation of sensor
        to world coordinate frame. Shape [4, 4].
    axis_mode (AxisMode): Coordinate convention of the current sensor.
    timestamp (int): Sensor timestamp in Unix format.

    points3d (Tensor): 3D pointcloud data, assumed to be [N, 3] and in sensor
        frame.
    colors3d (Tensor): Associated color values for each point, [N, 3].

    semantics3d:
    instances3d:
    boxes3d (Tensor): [N, 10], each row consists of center (XYZ), dimensions
        (WLH), and orientation quaternion (WXYZ).
    boxes3d_classes (Tensor): Associated semantic classes of 3D bounding boxes,
        [N,].
    """

    # transformation parameters
    transform_params = "transform_params"
    batch_transform_params = "batch_transform_params"

    # image based inputs
    images = "images"
    original_hw = "original_hw"
    input_hw = "input_hw"

    # 2D annotations
    boxes2d = "boxes2d"
    boxes2d_classes = "boxes2d_classes"
    boxes2d_track_ids = "boxes2d_track_ids"
    masks = "masks"
    segmentation_masks = "segmentation_masks"

    # sensor calibration
    intrinsics = "intrinsics"
    extrinsics = "extrinsiscs"
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
