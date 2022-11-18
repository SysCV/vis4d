"""Horizontal flip augmentation."""
import torch

from vis4d.data.const import COMMON_KEYS, AxisMode

from .base import Transform


@Transform()
def flip_image(direction: str = "horizontal"):
    """Flip a tensor of shape [N, C, H, W] horizontally.

    Args:
        direction (str, optional): Either vertical or horizontal. Defaults to
            "horizontal".
    """

    def _flip(tensor: torch.Tensor) -> torch.Tensor:
        if direction == "horizontal":
            return tensor.flip(-1)
        elif direction == "vertical":
            return tensor.flip(-2)
        raise NotImplementedError(f"Direction {direction} not known!")

    return _flip


@Transform(
    in_keys=(COMMON_KEYS.boxes2d, COMMON_KEYS.images),
    out_keys=(COMMON_KEYS.boxes2d,),
)
def flip_boxes2d(direction: str = "horizontal"):
    """Flip 2D bounding box tensor.

    Args:
        direction (str, optional): Either vertical or horizontal. Defaults to
            "horizontal".
    """

    def _flip(boxes: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if direction == "horizontal":
            im_width = image.size(3)
            tmp = im_width - boxes[..., 2::4]
            boxes[..., 2::4] = im_width - boxes[..., 0::4]
            boxes[..., 0::4] = tmp
            return boxes
        elif direction == "vertical":
            im_height = image.size(2)
            tmp = im_height - boxes[..., 3::4]
            boxes[..., 3::4] = im_height - boxes[..., 1::4]
            boxes[..., 1::4] = tmp
            return boxes
        raise NotImplementedError(f"Direction {direction} not known!")

    return _flip


def get_axis(direction: str, axis_mode: AxisMode) -> int:
    """Get axis number of certain direction given axis mode.

    Args:
        direction (str): One of horizontal, vertical and lateral.
        axis_mode (AxisMode): axis mode.

    Returns:
        int: Number of axis in certain direction.
    """
    assert direction in ["horizontal", "lateral", "vertical"]
    coord_mapping = {
        AxisMode.ROS: {
            "horizontal": 0,
            "lateral": 1,
            "vertical": 2,
        },
        AxisMode.OpenCV: {
            "horizontal": 0,
            "vertical": 1,
            "lateral": 2,
        },
    }
    return coord_mapping[axis_mode][direction]


@Transform(in_keys=(COMMON_KEYS.boxes3d,), out_keys=(COMMON_KEYS.boxes3d,))
def flip_boxes3d(direction: str = "horizontal"):
    """ "Flip 3D bounding box tensor.

    Args:
        direction (str, optional): Either vertical or horizontal. Defaults to
            "horizontal".
    """

    def _flip(boxes: torch.Tensor, axis_mode: AxisMode) -> torch.Tensor:
        if direction == "horizontal":
            boxes[:, get_axis(direction, axis_mode)] *= -1.0
            # boxes[:, 7] = normalize_angle(np.pi - boxes[:, 7])  TODO align with Quaternion
            return boxes
        elif direction == "vertical":
            boxes[:, get_axis(direction, axis_mode)] *= -1.0
            return boxes
        raise NotImplementedError(f"Direction {direction} not known!")

    return _flip


@Transform(in_keys=(COMMON_KEYS.points3d,), out_keys=(COMMON_KEYS.points3d,))
def flip_points3d(direction: str = "horizontal"):
    """Flip pointcloud tensor.

    Args:
        direction (str, optional): Either vertical or horizontal. Defaults to "horizontal".
    """

    def _flip(points3d: torch.Tensor) -> torch.Tensor:
        if direction == "horizontal":
            points3d[:, 0] *= -1.0
            return points3d
        elif direction == "vertical":
            points3d[:, 1] *= -1.0
            return points3d
        raise NotImplementedError(f"Direction {direction} not known!")

    return _flip


@Transform(
    in_keys=(COMMON_KEYS.intrinsics,), out_keys=(COMMON_KEYS.intrinsics,)
)
def flip_intrinsics(direction: str = "horizontal"):
    """Modify intrinsics for image flip.

    Args:
        direction (str, optional): Either vertical or horizontal. Defaults to
            "horizontal".
    """

    def _flip(intrinsics: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if direction == "horizontal":
            center = image.size(3) / 2
            intrinsics[0, 2] = center - intrinsics[0, 2] - center
            return intrinsics
        elif direction == "vertical":
            center = image.size(2) / 2
            intrinsics[1, 2] = center - intrinsics[1, 2] - center
            return intrinsics
        raise NotImplementedError(f"Direction {direction} not known!")

    return _flip
