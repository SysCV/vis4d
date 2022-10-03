"""Horizontal flip augmentation."""
from typing import Tuple

import torch

from vis4d.data.datasets.base import DataKeys, DictData
from vis4d.data.transforms.base import Transform
from vis4d.struct_to_revise import DictStrAny


def hflip(tensor: torch.Tensor) -> torch.Tensor:
    """Flip a tensor of shape [N, C, H, W] horizontally."""
    return tensor.flip(-1)


def hflip_boxes(boxes: torch.Tensor, im_width: float) -> torch.Tensor:
    """Flip bounding box tensor."""
    tmp = im_width - boxes[..., 2::4]
    boxes[..., 2::4] = im_width - boxes[..., 0::4]
    boxes[..., 0::4] = tmp
    return boxes


# TODO implement for 3D data
# def apply_box3d(
#     self, boxes: List[Boxes3D], parameters: AugParams
# ) -> List[Boxes3D]:
#     """Apply augmentation to input box3d."""
#     for i, box in enumerate(boxes):
#         if len(box) > 0 and parameters["apply"][i]:
#             box.boxes[:, 0] *= -1.0
#             box.boxes[:, 7] = normalize_angle(np.pi - box.boxes[:, 7])
#     return boxes

# def apply_points(
#     self, points: PointCloud, parameters: AugParams
# ) -> PointCloud:
#     """Apply augmentation to input points."""
#     if parameters["apply"]:
#         points.tensor[:, :, 0] *= -1.0
#     return points

# def apply_intrinsics(
#     self, intrinsics: Intrinsics, parameters: AugParams
# ) -> Intrinsics:
#     """Apply augmentation to input intrinsics."""
#     center = parameters["batch_shape"][3] / 2
#     for i, _intrinsics in enumerate(intrinsics):
#         if parameters["apply"]:
#             _intrinsics.tensor[i][0][2] = center - (
#                 _intrinsics.tensor[i][0][2] - center
#             )
#     return intrinsics


class HorizontalFlip(Transform):
    """Horizontal flip augmentation class."""

    def __init__(
        self, in_keys: Tuple[str, ...] = (DataKeys.images, DataKeys.boxes2d)
    ):
        """Init."""
        super().__init__(in_keys)

    def generate_parameters(self, data: DictData) -> DictStrAny:
        """Generate flip parameters (empty)."""
        im_shape = data[DataKeys.images].shape[-2:]
        return dict(im_shape=im_shape)

    def __call__(self, data: DictData, parameters: DictStrAny) -> DictData:
        """Apply horizontal flip."""
        im_shape = parameters["im_shape"]
        data[DataKeys.images] = hflip(data[DataKeys.images])
        data[DataKeys.boxes2d] = hflip_boxes(
            data[DataKeys.boxes2d], im_shape[1]
        )
        return data
