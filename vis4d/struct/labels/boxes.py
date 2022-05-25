"""Vis4D Boxes data structures."""
from typing import List, Optional, Tuple, Type, TypedDict

import torch
from vis4d.common.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)


# TODO adjust util functions


def merge(cls: Type["TBoxes"], instances: List["TBoxes"]) -> "TBoxes":
    """Merges a list of Boxes into a single Boxes."""
    assert isinstance(instances, (list, tuple))
    assert len(instances) > 0
    assert all((isinstance(inst, Boxes) for inst in instances))
    assert all(instances[0].device == inst.device for inst in instances)

    boxes, class_ids, track_ids = [], [], []
    has_class_ids = all((b.class_ids is not None for b in instances))
    has_track_ids = all((b.track_ids is not None for b in instances))
    min_param = min((b.boxes.shape[-1] for b in instances))
    for b in instances:
        boxes.append(b.boxes[:, :min_param])
        if has_class_ids:
            class_ids.append(b.class_ids)
        if has_track_ids:
            track_ids.append(b.track_ids)

    cat_boxes = cls(
        torch.cat(boxes),
        torch.cat(class_ids) if has_class_ids else None,
        torch.cat(track_ids) if has_track_ids else None,
    )
    return cat_boxes


class Boxes2D(TypedDict):
    """Container class for 2D boxes.

    boxes: torch.FloatTensor: (N, [4, 5]) where each entry is defined by
    [x1, y1, x2, y2, Optional[score]]
    class_ids: torch.LongTensor: (N,) where each entry is the class id of
    the respective box.
    track_ids: torch.LongTensor (N,) where each entry is the track id of
    the respective box.
    """
    class_id: torch.Tensor
    score: Optional[torch.Tensor]
    track_id: Optional[torch.Tensor]



def scale(self, scale_factor_xy: Tuple[float, float]) -> None:
    """Scale bounding boxes according to factor."""
    self.boxes[:, [0, 2]] *= scale_factor_xy[0]
    self.boxes[:, [1, 3]] *= scale_factor_xy[1]


def clip(self, image_wh: Tuple[float, float]) -> None:
    """Clip bounding boxes according to image_wh."""
    self.boxes[:, [0, 2]] = self.boxes[:, [0, 2]].clamp(0, image_wh[0] - 1)
    self.boxes[:, [1, 3]] = self.boxes[:, [1, 3]].clamp(0, image_wh[1] - 1)


def score(self) -> Optional[torch.Tensor]:
    """Return scores of 2D bounding boxes as tensor."""
    if not self.boxes.shape[-1] == 5:
        return None
    return self.boxes[:, -1]


def center(self) -> torch.Tensor:
    """Return center of 2D bounding boxes as tensor."""
    ctr_x = (self.boxes[:, 0] + self.boxes[:, 2]) / 2
    ctr_y = (self.boxes[:, 1] + self.boxes[:, 3]) / 2
    return torch.stack([ctr_x, ctr_y], -1)


def area(self) -> torch.Tensor:
    """Compute area of each bounding box."""
    area = (self.boxes[:, 2] - self.boxes[:, 0]).clamp(0) * (
        self.boxes[:, 3] - self.boxes[:, 1]
    ).clamp(0)
    return area


class Boxes3D(TypedDict):
    """Container for 3D boxes.

    location: torch.FloatTensor (N, 3) xyz
    dimension: torch.FloatTenosr (N, 3) hwl
    rotation: torch.FloatTensor (N, 4) qxyz
    class_ids: torch.LongTensor: (N,) where each entry is the class id of
    the respective box.
    track_ids: torch.LongTensor (N,) where each entry is the track id of
    the respective box.

    x,y,z are in OpenCV camera coordinate system. h, w, l are the 3D box
    dimensions and correspond to their respective axis (length first (x),
    height second (y), width last (z). The rotations are represented by
    quaternions in TODO convention
    """

    location: torch.Tensor
    dimension: torch.Tensor
    rotation: torch.Tensor
    class_id: torch.Tensor
    score: Optional[torch.Tensor]
    track_id: Optional[torch.Tensor]


def create_boxes3d(
    location: torch.Tensor,
    dimension: torch.Tensor,
    rotation: torch.Tensor,
    class_id: torch.Tensor,
    score: Optional[torch.Tensor] = None,
    track_id: Optional[torch.Tensor] = None,
) -> Boxes3D:
    """Check and create boxes3d typed dictionary."""
    assert len(location.shape) == 2, location.shape[1] == 3
    assert len(dimension.shape) == 2, dimension.shape[1] == 3
    assert len(rotation.shape) == 2, rotation.shape[1] == 3
    assert location.shape[0] == dimension.shape[0] == rotation.shape[0]
    assert len(class_id.shape) == 1, class_id.shape[0] == location.shape[0]
    if score is not None:
        assert len(score.shape) == 1, score.shape[0] == location.shape[0]
    if track_id is not None:
        assert len(track_id.shape) == 1, track_id.shape[0] == location.shape[0]

    box_dict: Boxes3D = {
        "location": location,
        "dimension": dimension,
        "rotation": rotation,
        "class_id": class_id,
        "score": score,
        "track_id": track_id,
    }
    return box_dict


def transform_boxes3d(
    self, extrinsics: Extrinsics, in_image_frame: bool = False
) -> None:  # TODO adjust
    """Transform Boxes3D with given Extrinsics.

    Note: Mutates current Boxes3D.
    """
    if len(extrinsics) > 1:
        raise ValueError(
            f"Expected single Extrinsics but got len {len(extrinsics)}!"
        )

    center = torch.cat(
        [self.center, torch.ones_like(self.boxes[:, 0:1])], -1
    ).unsqueeze(-1)
    self.boxes[:, :3] = (extrinsics.tensor @ center)[:, :3, 0]
    rot = extrinsics.rotation @ euler_angles_to_matrix(self.orientation)

    if in_image_frame:
        # we use XZY convention here, since Z usually points up, but we
        # assume OpenCV cam coordinates (Y points down).
        self.boxes[:, 6:9] = matrix_to_euler_angles(rot, "XZY")[:, [0, 2, 1]]
    else:
        self.boxes[:, 6:9] = matrix_to_euler_angles(rot)
