"""Interface Vis4D augmentations."""
from typing import Dict, List, Optional, Tuple

import torch

from vis4d.common.registry import RegistryHolder
from vis4d.struct import (
    Boxes2D,
    Boxes3D,
    DictStrAny,
    Extrinsics,
    Images,
    InputSample,
    InstanceMasks,
    Intrinsics,
    PointCloud,
    SemanticMasks,
    TMasks,
)

from .utils import sample_batched

AugParams = DictStrAny


class BaseAugmentation(metaclass=RegistryHolder):
    """Base augmentation class."""

    def __init__(
        self,
        prob: float = 1.0,
        same_on_batch: bool = False,
        same_on_ref: bool = True,
    ):
        """Initialize augmentation."""
        super().__init__()
        self.prob = prob
        self.same_on_batch = same_on_batch
        self.same_on_ref = same_on_ref
        self.num_samples = 1

    def generate_parameters(self, sample: InputSample) -> AugParams:
        """Generate current parameters."""
        parameters = {}
        parameters["apply"] = sample_batched(
            len(sample), self.prob, self.same_on_batch
        )
        return parameters

    # pylint: disable=unused-argument,no-self-use
    def apply_image(self, images: Images, parameters: AugParams) -> Images:
        """Apply augmentation to input image."""
        return images

    def apply_box2d(
        self, boxes: List[Boxes2D], parameters: AugParams
    ) -> List[Boxes2D]:
        """Apply augmentation to input box2d."""
        return boxes

    def apply_intrinsics(
        self, intrinsics: Intrinsics, parameters: AugParams
    ) -> Intrinsics:
        """Apply augmentation to input intrinsics."""
        return intrinsics

    def apply_extrinsics(
        self, extrinsics: Extrinsics, parameters: AugParams
    ) -> Extrinsics:
        """Apply augmentation to input extrinsics."""
        return extrinsics

    def apply_points(
        self, points: PointCloud, parameters: AugParams
    ) -> PointCloud:
        """Apply augmentation to input points."""
        return points

    def apply_box3d(
        self, boxes: List[Boxes3D], parameters: AugParams
    ) -> List[Boxes3D]:
        """Apply augmentation to input box3d."""
        return boxes

    def apply_mask(
        self, masks: List[TMasks], parameters: AugParams
    ) -> List[TMasks]:
        """Apply augmentation to input mask."""
        return masks

    def apply_instance_mask(
        self, masks: List[InstanceMasks], parameters: AugParams
    ) -> List[InstanceMasks]:
        """Apply augmentation to input instance mask."""
        return self.apply_mask(masks, parameters)

    def apply_semantic_mask(
        self, masks: List[SemanticMasks], parameters: AugParams
    ) -> List[SemanticMasks]:
        """Apply augmentation to input semantic mask."""
        return self.apply_mask(masks, parameters)

    def apply_other_targets(
        self, other: List[Dict[str, torch.Tensor]], parameters: AugParams
    ) -> List[Dict[str, torch.Tensor]]:
        """Apply augmentation to other, user-defined targets."""
        return other

    def apply_other_inputs(
        self, other: List[Dict[str, torch.Tensor]], parameters: AugParams
    ) -> List[Dict[str, torch.Tensor]]:
        """Apply augmentation to other, user-defined inputs."""
        return other

    def __call__(
        self, sample: InputSample, parameters: Optional[AugParams] = None
    ) -> Tuple[InputSample, AugParams]:
        """Apply augmentations to input sample."""
        if parameters is None or not self.same_on_ref:
            parameters = self.generate_parameters(sample)

        sample.images = self.apply_image(sample.images, parameters)
        sample.intrinsics = self.apply_intrinsics(
            sample.intrinsics, parameters
        )
        sample.extrinsics = self.apply_extrinsics(
            sample.extrinsics, parameters
        )

        sample.points = self.apply_points(sample.points, parameters)

        sample.other = self.apply_other_inputs(sample.other, parameters)
        sample.targets.boxes2d = self.apply_box2d(
            sample.targets.boxes2d, parameters
        )
        sample.targets.boxes3d = self.apply_box3d(
            sample.targets.boxes3d, parameters
        )
        sample.targets.instance_masks = self.apply_instance_mask(
            sample.targets.instance_masks, parameters
        )
        sample.targets.semantic_masks = self.apply_semantic_mask(
            sample.targets.semantic_masks, parameters
        )
        sample.targets.other = self.apply_other_targets(
            sample.targets.other, parameters
        )
        return sample, parameters

    def __repr__(self) -> str:  # pragma: no cover
        """Print class & params, s.t. user can inspect easily via cmd line."""
        attr_str = ""
        for k, v in vars(self).items():
            if k != "type" and not k.startswith("_"):
                attr_str += f"{k}={str(v)}, "
        attr_str = attr_str.rstrip(", ")
        return f"{self.__class__.__name__}({attr_str})"
