"""Interface for using Kornia augmentations in Vis4D.

Kornia augmentations use the following functions:
- generate_parameters: Generate the (random) parameters for computing the
current transformation.
- compute_transformation: Compute a geometric transformation that transforms
the input. Optionally in Kornia but required here, since we need the transform
to maintain valid projective geometry in 3D tracking.
- apply_transform: Apply the transformation to the input.

Reference: https://kornia.readthedocs.io/en/latest/augmentation.base.html
"""
from typing import Dict, List, Tuple, Union

import torch
from kornia import augmentation as kornia_augmentation

from vis4d.struct import Boxes2D, Images, InputSample, Intrinsics, TMasks

from ..utils import transform_bbox
from .base import AugParams, BaseAugmentation, BaseAugmentationConfig


class KorniaAugmentationConfig(BaseAugmentationConfig):
    """Config for Kornia augmentation wrapper."""

    kornia_type: str
    kwargs: Dict[
        str,
        Union[
            bool,
            float,
            str,
            Tuple[float, float],
            Tuple[int, int],
            List[Tuple[int, int]],
        ],
    ] = {}


class KorniaAugmentationWrapper(BaseAugmentation):
    """Kornia augmentation wrapper class."""

    def __init__(self, cfg: BaseAugmentationConfig):
        """Initialize wrapper."""
        super().__init__(cfg)
        self.cfg: KorniaAugmentationConfig = KorniaAugmentationConfig(
            **cfg.dict()
        )
        augmentation = getattr(kornia_augmentation, self.cfg.kornia_type)
        self.augmentor = augmentation(p=1.0, **self.cfg.kwargs)

    def generate_parameters(self, sample: InputSample) -> AugParams:
        """Generate current parameters."""
        parameters = super().generate_parameters(sample)
        _params = self.augmentor.generate_parameters(
            sample.images.tensor.shape
        )
        parameters.update(_params)
        parameters["batch_prob"] = parameters["apply"]
        transf = self.augmentor.compute_transformation(
            sample.images.tensor, _params
        )
        transf[~parameters["apply"]] = torch.eye(3, device=transf.device)
        parameters["transform"] = transf
        return parameters

    def apply_intrinsics(
        self,
        intrinsics: Intrinsics,
        parameters: AugParams,
    ) -> Intrinsics:
        """Apply augmentation to input intrinsics."""
        transform = parameters["transform"]
        return Intrinsics(torch.matmul(transform, intrinsics.tensor))

    def apply_image(self, images: Images, parameters: AugParams) -> Images:
        """Apply augmentation to input image."""
        all_ims = []
        for i, im in enumerate(images):
            if parameters["apply"][i]:
                im_t = self.augmentor.apply_transform(
                    im.tensor / 255.0, parameters, parameters["transform"]
                )
                all_ims.append(
                    Images(im_t * 255, [(im_t.shape[3], im_t.shape[2])])
                )
            else:
                all_ims.append(im)

        if len(all_ims) == 1:
            return all_ims[0]
        return Images.cat(all_ims)

    def apply_box2d(
        self,
        boxes: List[Boxes2D],
        parameters: AugParams,
    ) -> List[Boxes2D]:
        """Apply augmentation to input box2d."""
        for i, box in enumerate(boxes):
            if len(box) > 0 and parameters["apply"][i]:
                boxes[i].boxes[:, :4] = transform_bbox(
                    parameters["transform"][i],
                    box.boxes[:, :4],
                )
        return boxes

    def apply_mask(
        self,
        masks: List[TMasks],
        parameters: AugParams,
    ) -> List[TMasks]:
        """Apply augmentation to input mask."""
        for i, mask in enumerate(masks):
            if len(mask) > 0 and parameters["apply"][i]:
                mask.masks = (
                    self.augmentor.apply_transform(
                        mask.masks.float().unsqueeze(1),
                        parameters,
                        parameters["transform"][i],
                    )
                    .squeeze(1)
                    .type(mask.masks.dtype)
                )
        return masks


class KorniaColorJitter(KorniaAugmentationWrapper):
    """Wrapper for Kornia color jitter augmentation class."""

    def __init__(self, cfg: BaseAugmentationConfig):
        """Init."""
        cfg.__dict__.update({"kornia_type": "ColorJitter"})
        super().__init__(cfg)

    def apply_mask(
        self, masks: List[TMasks], parameters: AugParams
    ) -> List[TMasks]:
        """Skip augmentation for mask."""
        return masks
