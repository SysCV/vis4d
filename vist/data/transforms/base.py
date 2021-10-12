"""Interface for using and extending kornia augmentations in VisT.

Kornia augmentations require the following functions:
- generate_parameters: Generate the (random) parameters for computing the
current transformation.
- compute_transformation: Compute a geometric transformation that transforms
the input. Optionally in Kornia but required here, since we need the transform
to maintain valid projective geometry in 3D tracking.
- apply_transform: Apply the transformation to the input.

Reference: https://kornia.readthedocs.io/en/latest/augmentation.base.html
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from kornia import augmentation as kornia_augmentation
from kornia.augmentation.base import GeometricAugmentationBase2D
from pydantic.main import BaseModel

from vist.common.registry import RegistryHolder
from vist.data.utils import transform_bbox
from vist.struct import DictStrAny, Images, InputSample
from vist.struct.labels import Bitmasks, Boxes2D, Boxes3D

AugParams = Dict[str, torch.Tensor]


class AugmentationConfig(BaseModel):
    """Data augmentation instance config."""

    type: str
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
    ]


class KorniaAugWrapper(GeometricAugmentationBase2D, metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Subclass kornia Augmentation to support registry."""

    def __repr__(self) -> str:
        """Print class & params, s.t. user can inspect easily via cmd line."""
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> DictStrAny:
        """Generate current parameters."""
        raise NotImplementedError

    def compute_transformation(  # pylint: disable=arguments-renamed
        self, inputs: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get the corresponding deterministic transform for a given input.

        Args:
            inputs: Image tensor.
            params: parameters needed for computing transformation matrix.

        Returns:
            Transform: Returns the deterministic transform.
        """
        raise NotImplementedError

    def apply_transform(  # pylint: disable=arguments-renamed
        self, inputs: torch.Tensor, params: AugParams, transform: torch.Tensor
    ) -> torch.Tensor:
        """Apply the transformation given parameters and transform."""
        raise NotImplementedError

    def inverse_transform(  # pylint: disable=arguments-renamed
        self,
        inputs: torch.Tensor,
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
        **kwargs: DictStrAny,
    ) -> torch.Tensor:
        """Apply inverse of transform given input and transform parameters."""
        raise NotImplementedError


class BaseAugmentation(metaclass=RegistryHolder):
    """Base augmentation class."""

    def __init__(self, cfg: AugmentationConfig):
        """Initialize augmentation."""
        self.cfg = cfg

        # get Kornia augmentation
        registry = RegistryHolder.get_registry(KorniaAugWrapper)
        if cfg.kornia_type in registry:
            augmentation = registry[cfg.kornia_type]
        elif hasattr(kornia_augmentation, cfg.kornia_type):
            augmentation = getattr(kornia_augmentation, cfg.kornia_type)
        else:
            raise ValueError(
                f"Kornia Augmentation {cfg.kornia_type} not known!"
            )
        self.augmentor = augmentation(**cfg.kwargs)

    def generate_parameters(self, batch_shape: torch.Size) -> DictStrAny:
        """Generate parameters for augmentation."""
        return self.augmentor.forward_parameters(batch_shape)

    def __call__(
        self, sample: InputSample, parameters: DictStrAny
    ) -> InputSample:
        """Apply augmentations to input sample."""
        sample.images = self.apply_image(sample.images, parameters)
        sample.boxes2d = self.apply_box2d(sample.boxes2d, parameters)
        sample.boxes3d = self.apply_box3d(sample.boxes3d, parameters)
        sample.bitmasks = self.apply_mask(sample.bitmasks, parameters)
        return sample

    def apply_image(  # pylint: disable=unused-argument
        self, image: Images, parameters: DictStrAny
    ) -> Images:
        """Apply augmentation to input image."""
        imaget, tm = self.augmentor(
            image.tensor, parameters, return_transform=True
        )
        parameters["transform_matrix"] = tm
        return Images(imaget, [(imaget.shape[3], imaget.shape[2])])

    def apply_box2d(  # pylint: disable=unused-argument
        self, boxes: Sequence[Boxes2D], parameters: DictStrAny
    ) -> Sequence[Boxes2D]:
        """Apply augmentation to input box2d."""
        assert "transform_matrix" in parameters
        boxes[0].boxes[:, :4] = transform_bbox(
            parameters["transform_matrix"][0],
            boxes[0].boxes[:, :4],
        )
        return boxes

    def apply_box3d(  # pylint: disable=unused-argument
        self, boxes: Sequence[Boxes3D], parameters: DictStrAny
    ) -> Sequence[Boxes3D]:
        """Apply augmentation to input box3d."""
        return boxes

    def apply_mask(  # pylint: disable=unused-argument
        self, masks: Sequence[Bitmasks], parameters: DictStrAny
    ) -> Sequence[Bitmasks]:
        """Apply augmentation to input mask."""
        return masks

    def __repr__(self) -> str:
        """Print class & params, s.t. user can inspect easily via cmd line."""
        return f"({self.augmentor.__repr__()})"


def build_augmentation(
    cfg: AugmentationConfig,
) -> BaseAugmentation:
    """Build a single augmentation."""
    registry = RegistryHolder.get_registry(BaseAugmentation)

    if cfg.type in registry:
        augmentation = registry[cfg.type]
    elif hasattr(kornia_augmentation, cfg.type):
        augmentation = getattr(kornia_augmentation, cfg.type)
    else:
        raise ValueError(f"VisT Augmentation {cfg.type} not known!")
    module = augmentation(cfg)
    return module  # type: ignore


def build_augmentations(
    cfgs: Optional[List[AugmentationConfig]],
) -> List[BaseAugmentation]:
    """Build a list of augmentations and return these as List."""
    augmentations = []
    if cfgs is not None:
        for aug_cfg in cfgs:
            augmentations.append(build_augmentation(aug_cfg))
    return augmentations
