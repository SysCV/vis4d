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
from pydantic.main import BaseModel

from vist.common.registry import RegistryHolder
from vist.data.utils import transform_bbox
from vist.struct import (
    Bitmasks,
    Boxes2D,
    Boxes3D,
    DictStrAny,
    Images,
    InputSample,
)

from .utils import batch_prob_generator, identity_matrix

AugParams = Dict[str, torch.Tensor]


class AugmentationConfig(BaseModel):
    """Data augmentation instance config."""

    type: str
    kornia_type: Optional[str] = None
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


class BaseAugmentation(metaclass=RegistryHolder):
    """Base augmentation class."""

    def __init__(self, cfg: AugmentationConfig):
        """Initialize augmentation."""
        self.cfg = cfg
        self.prob = 0.5
        self.prob_batch = 1.0
        self.same_on_batch = False

    def generate_parameters(self, batch_shape: torch.Size) -> DictStrAny:
        """Generate current parameters."""
        parameters = {}
        parameters["batch_prob"] = batch_prob_generator(
            batch_shape, self.prob, self.prob_batch, self.same_on_batch
        )
        return parameters

    def compute_transformation(  # pylint: disable=unused-argument,no-self-use
        self, inputs: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get the corresponding deterministic transform for a given input.

        Args:
            inputs: Image tensor.
            params: parameters needed for computing transformation matrix.

        Returns:
            Transform: Returns the deterministic transform.
        """
        return identity_matrix(inputs)

    def apply_image(  # pylint: disable=unused-argument,no-self-use
        self, image: Images, parameters: DictStrAny, transform: torch.Tensor
    ) -> Images:
        """Apply augmentation to input image."""
        return image

    def apply_box2d(  # pylint: disable=unused-argument,no-self-use
        self,
        boxes: Sequence[Boxes2D],
        parameters: DictStrAny,
        transform: torch.Tensor,
    ) -> Sequence[Boxes2D]:
        """Apply augmentation to input box2d."""
        return boxes

    def apply_box3d(  # pylint: disable=unused-argument,no-self-use
        self,
        boxes: Sequence[Boxes3D],
        parameters: DictStrAny,
        transform: torch.Tensor,
    ) -> Sequence[Boxes3D]:
        """Apply augmentation to input box3d."""
        return boxes

    def apply_mask(  # pylint: disable=unused-argument,no-self-use
        self,
        masks: Sequence[Bitmasks],
        parameters: DictStrAny,
        transform: torch.Tensor,
    ) -> Sequence[Bitmasks]:
        """Apply augmentation to input mask."""
        return masks

    def __call__(
        self, sample: InputSample, parameters: DictStrAny, training: bool
    ) -> Tuple[InputSample, torch.Tensor]:
        """Apply augmentations to input sample."""
        transform = self.compute_transformation(
            sample.images.tensor, parameters
        )
        sample.images = self.apply_image(sample.images, parameters, transform)
        if training:
            sample.boxes2d = self.apply_box2d(
                sample.boxes2d, parameters, transform
            )
            sample.boxes3d = self.apply_box3d(
                sample.boxes3d, parameters, transform
            )
            sample.bitmasks = self.apply_mask(
                sample.bitmasks, parameters, transform
            )
        return sample, transform

    def __repr__(self) -> str:
        """Print class & params, s.t. user can inspect easily via cmd line."""
        return self.__class__.__name__


class KorniaAugmentationWrapper(BaseAugmentation, metaclass=RegistryHolder):
    """Kornia augmentation wrapper class."""

    def __init__(self, cfg: AugmentationConfig):
        """Initialize wrapper."""
        super().__init__(cfg)
        self.prob = 1.0
        augmentation = getattr(
            kornia_augmentation, cfg.kornia_type  # type: ignore
        )
        self.augmentor = augmentation(**cfg.kwargs)

    def generate_parameters(self, batch_shape: torch.Size) -> DictStrAny:
        """Generate current parameters."""
        parameters = super().generate_parameters(batch_shape)
        to_apply = parameters["batch_prob"]
        _params = self.augmentor.generate_parameters(
            torch.Size((int(to_apply.sum().item()), *batch_shape[1:]))
        )
        parameters.update(_params)
        return parameters

    def compute_transformation(
        self, inputs: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get the corresponding deterministic transform for a given input."""
        return self.augmentor.compute_transformation(inputs, params)

    def apply_transform(
        self,
        inputs: torch.Tensor,
        params: AugParams,
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the transformation given parameters and transform."""
        return self.augmentor.apply_transform(inputs, params, transform)

    def apply_image(
        self, image: Images, parameters: DictStrAny, transform: torch.Tensor
    ) -> Images:
        """Apply augmentation to input image."""
        imaget = self.apply_transform(image.tensor, parameters, transform)
        return Images(imaget, [(imaget.shape[3], imaget.shape[2])])

    def apply_box2d(  # pylint: disable=unused-argument
        self,
        boxes: Sequence[Boxes2D],
        parameters: DictStrAny,
        transform: torch.Tensor,
    ) -> Sequence[Boxes2D]:
        """Apply augmentation to input box2d."""
        if len(boxes[0]) != 0:
            boxes[0].boxes[:, :4] = transform_bbox(
                transform[0],
                boxes[0].boxes[:, :4],
            )
        return boxes

    def apply_box3d(  # pylint: disable=unused-argument
        self,
        boxes: Sequence[Boxes3D],
        parameters: DictStrAny,
        transform: torch.Tensor,
    ) -> Sequence[Boxes3D]:
        """Apply augmentation to input box3d."""
        return boxes

    def apply_mask(
        self,
        masks: Sequence[Bitmasks],
        parameters: DictStrAny,
        transform: torch.Tensor,
    ) -> Sequence[Bitmasks]:
        """Apply augmentation to input mask."""
        if len(masks[0]) != 0:
            masks[0].masks = (
                self.apply_transform(
                    masks[0].masks.float().unsqueeze(1), parameters, transform
                )
                .squeeze(1)
                .type(masks[0].masks.dtype)
            )
        return masks

    def __call__(
        self, sample: InputSample, parameters: DictStrAny, training: bool
    ) -> Tuple[InputSample, torch.Tensor]:
        """Apply augmentations to input sample."""
        bprob = parameters["batch_prob"]
        # if no augmentation needed
        if torch.sum(bprob) == 0:
            trans_matrix = [identity_matrix(sample.images.tensor)]
        # if all data needs to be augmented
        elif torch.sum(bprob) == len(bprob):
            _, trans_matrix = super().__call__(sample, parameters, training)
        else:
            trans_matrix = identity_matrix(sample.images.tensor)
            _, trans_matrix[bprob] = super().__call__(
                sample[bprob], parameters, training
            )

        return sample, trans_matrix

    def __repr__(self) -> str:
        """Print class & params, s.t. user can inspect easily via cmd line."""
        return self.augmentor.__repr__()  # type: ignore


def build_kornia_augmentation(
    cfg: AugmentationConfig,
) -> KorniaAugmentationWrapper:
    """Build Kornia augmentation."""
    assert cfg.kornia_type is not None
    kornia_registry = RegistryHolder.get_registry(KorniaAugmentationWrapper)
    if cfg.type in kornia_registry:
        augmentation = kornia_registry[cfg.type]
        module: KorniaAugmentationWrapper = augmentation(cfg)
    elif hasattr(kornia_augmentation, cfg.kornia_type):
        module = KorniaAugmentationWrapper(cfg)
    else:
        raise ValueError(f"Kornia Augmentation {cfg.type} not known!")
    return module


def build_augmentation(cfg: AugmentationConfig) -> BaseAugmentation:
    """Build a single augmentation."""
    if cfg.kornia_type is not None:
        # use Kornia augmentation
        module = build_kornia_augmentation(cfg)
    else:
        # use VisT augmentation
        registry = RegistryHolder.get_registry(BaseAugmentation)
        if cfg.type in registry:
            augmentation = registry[cfg.type]
            module = augmentation(cfg, **cfg.kwargs)
        elif hasattr(kornia_augmentation, cfg.type):
            # default to using Kornia augmentation
            cfg.kornia_type = cfg.type
            module = build_kornia_augmentation(cfg)
        else:
            raise ValueError(f"VisT Augmentation {cfg.type} not known!")
    return module


def build_augmentations(
    cfgs: Optional[List[AugmentationConfig]],
) -> List[BaseAugmentation]:
    """Build a list of augmentations and return these as List."""
    augmentations = []
    if cfgs is not None:
        for aug_cfg in cfgs:
            augmentations.append(build_augmentation(aug_cfg))
    return augmentations
