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
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from kornia import augmentation as kornia_augmentation
from pydantic.main import BaseModel

from vist.common.registry import RegistryHolder
from vist.data.utils import transform_bbox
from vist.struct import DictStrAny, Images, InputSample
from vist.struct.labels import Bitmasks, Boxes2D, Boxes3D

from .utils import identity_matrix, batch_prob_generator

AugParams = Dict[str, torch.Tensor]


class AugmentationConfig(BaseModel):
    """Data augmentation instance config."""

    type: str
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
        self.set_rng_device_and_dtype(
            torch.device("cpu"), torch.get_default_dtype()
        )

    def generate_parameters(self, batch_shape: torch.Size) -> DictStrAny:
        """Generate current parameters."""
        raise NotImplementedError

    def compute_transformation(
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

    def apply_transform(
        self, inputs: torch.Tensor, params: AugParams, transform: torch.Tensor
    ) -> torch.Tensor:
        """Apply the transformation given parameters and transform."""
        raise NotImplementedError

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

    def _apply_func(  # type: ignore
        self,
        sample: InputSample,
        parameters: Dict[str, torch.Tensor],
        transform: torch.Tensor,
    ) -> None:
        """Apply transform to input sample."""
        sample.images = self.apply_image(sample.images, parameters, transform)
        sample.boxes2d = self.apply_box2d(
            sample.boxes2d, parameters, transform
        )
        sample.boxes3d = self.apply_box3d(
            sample.boxes3d, parameters, transform
        )
        sample.bitmasks = self.apply_mask(
            sample.bitmasks, parameters, transform
        )

    def __call__(
        self,
        sample: InputSample,
        parameters: DictStrAny,
    ) -> Tuple[InputSample, torch.Tensor]:  # type: ignore
        """Apply augmentations to input sample."""
        if "batch_prob" not in parameters:
            batch_size = sample.images.tensor.shape[0]
            parameters["batch_prob"] = torch.tensor([True] * batch_size)
            warnings.warn(
                "`batch_prob` is not found in params."
                " Will assume applying on all data."
            )

        to_apply = parameters["batch_prob"]
        # if no augmentation needed
        if torch.sum(to_apply) == 0:
            trans_matrix = identity_matrix(sample)
        # if all data needs to be augmented
        elif torch.sum(to_apply) == len(to_apply):
            trans_matrix = self.compute_transformation(
                sample.images.tensor, parameters
            )
            self._apply_func(sample, parameters, trans_matrix)
        else:
            trans_matrix = identity_matrix(sample)
            trans_matrix[to_apply] = self.compute_transformation(
                sample[to_apply], parameters
            )
            self._apply_func(sample[to_apply], parameters, trans_matrix)

        return sample, trans_matrix

    def forward_parameters(self, batch_shape) -> Dict[str, torch.Tensor]:
        """Generate parameters for forward pass."""
        to_apply = batch_prob_generator(
            batch_shape, self.prob, self.prob_batch, self.same_on_batch
        )
        _params = self.generate_parameters(
            torch.Size((int(to_apply.sum().item()), *batch_shape[1:]))
        )
        if _params is None:
            _params = {}
        _params["batch_prob"] = to_apply
        return _params

    def set_rng_device_and_dtype(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Change the random generation device and dtype.

        Note:
            The generated random numbers are not reproducible across different
            devices and dtypes.
        """
        self.device = device
        self.dtype = dtype

    def __repr__(self) -> str:
        """Print class & params, s.t. user can inspect easily via cmd line."""
        return (
            f"p={self.prob}, p_batch={self.prob_batch},"
            f" same_on_batch={self.same_on_batch}"
        )


def build_augmentation(cfg: AugmentationConfig) -> BaseAugmentation:
    """Build a single augmentation."""
    registry = RegistryHolder.get_registry(BaseAugmentation)
    if cfg.type in registry:
        augmentation = registry[cfg.type]
    # elif hasattr(kornia_augmentation, cfg.type):
    #     augmentation = getattr(kornia_augmentation, cfg.type)
    else:
        raise ValueError(f"Augmentation {cfg.type} not known!")
    module = augmentation(cfg, **cfg.kwargs)
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
