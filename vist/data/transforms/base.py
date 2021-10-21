"""Interface VisT augmentations."""
from typing import List, Optional, Sequence, Tuple

from pydantic.main import BaseModel

from vist.common.registry import RegistryHolder
from vist.struct import (
    Bitmasks,
    Boxes2D,
    Boxes3D,
    DictStrAny,
    Extrinsics,
    Images,
    InputSample,
    Intrinsics,
)

from .utils import sample_batched

AugParams = DictStrAny


class BaseAugmentationConfig(BaseModel, extra="allow"):
    """Data augmentation instance config."""

    prob: float = 1.0
    same_on_batch: bool = False
    type: str


class BaseAugmentation(metaclass=RegistryHolder):
    """Base augmentation class."""

    def __init__(self, cfg: BaseAugmentationConfig):
        """Initialize augmentation."""
        self.cfg = cfg

    def generate_parameters(self, sample: InputSample) -> AugParams:
        """Generate current parameters."""
        parameters = {}
        parameters["apply"] = sample_batched(
            len(sample), self.cfg.prob, self.cfg.same_on_batch
        )
        return parameters

    # pylint: disable=unused-argument,no-self-use
    def apply_image(self, images: Images, parameters: AugParams) -> Images:
        """Apply augmentation to input image."""
        return images

    def apply_box2d(
        self,
        boxes: Sequence[Boxes2D],
        parameters: AugParams,
    ) -> Sequence[Boxes2D]:
        """Apply augmentation to input box2d."""
        return boxes

    def apply_intrinsics(
        self,
        intrinsics: Intrinsics,
        parameters: AugParams,
    ) -> Intrinsics:
        """Apply augmentation to input intrinsics."""
        return intrinsics

    def apply_extrinsics(
        self,
        extrinsics: Extrinsics,
        parameters: AugParams,
    ) -> Extrinsics:
        """Apply augmentation to input extrinsics."""
        return extrinsics

    def apply_box3d(
        self,
        boxes: Sequence[Boxes3D],
        parameters: AugParams,
    ) -> Sequence[Boxes3D]:
        """Apply augmentation to input box3d."""
        return boxes

    def apply_mask(
        self,
        masks: Sequence[Bitmasks],
        parameters: AugParams,
    ) -> Sequence[Bitmasks]:
        """Apply augmentation to input mask."""
        return masks

    def __call__(
        self, sample: InputSample, parameters: Optional[AugParams] = None
    ) -> Tuple[InputSample, AugParams]:
        """Apply augmentations to input sample."""
        if parameters is None:
            parameters = self.generate_parameters(sample)

        sample.images = self.apply_image(sample.images, parameters)
        sample.intrinsics = self.apply_intrinsics(
            sample.intrinsics, parameters
        )
        sample.extrinsics = self.apply_extrinsics(
            sample.extrinsics, parameters
        )

        if len(sample.boxes2d):
            sample.boxes2d = self.apply_box2d(sample.boxes2d, parameters)
        if len(sample.boxes3d):
            sample.boxes3d = self.apply_box3d(sample.boxes3d, parameters)
        if len(sample.bitmasks):
            sample.bitmasks = self.apply_mask(sample.bitmasks, parameters)
        return sample, parameters

    def __repr__(self) -> str:
        """Print class & params, s.t. user can inspect easily via cmd line."""
        attr_str = ""
        for k, v in self.cfg.dict().items():
            attr_str += f"{k}={str(v)}, "
        attr_str.rstrip(", ")
        return f"{self.__class__.__name__}({attr_str})"


def build_augmentation(cfg: BaseAugmentationConfig) -> BaseAugmentation:
    """Build a single augmentation."""
    registry = RegistryHolder.get_registry(BaseAugmentation)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseAugmentation)
        return module
    raise NotImplementedError(f"Augmentation {cfg.type} not known!")


def build_augmentations(
    cfgs: Optional[List[BaseAugmentationConfig]],
) -> List[BaseAugmentation]:
    """Build a list of augmentations."""
    augmentations = []
    if cfgs is not None:
        for aug_cfg in cfgs:
            augmentations.append(build_augmentation(aug_cfg))
    return augmentations
