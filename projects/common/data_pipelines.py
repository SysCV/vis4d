"""Standard data augmentation pipelines."""
from typing import List, Optional, Tuple

from pytorch_lightning.utilities.rank_zero import rank_zero_info

from vis4d.common.io import BaseDataBackend, FileBackend, HDF5Backend
from vis4d.data.module import BaseDataModule
from vis4d.data.transforms import (
    BaseAugmentation,
    KorniaAugmentationWrapper,
    KorniaColorJitter,
    KorniaRandomHorizontalFlip,
    MixUp,
    Mosaic,
    RandomCrop,
    Resize,
)
from vis4d.struct import ArgsType


class CommonDataModule(BaseDataModule):
    """Common data module."""

    def __init__(
        self,
        experiment: str,
        *args: ArgsType,
        use_hdf5: bool = False,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        self.experiment = experiment
        self.use_hdf5 = use_hdf5
        super().__init__(*args, **kwargs)

    def create_datasets(self, stage: Optional[str] = None) -> None:
        """Create Train / Test / Predict Datasets."""
        raise NotImplementedError

    def _setup_backend(self) -> BaseDataBackend:
        """Setup data backend."""
        backend = FileBackend() if not self.use_hdf5 else HDF5Backend()
        rank_zero_info("Using data backend: %s", backend.__class__.__name__)
        return backend


def default(im_hw: Tuple[int, int]) -> List[BaseAugmentation]:
    """Generate default augmentation pipeline."""
    augs: List[BaseAugmentation] = []
    augs += [KorniaRandomHorizontalFlip(prob=0.5)]
    augs += [Resize(shape=im_hw, keep_ratio=True)]
    return augs


def multi_scale(im_hw: Tuple[int, int]) -> List[BaseAugmentation]:
    """Generate multi-scale training augmentation pipeline."""
    augs: List[BaseAugmentation] = []
    augs += [Resize(shape=im_hw, scale_range=(0.8, 1.2), keep_ratio=True)]
    augs += [RandomCrop(shape=im_hw)]
    augs += [KorniaRandomHorizontalFlip(prob=0.5)]
    return augs


def mosaic_mixup(
    im_hw: Tuple[int, int],
    clip_inside_image: bool = True,
    multiscale_sizes: Optional[List[Tuple[int, int]]] = None,
) -> List[BaseAugmentation]:
    """Generate augmentation pipeline used for YOLOX training."""
    augs: List[BaseAugmentation] = []
    augs += [Mosaic(out_shape=im_hw, clip_inside_image=clip_inside_image)]
    augs += [
        KorniaAugmentationWrapper(
            prob=1.0,
            kornia_type="RandomAffine",
            kwargs={
                "degrees": 10.0,
                "translate": [0.1, 0.1],
                "scale": [0.5, 1.5],
                "shear": [2.0, 2.0],
            },
        )
    ]
    augs += [MixUp(out_shape=im_hw, clip_inside_image=clip_inside_image)]
    augs += [KorniaRandomHorizontalFlip(prob=0.5)]
    if multiscale_sizes is None:
        augs += [Resize(shape=im_hw, keep_ratio=True)]
    else:
        augs += [
            Resize(
                shape=multiscale_sizes, multiscale_mode="list", keep_ratio=True
            )
        ]
    return augs


def add_colorjitter(augs: List[BaseAugmentation], p: float = 0.5) -> None:
    """Add color jitter to existing augmentation pipeline."""
    augs += [
        KorniaColorJitter(
            prob=p,
            kwargs={
                "brightness": [0.875, 1.125],
                "contrast": [0.5, 1.5],
                "saturation": [0.5, 1.5],
                "hue": [-0.1, 0.1],
            },
        )
    ]
