"""Class for processing Scalabel type datasets."""
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel
from pytorch_lightning.utilities.distributed import (
    rank_zero_info,
    rank_zero_warn,
)
from scalabel.label.typing import Extrinsics as ScalabelExtrinsics
from scalabel.label.typing import Frame, FrameGroup, ImageSize
from scalabel.label.typing import Intrinsics as ScalabelIntrinsics
from scalabel.label.typing import Label
from scalabel.label.utils import (
    check_crowd,
    check_ignored,
    get_matrix_from_extrinsics,
    get_matrix_from_intrinsics,
)

from ..common.io import DataBackendConfig, build_data_backend
from ..common.registry import RegistryHolder
from ..struct import (
    Boxes2D,
    Boxes3D,
    Extrinsics,
    Images,
    InputSample,
    InstanceMasks,
    Intrinsics,
    LabelInstances,
    PointCloud,
    SemanticMasks,
)
from .transforms import AugParams, BaseAugmentationConfig, build_augmentations
from .utils import im_decode


class SampleMapperConfig(BaseModel):
    """Config for Mapper."""

    type: str = "BaseSampleMapper"
    data_backend: DataBackendConfig = DataBackendConfig()
    categories: Optional[List[str]] = None
    fields_to_load: List[str] = ["boxes2d"]
    skip_empty_samples: bool = False
    clip_bboxes_to_image: bool = True
    min_bboxes_area: float = 7.0 * 7.0
    transformations: Optional[List[BaseAugmentationConfig]] = None


class BaseSampleMapper(metaclass=RegistryHolder):
    """A callable that converts a Scalabel Frame to a Vis4D InputSample."""

    def __init__(
        self,
        cfg: SampleMapperConfig,
        cats_name2id: Dict[str, Dict[str, int]],
        training: bool,
        image_channel_mode: str = "RGB",
    ) -> None:
        """Init Scalabel Mapper."""
        self.cfg = cfg
        self.training = training
        self.image_channel_mode = image_channel_mode
        if self.cfg.skip_empty_samples and not self.training:
            rank_zero_warn(  # pragma: no cover
                "'skip_empty_samples' activated in test mode. This option is "
                "only available in training."
            )

        self.data_backend = build_data_backend(self.cfg.data_backend)
        rank_zero_info("Using data backend: %s", self.cfg.data_backend.type)
        self.transformations = build_augmentations(self.cfg.transformations)
        rank_zero_info("Transformations used: %s", self.transformations)

        fields_to_load = self.cfg.fields_to_load
        allowed_files = [
            "boxes2d",
            "boxes3d",
            "instance_masks",
            "semantic_masks",
            "intrinsics",
            "extrinsics",
            "pointcloud",
        ]
        self.cats_name2id = {}
        for field in fields_to_load:
            assert (
                field in allowed_files
            ), f"Unrecognized field={field}, allowed fields={allowed_files}"
            assert (
                "all" in cats_name2id or field in cats_name2id
            ), f"Field={field} not specified in category_mapping"
            if "all" in cats_name2id:
                self.cats_name2id[field] = cats_name2id["all"]
            else:
                self.cats_name2id[field] = cats_name2id[field]

    def load_input(
        self,
        sample: Frame,
        use_empty: Optional[bool] = False,
        group_url: Optional[str] = None,
        group_extrinsics: Optional[ScalabelExtrinsics] = None,
    ) -> InputSample:
        """Load image according to data_backend."""
        if not use_empty:
            assert sample.url is not None
            im_bytes = self.data_backend.get(sample.url)
            image = im_decode(im_bytes, mode=self.image_channel_mode)
        else:
            image = np.empty((128, 128, 3), dtype=np.uint8)

        sample.size = ImageSize(width=image.shape[1], height=image.shape[0])
        image = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)),
            dtype=torch.float32,
        ).unsqueeze(0)
        images = Images(image, [(image.shape[3], image.shape[2])])
        input_data = InputSample([copy.deepcopy(sample)], images)

        if (
            sample.intrinsics is not None
            and "intrinsics" in self.cfg.fields_to_load
        ):
            input_data.intrinsics = self.load_intrinsics(sample.intrinsics)

        if (
            sample.extrinsics is not None
            and "extrinsics" in self.cfg.fields_to_load
        ):
            input_data.extrinsics = self.load_extrinsics(sample.extrinsics)

        if (
            group_url is not None
            and group_extrinsics is not None
            and "pointcloud" in self.cfg.fields_to_load
        ):
            input_data.points = self.load_point(
                group_url, group_extrinsics, input_data.extrinsics
            )

        return input_data

    def load_annotation(
        self,
        sample: InputSample,
        labels: Optional[List[Label]],
    ) -> None:
        """Transform annotations."""
        labels_used = []
        if labels is not None:
            instance_id_dict = {}
            for label in labels:
                assert label.attributes is not None
                assert label.category is not None
                if not check_crowd(label) and not check_ignored(label):
                    labels_used.append(label)
                    if label.id not in instance_id_dict:
                        instance_id_dict[label.id] = int(
                            label.attributes["instance_id"]
                        )

            if labels_used:
                if "instance_masks" in self.cfg.fields_to_load:
                    instance_masks = InstanceMasks.from_scalabel(
                        labels_used,
                        self.cats_name2id["instance_masks"],
                        instance_id_dict,
                        sample.metadata[0].size,
                    )
                    sample.targets.instance_masks = [instance_masks]

                if "semantic_masks" in self.cfg.fields_to_load:
                    semantic_masks = SemanticMasks.from_scalabel(
                        labels_used,
                        self.cats_name2id["semantic_masks"],
                        instance_id_dict,
                        sample.metadata[0].size,
                    )
                    sample.targets.semantic_masks = [semantic_masks]

                if "boxes2d" in self.cfg.fields_to_load:
                    boxes2d = Boxes2D.from_scalabel(
                        labels_used,
                        self.cats_name2id["boxes2d"],
                        instance_id_dict,
                    )
                    if len(sample.targets.instance_masks[0]) > 0 and (
                        len(boxes2d) == 0
                        or len(boxes2d)
                        != len(sample.targets.instance_masks[0])
                    ):  # pragma: no cover
                        boxes2d = sample.targets.instance_masks[
                            0
                        ].get_boxes2d()
                    sample.targets.boxes2d = [boxes2d]

                if "boxes3d" in self.cfg.fields_to_load:
                    boxes3d = Boxes3D.from_scalabel(
                        labels_used,
                        self.cats_name2id["boxes3d"],
                        instance_id_dict,
                    )
                    sample.targets.boxes3d = [boxes3d]

    def transform_input(
        self,
        sample: InputSample,
        parameters: Optional[List[AugParams]] = None,
    ) -> List[AugParams]:
        """Apply transforms to input sample."""
        if parameters is None:
            parameters = []
        else:
            assert len(parameters) == len(self.transformations), (
                "Length of augmentation parameters must equal the number of "
                "augmentations!"
            )
        for i, aug in enumerate(self.transformations):
            if len(parameters) < len(self.transformations):
                parameters.append(aug.generate_parameters(sample))
            sample, _ = aug(sample, parameters[i])
        return parameters

    def postprocess_annotation(
        self, im_wh: Tuple[int, int], targets: LabelInstances
    ) -> None:
        """Process annotations after transform."""
        if len(targets.boxes2d[0]) == 0:
            return
        if self.cfg.clip_bboxes_to_image:
            targets.boxes2d[0].clip(im_wh)
        keep = targets.boxes2d[0].area >= self.cfg.min_bboxes_area
        targets.boxes2d = [targets.boxes2d[0][keep]]
        if len(targets.boxes3d[0]) > 0:
            targets.boxes3d = [targets.boxes3d[0][keep]]
        if len(targets.instance_masks[0]) > 0:
            targets.instance_masks = [targets.instance_masks[0][keep]]

    @staticmethod
    def load_intrinsics(intrinsics: ScalabelIntrinsics) -> Intrinsics:
        """Transform intrinsic camera matrix according to augmentations."""
        intrinsic_matrix = torch.from_numpy(
            get_matrix_from_intrinsics(intrinsics)
        ).to(torch.float32)
        return Intrinsics(intrinsic_matrix)

    @staticmethod
    def load_extrinsics(extrinsics: ScalabelExtrinsics) -> Extrinsics:
        """Transform extrinsics from Scalabel to Vis4D."""
        extrinsics_matrix = torch.from_numpy(
            get_matrix_from_extrinsics(extrinsics)
        ).to(torch.float32)
        return Extrinsics(extrinsics_matrix)

    def load_point(
        self,
        group_url: str,
        group_extrinsics: ScalabelExtrinsics,
        input_data_extrinsics: Extrinsics,
        num_point_feature: int = 4,
        radius: float = 1.0,
    ) -> PointCloud:
        """Load pointcloud points and filter the near ones."""
        points = np.fromfile(group_url, dtype=np.float32)  # type: ignore # pylint: disable=line-too-long
        s = points.shape[0]
        if s % 5 != 0:
            points = points[: s - (s % 5)]
        points = points.reshape(-1, 5)[:, :num_point_feature].T

        x_filt = np.abs(points[0, :]) < radius
        y_filt = np.abs(points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        points = points[:, not_close].T
        point_cloud = PointCloud(torch.as_tensor(points))

        points_extrinsics = self.load_extrinsics(group_extrinsics)

        hom_points = torch.cat(
            [
                point_cloud.tensor[:, :, :3],
                torch.ones_like(point_cloud.tensor[:, :, 0:1]),
            ],
            -1,
        )
        points_world = hom_points @ points_extrinsics.transpose().tensor
        point_cloud.tensor[:, :, :3] = (
            points_world @ input_data_extrinsics.inverse().transpose().tensor
        )[:, :, :3]
        return point_cloud

    def __call__(
        self,
        sample: Frame,
        parameters: Optional[List[AugParams]] = None,
        group_url: Optional[str] = None,
        group_extrinsics: Optional[ScalabelExtrinsics] = None,
    ) -> Tuple[Optional[InputSample], Optional[List[AugParams]]]:
        """Prepare a single sample in detect format.

        Args:
            sample (Frame): Metadata of one image, in scalabel format.
            Serialized as dict due to multi-processing.
            parameters (List[AugParams]): Augmentation parameter list.
            group_url (str): Url of group sensor path.
            group_extrinsics (ScalabelExtrinsics): Extrinsics for group sensor.

        Returns:
            InputSample: Data format that the model accepts.
            List[AugParams]: augmentation parameters, s.t. ref views can be
            augmented with the same parameters.
        """
        if (
            self.cfg.skip_empty_samples
            and (sample.labels is None or len(sample.labels) == 0)
            and self.training
        ):
            return None, None  # pragma: no cover

        # load input data
        input_data = self.load_input(
            sample,
            use_empty=isinstance(sample, FrameGroup),
            group_url=group_url,
            group_extrinsics=group_extrinsics,
        )

        if self.training:
            # load annotations to input sample
            self.load_annotation(input_data, sample.labels)

        # apply transforms to input sample
        parameters = self.transform_input(input_data, parameters)

        if not self.training:
            return input_data, parameters

        # postprocess boxes after transforms
        self.postprocess_annotation(
            input_data.images.image_sizes[0], input_data.targets
        )

        if self.cfg.skip_empty_samples and input_data.targets.empty:
            return None, None  # pragma: no cover
        return input_data, parameters


def build_mapper(
    cfg: SampleMapperConfig,
    cats_name2id: Dict[str, Dict[str, int]],
    training: bool,
    image_channel_mode: str = "RGB",
) -> BaseSampleMapper:
    """Build a mapper."""
    registry = RegistryHolder.get_registry(BaseSampleMapper)
    registry["BaseSampleMapper"] = BaseSampleMapper
    if cfg.type in registry:
        module = registry[cfg.type](
            cfg, cats_name2id, training, image_channel_mode
        )
        assert isinstance(module, BaseSampleMapper)
    else:
        raise NotImplementedError(f"Mapper type {cfg.type} not found.")
    return module
