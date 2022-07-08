"""Dataset loader for scalabel format."""
from typing import Dict, List, Optional, Tuple, Union

import torch
from scalabel.eval.detect import evaluate_det
from scalabel.eval.ins_seg import evaluate_ins_seg
from scalabel.eval.mot import acc_single_video_mot, evaluate_track
from scalabel.eval.mots import acc_single_video_mots, evaluate_seg_track
from scalabel.eval.pan_seg import evaluate_pan_seg
from scalabel.eval.result import Result
from scalabel.eval.sem_seg import evaluate_sem_seg
from scalabel.label.io import group_and_sort, load, load_label_config, save
from scalabel.label.typing import Config, Dataset
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

from vis4d.struct import (
    CategoryMap,
    Extrinsics,
    InputData,
    Intrinsics,
    MetricLogs,
    ModelOutput,
)

from ..utils import im_decode
from .base import BaseDataset, DataDict


def _detect(
    pred: List[Frame],
    gt: List[Frame],
    cfg: Config,
    ignore_unknown_cats: bool,  # pylint: disable=unused-argument
) -> Result:
    """Wrapper for evaluate_det function."""
    return evaluate_det(gt, pred, cfg, nproc=1)


def _ins_seg(
    pred: List[Frame],
    gt: List[Frame],
    cfg: Config,
    ignore_unknown_cats: bool,  # pylint: disable=unused-argument
) -> Result:
    """Wrapper for evaluate_ins_seg function."""
    return evaluate_ins_seg(gt, pred, cfg, nproc=1)


def _track(
    pred: List[Frame], gt: List[Frame], cfg: Config, ignore_unknown_cats: bool
) -> Result:
    """Wrapper for evaluate_track function."""
    return evaluate_track(
        acc_single_video_mot,
        group_and_sort(gt),
        group_and_sort(pred),
        cfg,
        nproc=1,
        ignore_unknown_cats=ignore_unknown_cats,
    )


def _seg_track(
    pred: List[Frame], gt: List[Frame], cfg: Config, ignore_unknown_cats: bool
) -> Result:
    """Wrapper for evaluate_seg_track function."""
    return evaluate_seg_track(
        acc_single_video_mots,
        group_and_sort(gt),
        group_and_sort(pred),
        cfg,
        nproc=1,
        ignore_unknown_cats=ignore_unknown_cats,
    )


def _sem_seg(
    pred: List[Frame],
    gt: List[Frame],
    cfg: Config,
    ignore_unknown_cats: bool,  # pylint: disable=unused-argument
) -> Result:
    """Wrapper for evaluate_sem_seg function."""
    return evaluate_sem_seg(gt, pred, cfg, nproc=1)


def _pan_seg(
    pred: List[Frame],
    gt: List[Frame],
    cfg: Config,
    ignore_unknown_cats: bool,
) -> Result:
    """Wrapper for evaluate_pan_seg function."""
    return evaluate_pan_seg(
        gt, pred, cfg, nproc=1, ignore_unknown_cats=ignore_unknown_cats
    )


_eval_mapping = dict(
    detect=_detect,
    track=_track,
    ins_seg=_ins_seg,
    seg_track=_seg_track,
    sem_seg=_sem_seg,
    pan_seg=_pan_seg,
)


class Scalabel(BaseDataset):
    """Preprocess Scalabel format data into Vis4D input format."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewer: Optional[LabelViewer] = None

    def load_dataset(self) -> List[DataDict]:
        """Load Scalabel frames from json."""
        assert self.annotations is not None
        dataset = load(self.annotations, nprocs=self.num_processes)
        add_data_path(data_root, dataset.frames)
        if dataset.groups is not None:
            add_data_path(data_root, dataset.groups)

        metadata_cfg = dataset.config
        if self.config_path is not None:
            metadata_cfg = load_label_config(self.config_path)
        assert metadata_cfg is not None
        dataset.config = metadata_cfg

        cmpt_gbl_ids = compute_global_instance_ids

        t = Timer()
        frequencies = prepare_labels(dataset.frames, class_list, cmpt_gbl_ids)
        rank_zero_info(
            f"Preprocessing {len(dataset.frames)} frames takes {t.time():.2f}"
            " seconds."
        )
        print_class_histogram(frequencies)

        if self.dataset.groups is not None:
            t.reset()
            prepare_labels(self.dataset.groups, class_list, cmpt_gbl_ids)
            rank_zero_info(
                f"Preprocessing {len(self.dataset.groups)} groups takes "
                f"{t.time():.2f} seconds."
            )
            dataset.groups = DatasetFromList(self.dataset.groups)

    def evaluate(
        self, metric: str, predictions: List[Frame], gts: List[Frame]
    ) -> Tuple[MetricLogs, str]:
        """Convert predictions from Scalabel format and evaluate.

        Returns a dictionary of scores to log and a pretty printed string.
        """
        result = _eval_mapping[metric](
            predictions, gts, self.metadata_cfg, self.ignore_unknown_cats
        )
        log_dict = {f"{metric}/{k}": v for k, v in result.summary().items()}
        return log_dict, str(result)

    def save_predictions(
        self, output_dir: str, metric: str, predictions: ModelOutput
    ) -> None:
        """Save model predictions in Scalabel format."""
        # TODO idx_to_class into dataset
        output = [
            p.to(torch.device("cpu")).to_scalabel(idx_to_class)
            for p in predictions
        ]
        file_path = os.path.join(output_dir, f"{metric}_predictions.json")
        save(file_path, output)

    def visualize_predictions(self):
        """Visualize predictions."""
        if self.viewer is None or reset_viewer:
            size = metadata.size
            assert size is not None
            w, h = size.width, size.height
            self.viewer = LabelViewer(UIConfig(width=w, height=h))
        video_name = (
            prediction.videoName if prediction.videoName is not None else ""
        )
        save_path = os.path.join(
            save_dir,
            video_name,
            prediction.name,
        )
        self.viewer.draw(
            np.array(preprocess_image(images.tensor[0])),
            prediction,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.viewer.save(save_path)

    def setup_categories(self, category_map: CategoryMap) -> None:
        """Setup categories."""
        if category_map is None:
            class_list = list(
                set(
                    c.name
                    for c in get_leaf_categories(
                        dataset.metadata_cfg.categories
                    )
                )
            )
            category_map = {c: i for i, c in enumerate(class_list)}
        else:
            class_list = list(
                set(
                    cls
                    for field in category_map
                    for cls in list(category_map[field].keys())
                )
            )
            discard_labels_outside_set(self.dataset, class_list)

        for target in self.targets_to_load:
            if isinstance(list(category_map.values())[0], int):
                self.cats_name2id[target] = category_map  # type: ignore
            else:
                assert (
                    target in category_map
                ), f"Target={target} not specified in category_mapping"
                target_map = category_map[target]
                assert isinstance(target_map, dict)
                self.cats_name2id[target] = target_map

    def load_inputs(
        self,
        sample: Frame,
        use_empty: Optional[bool] = False,
        group_url: Optional[str] = None,
        group_extrinsics: Optional[ScalabelExtrinsics] = None,
    ) -> InputData:
        """Load image according to data_backend."""
        input_data = InputSample([copy.deepcopy(sample)])
        if sample.url is not None and "images" in self.inputs_to_load:
            if not use_empty:
                im_bytes = self.data_backend.get(sample.url)
                image = im_decode(
                    im_bytes,
                    mode=self.image_channel_mode,
                    backend=self.image_backend,
                )
                input_data.metadata[0].size = ImageSize(
                    width=image.shape[1], height=image.shape[0]
                )
            else:
                image = np.empty((128, 128, 3), dtype=np.uint8)

            image = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)),
                dtype=torch.float32,
            ).unsqueeze(0)
            images = Images(image, [(image.shape[3], image.shape[2])])
            input_data.images = images
        if (
            sample.intrinsics is not None
            and "intrinsics" in self.inputs_to_load
        ):
            input_data.intrinsics = self.load_intrinsics(sample.intrinsics)

        if (
            sample.extrinsics is not None
            and "extrinsics" in self.inputs_to_load
        ):
            input_data.extrinsics = self.load_extrinsics(sample.extrinsics)

        if (
            group_url is not None
            and group_extrinsics is not None
            and "pointcloud" in self.inputs_to_load
        ):
            input_data.points = self.load_points(
                group_url, group_extrinsics, input_data.extrinsics
            )

        return input_data

    def load_annotations(self, sample, labels: Optional[List[Label]]) -> None:
        """Transform annotations."""
        if labels is None:
            return
        labels_used, instid_map = [], {}
        for label in labels:
            assert label.attributes is not None and label.category is not None
            if not check_crowd(label) and not check_ignored(label):
                labels_used.append(label)
                if label.id not in instid_map:
                    instid_map[label.id] = int(label.attributes["instance_id"])
        if not labels_used:
            return  # pragma: no cover

        frame = sample.metadata[0]
        if "instance_masks" in self.targets_to_load:
            ins_map = self.cats_name2id["instance_masks"]
            instance_masks = InstanceMasks.from_scalabel(
                labels_used, ins_map, instid_map, frame.size
            )
            sample.targets.instance_masks = [instance_masks]

        if "semantic_masks" in self.targets_to_load:
            sem_map = self.cats_name2id["semantic_masks"]
            semantic_masks = SemanticMasks.from_scalabel(
                labels_used, sem_map, instid_map, frame.size, self.bg_as_class
            )
            sample.targets.semantic_masks = [semantic_masks]

        if "boxes2d" in self.targets_to_load:
            boxes2d = scalabel_to_boxes2d(
                labels_used, self.cats_name2id["boxes2d"], instid_map
            )
            ins_masks = sample.targets.instance_masks[0]
            if len(ins_masks) > 0 and (
                len(boxes2d) == 0 or len(boxes2d) != len(ins_masks)
            ):  # pragma: no cover
                boxes2d = ins_masks.get_boxes2d()
            sample.targets.boxes2d = [boxes2d]

        if "boxes3d" in self.targets_to_load:
            boxes3d = Boxes3D.from_scalabel(
                labels_used, self.cats_name2id["boxes3d"], instid_map
            )
            sample.targets.boxes3d = [boxes3d]

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

    def load_points(
        self,
        group_url: str,
        group_extrinsics: ScalabelExtrinsics,
        input_data_extrinsics: Extrinsics,
        num_point_feature: int = 4,
        radius: float = 1.0,
    ) -> torch.Tensor:
        """Load pointcloud points and filter the near ones."""
        points_bytes = self.data_backend.get(group_url)
        points = np.frombuffer(points_bytes, dtype=np.float32)  # type: ignore # pylint: disable=line-too-long
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

    def load_sample(
        self,
        sample: DataDict,
        training: bool,
    ) -> Optional[InputData]:
        """Prepare a single sample in Vis4D format.

        Args:
            sample: Metadata of one image, in scalabel format.
            Serialized as dict due to multi-processing.
            training: If mode is training, annotations will be loaded.

        Returns:
            InputSample: Data format that the model accepts.
            List[AugParams]: augmentation parameters, s.t. ref views can be
            augmented with the same parameters.

        Raises:
            AttributeError: If category mappings have not been initialized.
        """
        if (
            self.skip_empty_samples
            and (sample.labels is None or len(sample.labels) == 0)
            and training
        ):
            return None  # pragma: no cover

        # load input data
        input_data = self.load_inputs(
            sample,
            use_empty=isinstance(sample, FrameGroup),
        )

        if len(self.targets_to_load) > 0 and training:
            if len(self.cats_name2id) == 0:
                raise AttributeError(
                    "Category mapping is empty but targets_to_load is not. "
                    "Please specify a category mapping."
                )

            # load annotations to input sample
            self.load_annotations(input_data, sample.labels)

        if self.skip_empty_samples and input_data.targets.empty:
            return None  # pragma: no cover
        return input_data


class ScalabelMultiSensor(Scalabel):
    """Scalabel format dataset with multiple sensors."""

    # TODO this should be SHIFT

    def __getitem__(self, item):
        frame2id = self.ref_sampler.frame_name_to_idx
        if not self.training:
            group = self.dataset.groups[cur_idx]
            if not self.dataset.multi_sensor_inference:
                cur_data = self.mapper(
                    self.dataset.frames[frame2id[group.frames[0]]],
                    self.training,
                )
                assert cur_data is not None
                return [cur_data]

            group_data = self.mapper(group, self.training)
            assert group_data is not None
            data = [group_data]
            for fname in group.frames:
                cur_data = self.mapper(
                    self.dataset.frames[frame2id[fname]], self.training
                )
                assert cur_data is not None
                data.append(cur_data)
            return data
        else:
            raise NotImplementedError


def add_data_path(
    data_root: str, frames: Union[List[Frame], List[FrameGroup]]
) -> None:
    """Add filepath to frame using data_root."""
    for ann in frames:
        assert ann.name is not None
        if ann.url is None:
            if ann.videoName is not None:
                ann.url = os.path.join(data_root, ann.videoName, ann.name)
            else:
                ann.url = os.path.join(data_root, ann.name)
        else:
            ann.url = os.path.join(data_root, ann.url)


def scalabel_to_boxes3d(
    cls,
    labels: List[Label],
    class_to_idx: Dict[str, int],
    label_id_to_idx: Optional[Dict[str, int]] = None,
    image_size: Optional[ImageSize] = None,
) -> "Boxes3D":
    """Convert from scalabel format to internal."""
    box_list, cls_list, idx_list = [], [], []
    has_class_ids = all((b.category is not None for b in labels))
    for i, label in enumerate(labels):
        box, score, box_cls, l_id = (
            label.box3d,
            label.score,
            label.category,
            label.id,
        )
        if box is None:
            continue
        if has_class_ids:
            if box_cls in class_to_idx:
                cls_list.append(class_to_idx[box_cls])
            else:  # pragma: no cover
                continue

        if score is None:
            box_list.append([*box.location, *box.dimension, *box.orientation])
        else:
            box_list.append(
                [*box.location, *box.dimension, *box.orientation, score]
            )
        idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
        idx_list.append(idx)

    if len(box_list) == 0:  # pragma: no cover
        return cls.empty()
    box_tensor = torch.tensor(box_list, dtype=torch.float32)
    class_ids = (
        torch.tensor(cls_list, dtype=torch.long) if has_class_ids else None
    )
    track_ids = torch.tensor(idx_list, dtype=torch.long)
    return Boxes3D(box_tensor, class_ids, track_ids)


def boxes3d_to_scalabel(
    self, idx_to_class: Optional[Dict[int, str]] = None
) -> List[Label]:
    """Convert from internal to scalabel format."""
    labels = []
    for i in range(len(self.boxes)):
        if self.track_ids is not None:
            label_id = str(self.track_ids[i].item())
        else:
            label_id = str(i)

        rx = float(self.boxes[i, 6])
        ry = float(self.boxes[i, 7])
        rz = float(self.boxes[i, 8])
        if self.boxes.shape[-1] == 10:
            score: Optional[float] = float(self.boxes[i, 9])
        else:
            score = None

        box = Box3D(
            location=[
                float(self.boxes[i, 0]),
                float(self.boxes[i, 1]),
                float(self.boxes[i, 2]),
            ],
            dimension=[
                float(self.boxes[i, 3]),
                float(self.boxes[i, 4]),
                float(self.boxes[i, 5]),
            ],
            orientation=[rx, ry, rz],
            alpha=-1.0,
        )
        label_dict = dict(id=label_id, box3d=box, score=score)

        if idx_to_class is not None:
            cls = idx_to_class[int(self.class_ids[i])]
        else:
            cls = str(int(self.class_ids[i]))  # pragma: no cover
        label_dict["category"] = cls
        labels.append(Label(**label_dict))

    return labels


def scalabel_to_boxes2d(
    cls,
    labels: List[Label],
    class_to_idx: Dict[str, int],
    label_id_to_idx: Optional[Dict[str, int]] = None,
    image_size: Optional[ImageSize] = None,
) -> "Boxes2D":
    """Convert from scalabel format to internal.

    NOTE: The box definition in Scalabel includes x2y2, whereas Vis4D and
    other software libraries like detectron2, mmdet do not include this,
    which is why we convert via box2d_to_xyxy.
    """
    box_list, cls_list, idx_list = [], [], []
    has_class_ids = all((b.category is not None for b in labels))
    for i, label in enumerate(labels):
        box, score, box_cls, l_id = (
            label.box2d,
            label.score,
            label.category,
            label.id,
        )
        if box is None:
            continue
        if has_class_ids:
            if box_cls in class_to_idx:
                cls_list.append(class_to_idx[box_cls])
            else:  # pragma: no cover
                continue

        if score is None:
            box_list.append([*box2d_to_xyxy(box)])
        else:
            box_list.append([*box2d_to_xyxy(box), score])

        idx = label_id_to_idx[l_id] if label_id_to_idx is not None else i
        idx_list.append(idx)

    if len(box_list) == 0:  # pragma: no cover
        return cls.empty()
    box_tensor = torch.tensor(box_list, dtype=torch.float32)
    class_ids = (
        torch.tensor(cls_list, dtype=torch.long) if has_class_ids else None
    )
    track_ids = torch.tensor(idx_list, dtype=torch.long)
    return Boxes2D(box_tensor, class_ids, track_ids)


def boxes2d_to_scalabel(
    self, idx_to_class: Optional[Dict[int, str]] = None
) -> List[Label]:
    """Convert from internal to scalabel format."""
    labels = []
    for i in range(len(self.boxes)):
        if self.track_ids is not None:
            label_id = str(self.track_ids[i].item())
        else:
            label_id = str(i)
        box = xyxy_to_box2d(
            float(self.boxes[i, 0]),
            float(self.boxes[i, 1]),
            float(self.boxes[i, 2]),
            float(self.boxes[i, 3]),
        )
        if self.boxes.shape[-1] == 5:
            score: Optional[float] = float(self.boxes[i, 4])
        else:
            score = None
        label_dict = dict(id=label_id, box2d=box, score=score)

        if idx_to_class is not None:
            cls = idx_to_class[int(self.class_ids[i])]
        else:
            cls = str(int(self.class_ids[i]))  # pragma: no cover
        label_dict["category"] = cls
        labels.append(Label(**label_dict))

    return labels
