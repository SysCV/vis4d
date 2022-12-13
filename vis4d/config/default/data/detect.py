from ml_collections.config_dict import ConfigDict

from vis4d.config.default.data.dataloader_pipeline import get_dataloader_config
from vis4d.config.util import class_config, instantiate_classes
from vis4d.data.io.hdf5 import HDF5Backend

trainer_config = ConfigDict({"batch_size": 64})

dataset = class_config(
    "vis4d.data.datasets.coco.COCO",
    data_root="/home/zrene/git/vis4d/tests/test_data/coco_test/",
    split="train",
    data_backend=HDF5Backend(),  # you can also use classes directly
)

# from torch.utils.data import DataLoader, Dataset

# from vis4d.data.loader import (
#     DataPipe,
#     build_inference_dataloaders,
#     build_train_dataloader,
# )
# from vis4d.data.transforms.base import compose, random_apply
# from vis4d.data.transforms.flip import flip_boxes2d, flip_image
# from vis4d.data.transforms.normalize import normalize_image
# from vis4d.data.transforms.pad import pad_image

#     """Default train preprocessing pipeline for detectors."""
#     resize_trans = [resize_image(im_hw, keep_ratio=True), resize_boxes2d()]
#     flip_trans = [flip_image(), flip_boxes2d()]
#     if with_mask:
#         resize_trans += [resize_masks()]
#         flip_trans += [flip_image()]
#     preprocess_fn = compose(
#         [*resize_trans, random_apply(flip_trans), normalize_image()]
#     )
#     batchprocess_fn = pad_image()
#     datapipe = DataPipe(datasets, preprocess_fn)
#     train_loader = build_train_dataloader(
#         datapipe,
#         samples_per_gpu=batch_size,
#         workers_per_gpu=num_workers,
#         batchprocess_fn=batchprocess_fn,
#     )
#     return train_loader
general_config = ConfigDict(
    {"image": {"height": 800, "width": 1333}, "augment_ratio": 0.5}
)

augment_transforms = [
    class_config("vis4d.data.transforms.flip.flip_image"),
    class_config("vis4d.data.transforms.flip.flip_boxes2d"),
]

transforms = [
    class_config(
        "vis4d.data.transforms.resize.resize_image",
        shape=[
            general_config.image.get_ref("height"),
            general_config.image.get_ref("width"),
        ],
        keep_ratio=True,
    ),
    class_config("vis4d.data.transforms.resize.resize_masks"),
    class_config(
        "vis4d.data.transforms.base.random_apply",
        transforms=augment_transforms,
        p=general_config.get_ref("augment_ratio"),
    ),
    class_config("vis4d.data.transforms.normalize.normalize_image"),
]
dataloader = get_dataloader_config(
    transforms,
    dataset,
    batch_size=64,
    batchprocess_fn=class_config("vis4d.data.transforms.pad.pad_image"),
)

config_dict_with_classes = instantiate_classes(dataloader)
