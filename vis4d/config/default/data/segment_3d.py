from ml_collections.config_dict import ConfigDict

from vis4d.config.default.data.dataloader_pipeline import get_dataloader_config
from vis4d.config.util import class_config, instantiate_classes

trainer_config = ConfigDict({"batch_size": 64})

dataset = class_config(
    "vis4d.data.datasets.s3dis.S3DIS",
    data_root="/home/zrene/git/vis4d/tests/test_data/s3d_test/",
    keys_to_load=["points3d", "colors3d", "semantics3d"],
)

# This is equivalent to this:
# dataset =  ConfigDict(
#     {
#         "class_path": "vis4d.data.datasets.s3dis.S3DIS",
#         "init_args": ConfigDict(
#             {
#                 "data_root": "/home/zrene/git/vis4d/tests/test_data/s3d_test/",
#                 "keys_to_load": ["points3d", "colors3d", "semantics3d"],
#             }
#         ),
#     }
# )

transforms = [
    class_config("vis4d.data.transforms.points.extract_pc_bounds"),
    class_config(
        "vis4d.data.transforms.point_sampling.sample_points_block_random",
        in_keys=dataset.init_args.get_ref("keys_to_load"),
        out_keys=dataset.init_args.get_ref("keys_to_load"),
    ),
    class_config(
        "vis4d.data.transforms.points.add_norm_noise",
        std=0.01,
    ),
    class_config(
        "vis4d.data.transforms.points.center_and_normalize",
        out_keys=("points3d_normalized",),
        normalize=False,
    ),
    class_config(
        "vis4d.data.transforms.points.move_pts_to_last_channel",
        in_keys=dataset.init_args.get_ref("keys_to_load")
        + ["points3d_normalized"],
        out_keys=dataset.init_args.get_ref("keys_to_load")
        + ["points3d_normalized"],
    ),
    class_config(
        "vis4d.data.transforms.points.concatenate_point_features",
        in_keys=dataset.init_args.get_ref("keys_to_load")
        + ["points3d_normalized"],
        out_keys="points3d",
    ),
]

dataloader = get_dataloader_config(
    transforms, dataset, batch_size=trainer_config.get_ref("batch_size")
)

# Raw config
print(dataloader)

# We can updated reference fields:
# E.g. Change batchsize
print("[Trainer BS] Before: ", trainer_config.batch_size)
print("[DLoader BS] Before: ", dataloader.init_args.samples_per_gpu)

print(" Updating Trainer Batch size")
trainer_config.batch_size = 8
print("Updated values:")
print("[Trainer BS]", trainer_config.batch_size)
print("[DLoader BS]", dataloader.init_args.samples_per_gpu)

print("Transforms Before: ")
for tf in transforms:
    print(tf.class_path)
    if "in_keys" in tf.get("init_args", {}):
        print(" => ", tf.init_args.in_keys)
    if "out_keys" in tf.get("init_args", {}):
        print(" <= ", tf.init_args.out_keys)
    print()

print("Setting dataset init args to ", '"points3d", "semantics3d"')
# Decide we do not want to load color anyway
dataset.init_args.keys_to_load = ["points3d", "semantics3d"]
for tf in transforms:
    print(tf.class_path)
    if "in_keys" in tf.get("init_args", {}):
        print(" => ", tf.init_args.in_keys)
    if "out_keys" in tf.get("init_args", {}):
        print(" <= ", tf.init_args.out_keys)
    print()

# Instantiate classes
# config_dict_with_classes = instantiate_classes(dataloader)
