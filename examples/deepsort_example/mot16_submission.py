"""MOT16 challenge submission."""
import json
import os
import shutil
from collections import defaultdict

# pylint: skip-file

output_dir = (
    "/home/yinjiang/systm/examples/deepsort_example/mot16_test_submission/"
)

# output_dir = (
#     "/home/yinjiang/systm/examples/deepsort_example/mot16_train_submission/"
# )

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)


test_video_names = [
    "MOT16-" + idx for idx in ["01", "03", "06", "07", "08", "12", "14"]
]
for video_name in test_video_names:
    file = open(os.path.join(output_dir, video_name + ".txt"), "w")
    file.close()

# train_video_names = [
#     "MOT16-" + idx for idx in ["02", "04", "05", "09", "10", "11", "13"]
# ]
# for video_name in train_video_names:
#     file = open(os.path.join(output_dir, video_name + ".txt"), "w")
#     file.close()

result_file = "/home/yinjiang/systm/openmt-workspace/DeepSORT/2021-06-29_07:24:39/MOT16_test/track_predictions.json"
results = json.load(
    open(
        result_file,
        "r",
    )
)["frames"]


def tlbr_to_tlwh(x1, y1, x2, y2):
    """Convert bounding box."""
    return x1, y1, x2 - x1, y2 - y1


res_dict = defaultdict(list)
for res in results:
    l = res["url"].split("/")
    video_name = l[3]
    frame_id = int(l[-1][:-4])
    assert video_name in test_video_names
    # assert video_name in train_video_names
    if res["labels"] is None:
        continue
    for track in res["labels"]:
        box2d = track["box2d"]
        x1, y1, w, h = tlbr_to_tlwh(
            box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
        )
        track_id = int(track["id"]) + 1
        res_dict[video_name].extend(
            [
                str(frame_id) + ",",
                str(track_id) + ",",
                str(x1) + ",",
                str(y1) + ",",
                str(w) + ",",
                str(h) + ",",
                str(1) + ",",
                str(1) + ",",
                str(1) + "\n",
            ]
        )
for k, v in res_dict.items():
    file_name = os.path.join(output_dir, k + ".txt")
    with open(file_name, "w") as f:
        f.writelines(v)
