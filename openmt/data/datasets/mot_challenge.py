"""Load MOTChallenge data and convert to scalabel format."""
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

from fvcore.common.timer import Timer
from scalabel.label.typing import Frame, Label

from .scalabel import prepare_scalabel_frames

logger = logging.getLogger(__name__)

# Classes in MOT:
#   1: 'pedestrian'
#   2: 'person on vehicle'
#   3: 'car'
#   4: 'bicycle'
#   5: 'motorbike'
#   6: 'non motorized vehicle'
#   7: 'static person'
#   8: 'distractor'
#   9: 'occluder'
#   10: 'occluder on the ground',
#   11: 'occluder full'
#   12: 'reflection'
DISCARD = [3, 4, 5, 6, 9, 10, 11]


def get_file_as_list(filepath: str) -> List[str]:
    """Get contents of a text file as list."""
    with open(filepath, "r") as f:
        contents = f.readlines()
    return contents


def parse_annotations(
    ann_filepath: str, name_mapping: Optional[Dict[str, str]] = None
) -> Dict[int, List[Label]]:
    """Parse annotation file into List of Scalabel Label type per frame."""
    outputs = defaultdict(list)
    for line in get_file_as_list(ann_filepath):
        gt = line.strip().split(",")
        class_id = gt[7]
        if int(class_id) in DISCARD:
            continue
        if name_mapping is not None:
            class_id = name_mapping[class_id]
        frame_id, ins_id = map(int, gt[:2])
        bbox = list(map(float, gt[2:6]))
        box2d = dict(
            x1=bbox[0], y1=bbox[1], x2=bbox[0] + bbox[2], y2=bbox[1] + bbox[3]
        )
        attributes = dict(visibility=gt[8])
        ann = Label(
            category=class_id,
            id=ins_id,
            box_2d=box2d,
            attributes=attributes,
        )
        outputs[frame_id].append(ann)
    return outputs


def convert_and_load_motchallenge(
    annotation_path: str,
    image_root: str,
    dataset_name: Optional[str] = None,
    ignore_categories: Optional[List[str]] = None,
    name_mapping: Optional[Dict[str, str]] = None,
    prepare_frames: bool = True,
) -> List[Frame]:
    """Convert motchallenge annotations to scalabel format and prepare them."""
    timer = Timer()
    frames = []
    for video in os.listdir(annotation_path):
        img_names = sorted(os.listdir(os.path.join(image_root, video, "img1")))
        annotations = parse_annotations(
            os.path.join(image_root, video, "gt/gt.txt"), name_mapping
        )

        for i, img_name in enumerate(img_names):
            frame = Frame(
                name=os.path.join("img1", img_name),
                video_name=video,
                frame_index=i,
                labels=annotations[i] if i in annotations else None,
            )
            frames.append(frame)

    logger.info(
        "Converting %s to Scalabel format takes %s seconds.",
        dataset_name,
        "{:.2f}".format(timer.seconds()),
    )
    if prepare_frames:
        prepare_scalabel_frames(
            frames, image_root, dataset_name, ignore_categories
        )
    return frames
