import torch

from vis4d.data.const import CommonKeys
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.data.datasets.scalabel import Scalabel
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    multi_sensor_collate,
)
from vis4d.data.transforms.normalize import normalize_image
from vis4d.data.transforms.pad import pad_image
from vis4d.op.base import ResNet
from vis4d.op.detect.faster_rcnn import FasterRCNNHead
from vis4d.op.detect.rcnn import RoI2Det
from vis4d.op.fpp import FPN
from vis4d.op.track.qdtrack import QDSimilarityHead
from vis4d.unittest.util import get_test_file


def test_inference():
    base = ResNet("resnet50")
    fpn = FPN(base.out_channels[2:], 256)
    faster_rcnn = FasterRCNNHead(num_classes=8)
    transform_detections = RoI2Det(
        faster_rcnn.rcnn_box_encoder, score_threshold=0.05
    )
    similarity_head = QDSimilarityHead()
    # track_memory = QDTrackMemory(memory_limit=10)  # TODO CC-3DT state
    # associate = QDTrackAssociation() # TODO association op
    # box3d_head = Box3DHead() # TODO
    # transform_box3d = RoI2Det3D() # TODO

    data_root = get_test_file("track/bdd100k-samples/images", rel_path="run")
    annotations = get_test_file("track/bdd100k-samples/labels", rel_path="run")
    config = get_test_file("track/bdd100k-samples/config.toml", rel_path="run")
    test_data = DataPipe(
        NuScenes(
            "data/nuscenes_mini/", version="v1.0-mini", split="mini_train"
        ),  # TODO box3d loading
        preprocess_fn=normalize_image(sensors=NuScenes._CAMERAS),
    )
    batch_fn = pad_image(sensors=NuScenes._CAMERAS)
    batch_size = 2
    test_loader = build_inference_dataloaders(
        test_data,
        samples_per_gpu=batch_size,
        workers_per_gpu=0,
        batchprocess_fn=batch_fn,
        collate_fn=multi_sensor_collate,
    )[0]

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # assume: inputs are consecutive frames
            data = data["CAM_FRONT"]
            images = data[CommonKeys.images]
            inputs_hw = data[CommonKeys.original_hw]

            features = base(images)
            features = fpn(features)

            detector_out = faster_rcnn(features, inputs_hw)
            boxes, scores, class_ids = transform_detections(
                *detector_out.roi, detector_out.proposals.boxes, inputs_hw
            )

            box3d_out = box3d_head(features, boxes)
            boxes3d = transform_box3d(box3d_out)
            embeddings = similarity_head(features, boxes)
            from vis4d.vis.image import imshow_bboxes3d

            imshow_bboxes3d(images, boxes3d, data[CommonKeys.intrinsics])

            # cur_memory = track_memory.get_current_tracks(boxes3d.device)
            # track_ids, filter_indices = associate(
            #     boxes3d,
            #     scores,
            #     class_ids,
            #     embeddings,
            #     cur_memory.track_ids,
            #     cur_memory.class_ids,
            #     cur_memory.embeddings,
            # )

            # data = QDTrackState(
            #     track_ids,
            #     box[filter_indices],
            #     score[filter_indices],
            #     cls_id[filter_indices],
            #     embeds[filter_indices],
            # )
            # track_memory.update(data)
