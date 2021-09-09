from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.coder import BaseBBoxCoder
import torch
import numpy as np


def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    idx = (rot[:, 1] > rot[:, 5]).float()
    alpha1 = torch.atan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(rot[:, 6] / rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


@BBOX_CODERS.register_module()
class Box3DCoder(BaseBBoxCoder):
    """3D bounding box coder."""

    def __init__(
        self,
        dep_log_scale=2.0,
        mean_center=[0.0, 0.0],
        mean_dep=30.0,
        std_center=[1.0, 1.0],
        std_dep=20.0,
        anchors=[[1.63, 1.52, 3.88]],
        depth="log",
    ):
        super(BaseBBoxCoder, self).__init__()
        self.mean_center = mean_center
        self.std_center = std_center
        self.mean_depth = mean_dep
        self.std_depth = std_dep
        self.depth_log_scale = dep_log_scale
        assert depth in [
            "log",
            "inv",
            "abs",
        ], "possible depth methods: log, inv, abs"
        self.depth_method = depth
        self.anchors = torch.FloatTensor(anchors)

    def encode(self, bboxes, gt_bboxes, gt_labels, img_meta):
        K = torch.FloatTensor(img_meta["img_info"]["calib"]).to(bboxes.device)
        gx = (gt_bboxes[:, 0] / gt_bboxes[:, 2]) * K[0, 0] + K[
            0, 2
        ]  # .clamp(0, img_meta['img_shape'][1]-1)
        gy = (gt_bboxes[:, 1] / gt_bboxes[:, 2]) * K[1, 1] + K[
            1, 2
        ]  # .clamp(0, img_meta['img_shape'][0]-1)
        px = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        py = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
        pw = bboxes[:, 2] - bboxes[:, 0]
        ph = bboxes[:, 3] - bboxes[:, 1]
        delta_center = torch.stack([(gx - px) / pw, (gy - py) / ph], -1)

        means = delta_center.new_tensor(self.mean_center).unsqueeze(0)
        stds = delta_center.new_tensor(self.std_center).unsqueeze(0)
        delta_center = delta_center.sub_(means).div_(stds)

        depth = self.encode_depth(gt_bboxes[:, 2:3])
        anchors = self.anchors.to(bboxes.device)[gt_labels.long()]
        dimensions = torch.log(gt_bboxes[:, 3:6] / anchors)

        rot_y = gt_bboxes[:, 6]
        # roty2alpha
        alpha = rot_y - torch.atan2(gx - K[0, 2], K[0, 0])
        alpha = alpha % (2 * np.pi) - np.pi
        bin_cls = torch.zeros((alpha.shape[0], 2), device=bboxes.device)
        bin_res = torch.zeros((alpha.shape[0], 2), device=bboxes.device)
        for i in range(alpha.shape[0]):
            if alpha[i] < np.pi / 6.0 or alpha[i] > 5 * np.pi / 6.0:
                bin_cls[i, 0] = 1
                bin_res[i, 0] = alpha[i] - (-0.5 * np.pi)

            if alpha[i] > -np.pi / 6.0 or alpha[i] < -5 * np.pi / 6.0:
                bin_cls[i, 1] = 1
                bin_res[i, 1] = alpha[i] - (0.5 * np.pi)

        return torch.cat(
            [delta_center, depth, dimensions, bin_cls, bin_res], -1
        )

    def decode(self, proposals, pred_bboxes, pred_labels, img_meta):
        K = torch.FloatTensor(img_meta["img_info"]["calib"]).to(
            pred_bboxes.device
        )
        pred_bboxes = pred_bboxes[
            torch.arange(pred_bboxes.shape[0]), pred_labels
        ]

        delta_center = pred_bboxes[:, 0:2]
        means = pred_bboxes.new_tensor(self.mean_center)
        stds = pred_bboxes.new_tensor(self.std_center)
        delta_center = delta_center * stds + means

        # Use network energy to shift the center of each roi
        px = (proposals[:, 0] + proposals[:, 2]) * 0.5
        py = (proposals[:, 1] + proposals[:, 3]) * 0.5
        pw = proposals[:, 2] - proposals[:, 0]
        ph = proposals[:, 3] - proposals[:, 1]
        delta_center[:, 0] = px + pw * delta_center[:, 0]
        delta_center[:, 1] = py + ph * delta_center[:, 1]
        depth = self.decode_depth(pred_bboxes[:, 2])
        center = (
            torch.cat(
                [delta_center, torch.ones_like(delta_center)[..., 0:1]], -1
            )
            @ torch.inverse(K).T
        )
        center *= depth.unsqueeze(-1)

        anchors = self.anchors.to(pred_bboxes.device)[pred_labels.long()]
        dimensions = torch.exp(pred_bboxes[:, 3:6]) * anchors

        orientation = pred_bboxes[:, 6:14]
        # bin 1
        divider1 = torch.sqrt(
            orientation[:, 2:3] ** 2 + orientation[:, 3:4] ** 2
        )
        b1sin = orientation[:, 2:3] / divider1
        b1cos = orientation[:, 3:4] / divider1

        # bin 2
        divider2 = torch.sqrt(
            orientation[:, 6:7] ** 2 + orientation[:, 7:8] ** 2
        )
        b2sin = orientation[:, 6:7] / divider2
        b2cos = orientation[:, 7:8] / divider2

        rot = torch.cat(
            [
                orientation[:, 0:2],
                b1sin,
                b1cos,
                orientation[:, 4:6],
                b2sin,
                b2cos,
            ],
            1,
        )

        # alpha2rot_y
        rot_y = get_alpha(rot) + torch.atan2(
            delta_center[..., 0] - K[0, 2], K[0, 0]
        )
        rot_y = rot_y % (2 * np.pi) - np.pi

        if pred_bboxes.shape[-1] > 14:  # if with confidence
            confidence = pred_bboxes[:, 14:15]
        else:
            confidence = torch.ones_like(pred_bboxes[:, :1])

        return torch.cat(
            [confidence, center, dimensions, rot_y.unsqueeze(-1)], -1
        )

    def encode_depth(self, depth_in):
        if self.depth_method == "inv":
            return 1.0 / depth_in
        elif self.depth_method == "log":
            return torch.log10(depth_in) * self.depth_log_scale
        elif self.depth_method == "abs":
            return (depth_in - self.mean_depth) / self.std_depth

    def decode_depth(self, depth_in):
        if self.depth_method == "inv":
            return 1.0 / depth_in
        elif self.depth_method == "log":
            return torch.pow(10, depth_in / self.depth_log_scale)
        elif self.depth_method == "abs":
            return self.mean_depth + depth_in * self.std_depth
