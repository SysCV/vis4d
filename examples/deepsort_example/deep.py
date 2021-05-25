from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2


class BasicBlock(nn.Module):  # type: ignore
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(
            c_out, c_out, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out),
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out),
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [
                BasicBlock(c_in, c_out, is_downsample=is_downsample),
            ]
        else:
            blocks += [
                BasicBlock(c_out, c_out),
            ]
    return nn.Sequential(*blocks)


class FeatureNet(nn.Module):  # type: ignore
    def __init__(self, num_classes=625, reid=False):
        super(FeatureNet, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(32, 32, repeat_times=2, is_downsample=False)
        # 32 64 32
        self.layer2 = make_layers(32, 64, repeat_times=2, is_downsample=True)
        # 64 32 16
        self.layer3 = make_layers(64, 128, repeat_times=2, is_downsample=True)
        # 128 16 8
        self.dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(128 * 16 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
        )
        # 256 1 1
        self.reid = reid
        self.batch_norm = nn.BatchNorm1d(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of feature net.

        output size: N x 128
        """
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        if self.reid:
            x = self.dense[0](x)
            x = self.dense[1](x)
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        x = self.dense(x)
        # B x 128
        # classifier
        x = self.classifier(x)
        return x


class FeatureExtractor(object):
    def __init__(self, model_weight_path, use_cuda=True):
        self.net = FeatureNet(reid=True)
        self.device = (
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        state_dict = torch.load(
            model_weight_path, map_location=lambda storage, loc: storage
        )["net_dict"]
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def _preprocess(self, im_crops):
        """
        1. to float with scale from 0 to 1
        2. resize to (64, 128) as Market1501 dataset did
        3. concatenate to a numpy array
        3. to torch Tensor
        4. normalize

        input im_crops: torch.Tensor of shape 3xHxW
        """

        def _resize(im, size):
            im = im.numpy().astype(np.float32)
            resized = cv2.resize(im / 255.0, size)
            return resized

        im_batch = []
        for im in im_crops:
            im = np.transpose(im, (1, 2, 0))
            im = _resize(im, self.size)
            im = self.norm(im).unsqueeze(0)
            im_batch.append(im)
        im_batch = torch.cat(im_batch, dim=0).float()

        return im_batch

    def __call__(self, im_crops: List[torch.Tensor]):
        """im_crops shape: 3xHxW"""
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features


if __name__ == "__main__":
    net = FeatureNet(reid=True)
    x = torch.randn(4, 3, 128, 64)
    y = net(x)
    print("shape y： ", y.shape)
