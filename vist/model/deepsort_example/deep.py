"""deep featureNet."""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

# pylint: disable= invalid-name


class BasicBlock(nn.Module):  # type: ignore
    """Basic build block."""

    def __init__(
        self, c_in: int, c_out: int, is_downsample: bool = False
    ) -> None:
        """Init."""
        super().__init__()
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

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward."""
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(
    c_in: int, c_out: int, repeat_times: int, is_downsample: bool = False
) -> nn.Sequential:
    """Make layers."""
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
    """Deep feature net."""

    def __init__(self, num_classes: int = 625):
        """Init."""
        super().__init__()
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
        self.batch_norm = nn.BatchNorm1d(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self.norm = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        """Forward function of feature net.

        output size: N x 128
        """
        x = torch.div(x, 255.0)
        x = self.norm(x)
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        if not train:
            x = self.dense[0](x)
            x = self.dense[1](x)
            # x is normalized to a unit sphere
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        x = self.dense(x)
        # N x 128
        # classifier
        x = self.classifier(x)
        return x
