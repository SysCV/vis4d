"""Testcases for ResNet base model."""

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from vis4d.op.base import ResNet, ResNetV1c


def test_resnet():
    """Test ResNet."""
    for r, resnet in enumerate([ResNet, ResNetV1c]):
        for resnet_name in ("resnet18", "resnet34", "resnet50", "resnet101"):
            if r == 1:
                resnet_name = f"{resnet_name}_v1c"
            model = resnet(resnet_name, pretrained=False)
            assert (
                model.deep_stem if resnet == ResNetV1c else not model.deep_stem
            )
            if resnet_name in {"resnet18", "resnet34"}:
                assert model.out_channels == [3, 3, 64, 128, 256, 512]
            elif resnet_name in {"resnet50", "resnet101"}:
                assert model.out_channels == [3, 3, 256, 512, 1024, 2048]
            assert model.trainable_layers == 5

            # test forward
            images = torch.randn(1, 3, 128, 128)
            outs = model(images)
            assert len(outs) == 6
            assert outs[0].shape == outs[1].shape == images.shape
            if resnet_name in {"resnet18", "resnet34"}:
                assert outs[2].shape == (1, 64, 32, 32)
                assert outs[3].shape == (1, 128, 16, 16)
                assert outs[4].shape == (1, 256, 8, 8)
                assert outs[5].shape == (1, 512, 4, 4)
            elif resnet_name in {"resnet50", "resnet101"}:
                assert outs[2].shape == (1, 256, 32, 32)
                assert outs[3].shape == (1, 512, 16, 16)
                assert outs[4].shape == (1, 1024, 8, 8)
                assert outs[5].shape == (1, 2048, 4, 4)

            # test trainable layers
            for i in range(1, 5):
                model.trainable_layers = i
                model.train()
                for j in range(1, 5):
                    if j < 5 - i:
                        assert not getattr(model, f"layer{j}").training
                    else:
                        assert getattr(model, f"layer{j}").training

            # test norm freezed
            model.trainable_layers = 5
            model.norm_frozen = True
            model.train()
            for m in model.modules():
                if isinstance(m, _BatchNorm):
                    assert not m.training
            model.norm_frozen = False
            model.train()
            for m in model.modules():
                if isinstance(m, _BatchNorm):
                    assert m.training

            # test freeze stages
            model.trainable_layers = 0
            model.train()
            if model.deep_stem:
                for param in model.stem.parameters():
                    assert not param.requires_grad
            else:
                for param in model.conv1.parameters():
                    assert not param.requires_grad
                for param in model.bn1.parameters():
                    assert not param.requires_grad
            for param in model.layer1.parameters():
                assert not param.requires_grad
            for param in model.layer2.parameters():
                assert not param.requires_grad
            for param in model.layer3.parameters():
                assert not param.requires_grad
            for param in model.layer4.parameters():
                assert not param.requires_grad
