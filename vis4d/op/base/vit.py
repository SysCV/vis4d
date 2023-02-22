"""Visual transformer models for classification."""
from __future__ import annotations

import torch
import torchvision.models.vision_transformer as _vit
from torch import nn

from vis4d.common import ArgsType

from .base import BaseModel


class ViT(BaseModel):
    """Wrapper for torch vision ViT backbones."""

    def __init__(
        self,
        vit_name: str,
        image_size: int = 224,
        patch_size: int | None = None,
        pretrained: bool = False,
        **kwargs: ArgsType,
    ):
        """Initialize the ViT base model from torch vision.

        Args:
            vit_name (str): Name of the ViT variant.
            image_size (int, optional): Size of input image. Defaults to 224.
            patch_size (int, optional): If set, resize the positional embedding
                to match the new patch size. Defaults to None, which keeps
                using the same patch size of the specified ViT variant.
            pretrained (bool, optional): Whether to load ImageNet
                pre-trained weights. Defaults to False.
            **kwargs (ArgsType): Parameters passed to the
                ``torchvision.models.vision_transformer.VisionTransformer``.
        """
        super().__init__()
        if vit_name not in {
            "vit_b_16",
            "vit_b_32",
            "vit_l_16",
            "vit_l_32",
            "vit_h_14",
        }:
            raise ValueError("The ViT name is not supported!")

        vit: nn.Module = _vit.__dict__[vit_name](
            weights="DEFAULT" if pretrained else None
        )

        # Interpolate positional embeddings
        if patch_size is None:
            patch_size = int(vit_name.split("_")[-1])
        model_state = _vit.interpolate_embeddings(
            image_size, patch_size, vit.state_dict()
        )

        self.vit: nn.Module = _vit.__dict__[vit_name](
            image_size=image_size, **kwargs
        )
        self.vit.load_state_dict(model_state)

        self.name = vit_name
        self.patch_size = patch_size

    @property
    def out_channels(self) -> list[int]:
        """Get the number of channels for feature embedding.

        Returns:
            list[int]: number of channels
        """
        if self.name in {"vit_b_16", "vit_b_32"}:
            channels = [768, 768]
        elif self.name in {"vit_l_16", "vit_l_32"}:
            channels = [1024, 1024]
        else:
            channels = [1280, 1280]
        return channels

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Torchvision ViT forward.

        Args:
            images (torch.Tensor): Image input tensor with shape [N, C, H, W].

        Returns:
            embeddings (list[torch.Tensor]): List of embedding tensors. It
                stores two levels of embeddings, before and after the seq-to-
                seq transformer in ViT, both with shape [N, num_patches, dim].
        """
        x = self.vit._process_input(images)  # pylint: disable=protected-access

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Call ViT encoder
        return [x, self.vit.encoder(x)]
