"""Visual Transformer base model."""
from __future__ import annotations

import torch
import torchvision.models.vision_transformer as _vit
from torch import nn

from vis4d.common import ArgsType

from .base import BaseModel


class TorchVisionViT(BaseModel):
    """Wrapper for torch vision ViT backbones."""

    def __init__(
        self,
        variant: str,
        image_size: int = 224,
        patch_size: int | None = None,
        pretrained: bool = False,
        **kwargs: ArgsType,
    ):
        """Initialize the ViT base model from torch vision.

        Args:
            variant (str): Name of the ViT variant.
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
        assert variant in {
            "vit_t_16",
            "vit_t_32",
            "vit_s_16",
            "vit_s_32",
            "vit_b_16",
            "vit_b_32",
            "vit_l_16",
            "vit_l_32",
            "vit_h_14",
        }, f"Unknown ViT variant: {variant}"
        self.variant = variant
        self.pretrained = pretrained
        if patch_size is None:
            self.patch_size = int(variant.split("_")[-1])
        else:
            self.patch_size = patch_size

        self.vit: nn.Module = self._get_vit_variant(
            image_size=image_size, **kwargs
        )
        if self.pretrained:
            if self.variant[:5] in {"vit_t", "vit_s"}:
                raise ValueError(
                    "This ViT variant does not have pretrained weights!"
                )
            # Interpolate positional embeddings for pretrained weights
            model_state = _vit.interpolate_embeddings(
                image_size, self.patch_size, self._get_pretrained_weights()
            )
            self.vit.load_state_dict(model_state)

        # Only compute gradients for the used parts
        self.vit.heads.head.weight.requires_grad = False
        self.vit.heads.head.bias.requires_grad = False

    def _get_vit_variant(self, **kwargs: ArgsType) -> nn.Module:
        """Get the ViT module based on the variant name."""
        if self.variant[:5] in {"vit_t", "vit_s"}:
            if self.variant == "vit_t_16":
                params = {
                    "patch_size": 16,
                    "hidden_dim": 192,
                    "num_heads": 3,
                    "mlp_dim": 768,
                }
            elif self.variant == "vit_t_32":
                params = {
                    "patch_size": 32,
                    "hidden_dim": 192,
                    "num_heads": 3,
                    "mlp_dim": 768,
                }
            elif self.variant == "vit_s_16":
                params = {
                    "patch_size": 16,
                    "hidden_dim": 384,
                    "num_heads": 6,
                    "mlp_dim": 1536,
                }
            elif self.variant == "vit_s_32":
                params = {
                    "patch_size": 32,
                    "hidden_dim": 384,
                    "num_heads": 6,
                    "mlp_dim": 1536,
                }
            params.update(
                {"num_layers": 12, "weights": None, "progress": False}
            )
            params.update(kwargs)
            vit: nn.Module = (
                _vit._vision_transformer(  # pylint: disable=protected-access
                    **params
                )
            )
        else:
            vit: nn.Module = _vit.__dict__[self.variant](**kwargs)
        return vit

    def _get_pretrained_weights(self):
        """Get the pretrained weights for the ViT variant."""
        vit: nn.Module = _vit.__dict__[self.variant](
            weights="DEFAULT" if self.pretrained else None
        )
        return vit.state_dict()

    @property
    def out_channels(self) -> list[int]:
        """Get the number of channels for feature embedding.

        Returns:
            list[int]: number of channels
        """
        if self.variant in {"vit_t_16", "vit_t_32"}:
            channels = [192, 192]
        elif self.variant in {"vit_s_16", "vit_s_32"}:
            channels = [384, 384]
        elif self.variant in {"vit_b_16", "vit_b_32"}:
            channels = [768, 768]
        elif self.variant in {"vit_l_16", "vit_l_32"}:
            channels = [1024, 1024]
        else:
            channels = [1280, 1280]
        return channels

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Torchvision ViT forward.

        Args:
            images (torch.Tensor): Image input tensor with shape [N, C, H, W].

        Returns:
            embeddings (torch.Tensor): List of embedding tensors. It stores two
                level of embeddings, before and after the seq-to-seq
                transformer in ViT, both with shape [N, num_patches, dim].
        """
        x = self.vit._process_input(images)  # pylint: disable=protected-access

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Call ViT encoder
        return [x, self.vit.encoder(x)]
