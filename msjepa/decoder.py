from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DenseFeatureDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        out_channels: int,
        in_channels: int = 0,
        use_image_skip: bool = False,
    ) -> None:
        super().__init__()
        if len(decoder_channels) == 1:
            decoder_channels = tuple(decoder_channels) * len(encoder_channels)

        if len(encoder_channels) != len(decoder_channels):
            raise ValueError("decoder_channels must contain one value per encoder stage or a single shared value.")

        self.use_image_skip = use_image_skip and in_channels > 0
        reversed_encoder_channels = list(reversed(tuple(encoder_channels)))
        reversed_decoder_channels = list(decoder_channels)

        self.lateral_projections = nn.ModuleList(
            [
                ConvNormAct(inc, outc, kernel_size=1)
                for inc, outc in zip(reversed_encoder_channels, reversed_decoder_channels)
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [
                ConvNormAct(reversed_decoder_channels[idx - 1] + reversed_decoder_channels[idx], reversed_decoder_channels[idx])
                for idx in range(1, len(reversed_decoder_channels))
            ]
        )
        self.refine = ConvNormAct(reversed_decoder_channels[-1], reversed_decoder_channels[-1])
        head_in = reversed_decoder_channels[-1] + (in_channels if self.use_image_skip else 0)
        self.image_fusion = (
            ConvNormAct(head_in, reversed_decoder_channels[-1], kernel_size=1) if self.use_image_skip else None
        )
        self.head = nn.Sequential(
            ConvNormAct(reversed_decoder_channels[-1], reversed_decoder_channels[-1]),
            nn.Conv2d(reversed_decoder_channels[-1], out_channels, kernel_size=1),
        )

    def forward(
        self,
        encoder_features: Sequence[Tensor],
        output_size: tuple[int, int],
        image: Tensor | None = None,
    ) -> Tensor:
        if not encoder_features:
            raise ValueError("encoder_features must contain at least one feature map.")

        features = list(reversed(tuple(encoder_features)))
        x = self.lateral_projections[0](features[0])

        for idx, skip in enumerate(features[1:], start=1):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            skip = self.lateral_projections[idx](skip)
            x = self.fusion_blocks[idx - 1](torch.cat([x, skip], dim=1))

        x = self.refine(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        if self.use_image_skip and image is not None:
            if image.shape[-2:] != output_size:
                image = F.interpolate(image, size=output_size, mode="bilinear", align_corners=False)
            x = torch.cat([x, image], dim=1)
            x = self.image_fusion(x)
        return self.head(x)
