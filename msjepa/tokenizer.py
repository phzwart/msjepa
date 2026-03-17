from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor, nn

from .utils import compute_patch_grid, to_2tuple


@dataclass(slots=True)
class PatchGridMetadata:
    original_size: tuple[int, int]
    padded_size: tuple[int, int]
    grid_size: tuple[int, int]
    patch_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]


class PatchTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | tuple[int, int] = 8,
        stride: int | tuple[int, int] = 8,
    ) -> None:
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.stride = to_2tuple(stride)
        patch_dim = in_channels * self.patch_size[0] * self.patch_size[1]

        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.stride)
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: Tensor) -> tuple[Tensor, PatchGridMetadata]:
        if images.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] image tensor, got shape {tuple(images.shape)}.")

        height, width = images.shape[-2:]
        grid_size, padding = compute_patch_grid((height, width), self.patch_size, self.stride)
        pad_h, pad_w = padding
        padded = F.pad(images, (0, pad_w, 0, pad_h))

        patches = self.unfold(padded).transpose(1, 2)
        tokens = self.norm(self.proj(patches))

        batch_size = images.shape[0]
        grid_h, grid_w = grid_size
        token_grid = tokens.transpose(1, 2).reshape(batch_size, -1, grid_h, grid_w)

        metadata = PatchGridMetadata(
            original_size=(height, width),
            padded_size=(height + pad_h, width + pad_w),
            grid_size=grid_size,
            patch_size=self.patch_size,
            stride=self.stride,
            padding=padding,
        )
        return token_grid, metadata
