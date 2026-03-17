from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def to_2tuple(value: int | Iterable[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    values = tuple(int(v) for v in value)
    if len(values) != 2:
        raise ValueError(f"Expected a pair of integers, got {values}.")
    return values


def compute_patch_grid(
    image_size: tuple[int, int],
    patch_size: tuple[int, int],
    stride: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    height, width = image_size
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride

    grid_h = 1 if height <= patch_h else math.ceil((height - patch_h) / stride_h) + 1
    grid_w = 1 if width <= patch_w else math.ceil((width - patch_w) / stride_w) + 1

    padded_h = max(height, (grid_h - 1) * stride_h + patch_h)
    padded_w = max(width, (grid_w - 1) * stride_w + patch_w)
    pad_h = padded_h - height
    pad_w = padded_w - width
    return (grid_h, grid_w), (pad_h, pad_w)


def stochastic_depth(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        return stochastic_depth(x, self.drop_prob, self.training)


def window_partition(x: Tensor, window_size: int) -> Tensor:
    batch_size, height, width, channels = x.shape
    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, channels)


def window_reverse(windows: Tensor, window_size: int, height: int, width: int) -> Tensor:
    channels = windows.shape[-1]
    batch_size = windows.shape[0] // ((height // window_size) * (width // window_size))
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch_size, height, width, channels)


def pad_to_window_size(x: Tensor, window_size: int) -> tuple[Tensor, tuple[int, int]]:
    height, width = x.shape[1:3]
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)

    x = x.permute(0, 3, 1, 2)
    x = F.pad(x, (0, pad_w, 0, pad_h))
    x = x.permute(0, 2, 3, 1).contiguous()
    return x, (pad_h, pad_w)
