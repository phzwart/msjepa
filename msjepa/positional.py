from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import to_2tuple


class AbsolutePositionEmbedding2D(nn.Module):
    def __init__(self, embed_dim: int, grid_size: int | tuple[int, int]) -> None:
        super().__init__()
        self.grid_size = to_2tuple(grid_size)
        self.embedding = nn.Parameter(torch.zeros(1, embed_dim, *self.grid_size))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] tensor, got shape {tuple(x.shape)}.")

        if x.shape[-2:] == self.embedding.shape[-2:]:
            return self.embedding

        return F.interpolate(
            self.embedding,
            size=x.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )


class RelativePositionBias2D(nn.Module):
    def __init__(self, window_size: int | tuple[int, int], num_heads: int) -> None:
        super().__init__()
        self.window_size = to_2tuple(window_size)
        window_h, window_w = self.window_size
        table_size = (2 * window_h - 1) * (2 * window_w - 1)

        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coords_h = torch.arange(window_h)
        coords_w = torch.arange(window_w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_h - 1
        relative_coords[:, :, 1] += window_w - 1
        relative_coords[:, :, 0] *= 2 * window_w - 1
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index, persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self) -> Tensor:
        window_area = self.window_size[0] * self.window_size[1]
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(window_area, window_area, -1)
        return bias.permute(2, 0, 1).contiguous()
