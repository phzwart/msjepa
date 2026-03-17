from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .utils import to_2tuple


@dataclass(slots=True)
class BlockTokenMasker:
    mask_ratio: float
    block_size: tuple[int, int] = (4, 4)
    seed: int = 42

    def __post_init__(self) -> None:
        self.block_size = to_2tuple(self.block_size)
        if not 0.0 <= self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in the range [0, 1).")

    def generate(
        self,
        batch_size: int,
        grid_size: tuple[int, int],
        device: torch.device | str | None = None,
        seed: int | None = None,
    ) -> Tensor:
        grid_h, grid_w = grid_size
        if self.mask_ratio <= 0.0:
            mask = torch.zeros((batch_size, 1, grid_h, grid_w), dtype=torch.bool)
            return mask if device is None else mask.to(device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed if seed is None else seed)

        target_tokens = max(1, int(round(self.mask_ratio * grid_h * grid_w)))
        block_h = min(self.block_size[0], grid_h)
        block_w = min(self.block_size[1], grid_w)
        masks = torch.zeros((batch_size, 1, grid_h, grid_w), dtype=torch.bool)

        for batch_idx in range(batch_size):
            while int(masks[batch_idx].sum().item()) < target_tokens:
                top = int(torch.randint(0, grid_h - block_h + 1, (1,), generator=generator).item())
                left = int(torch.randint(0, grid_w - block_w + 1, (1,), generator=generator).item())
                masks[batch_idx, :, top : top + block_h, left : left + block_w] = True

        return masks if device is None else masks.to(device)


def apply_token_mask(token_grid: Tensor, token_mask: Tensor, fill_value: float = 0.0) -> Tensor:
    if token_mask.ndim == 3:
        token_mask = token_mask.unsqueeze(1)

    if token_grid.shape[0] != token_mask.shape[0] or token_grid.shape[-2:] != token_mask.shape[-2:]:
        raise ValueError("token_mask must match the token grid batch size and spatial dimensions.")

    return token_grid.masked_fill(token_mask.expand(-1, token_grid.shape[1], -1, -1), fill_value)


def mask_coverage(token_mask: Tensor) -> float:
    return float(token_mask.float().mean().item())
