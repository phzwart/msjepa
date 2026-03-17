from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .positional import RelativePositionBias2D
from .utils import DropPath, pad_to_window_size, window_partition, window_reverse


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        dropout: float,
        use_relative_position_bias: bool,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.relative_position_bias = (
            RelativePositionBias2D(window_size, num_heads) if use_relative_position_bias else None
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        batch_windows, window_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, window_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.relative_position_bias is not None:
            attn = attn + self.relative_position_bias().unsqueeze(0)

        if attn_mask is not None:
            num_windows = attn_mask.shape[0]
            attn = attn.view(batch_windows // num_windows, num_windows, self.num_heads, window_tokens, window_tokens)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, window_tokens, window_tokens)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_windows, window_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
        use_relative_position_bias: bool,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            use_relative_position_bias=use_relative_position_bias,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim=dim, hidden_dim=hidden_dim, dropout=dropout)

    def _create_attention_mask(self, height: int, width: int, device: torch.device) -> Tensor:
        img_mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        counter = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = counter
                counter += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[1:3]
        residual = x
        x = self.norm1(x)

        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        shifted_x, (pad_h, pad_w) = pad_to_window_size(shifted_x, self.window_size)
        padded_h, padded_w = shifted_x.shape[1:3]

        attn_mask = None
        if self.shift_size > 0:
            attn_mask = self._create_attention_mask(padded_h, padded_w, shifted_x.device)

        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, attn_mask=attn_mask)
        shifted_x = window_reverse(attn_windows, self.window_size, padded_h, padded_w)

        if pad_h > 0 or pad_w > 0:
            shifted_x = shifted_x[:, :height, :width, :].contiguous()

        if self.shift_size > 0:
            shifted_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = residual + self.drop_path(shifted_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, height, width, channels = x.shape
        pad_h = height % 2
        pad_w = width % 2

        if pad_h > 0 or pad_w > 0:
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, pad_w, 0, pad_h))
            x = x.permute(0, 2, 3, 1).contiguous()
            height, width = x.shape[1:3]

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        return self.reduction(x)


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: Sequence[float],
        use_relative_position_bias: bool,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_path[index],
                    use_relative_position_bias=use_relative_position_bias,
                )
                for index in range(depth)
            ]
        )
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            x = block(x)
        stage_output = x
        if self.downsample is not None:
            x = self.downsample(x)
        return stage_output, x


class SwinEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depths: Sequence[int],
        num_heads: Sequence[int],
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
        use_relative_position_bias: bool,
    ) -> None:
        super().__init__()
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same number of stages.")

        self.num_stages = len(depths)
        self.stage_dims = tuple(embed_dim * (2**stage_idx) for stage_idx in range(self.num_stages))
        self.pos_drop = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path, steps=sum(depths)).tolist()
        layers = []
        cursor = 0
        for stage_idx, (depth, heads) in enumerate(zip(depths, num_heads)):
            layer = BasicLayer(
                dim=self.stage_dims[stage_idx],
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[cursor : cursor + depth],
                use_relative_position_bias=use_relative_position_bias,
                downsample=stage_idx < self.num_stages - 1,
            )
            cursor += depth
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    @property
    def output_channels(self) -> tuple[int, ...]:
        return self.stage_dims

    def forward(self, token_grid: Tensor) -> list[Tensor]:
        if token_grid.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] token grid, got shape {tuple(token_grid.shape)}.")

        x = token_grid.permute(0, 2, 3, 1).contiguous()
        x = self.pos_drop(x)

        outputs: list[Tensor] = []
        for layer in self.layers:
            stage_output, x = layer(x)
            outputs.append(stage_output.permute(0, 3, 1, 2).contiguous())
        return outputs
