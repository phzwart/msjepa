from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .utils import to_2tuple


@dataclass(slots=True)
class MSJEPAConfig:
    image_size: tuple[int, int] = (224, 224)
    in_channels: int = 3
    patch_size: tuple[int, int] = (8, 8)
    stride: tuple[int, int] = (8, 8)
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 2, 6, 2)
    num_heads: tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path: float = 0.1
    decoder_channels: tuple[int, ...] = (384, 256, 192, 128)
    dense_feature_dim: int = 128
    predictor_hidden_dim: int = 256
    predictor_depth: int = 2
    use_absolute_positional_embedding: bool = False
    use_relative_position_bias: bool = True
    mask_ratio: float = 0.6
    mask_block_size: tuple[int, int] = (4, 4)
    mask_seed: int = 42
    ema_decay: float = 0.996
    prediction_loss_type: str = "mse"
    feature_normalization: str = "l2"
    optimizer: str = "adamw"
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.05
    batch_size: int = 8
    num_epochs: int = 100
    scheduler: str = "cosine"
    warmup: int = 10
    checkpoint_dir: str = "checkpoints"
    use_amp: bool = True
    sigreg_weight: float = 0.05
    sigreg_target: str = "dense"
    flat_channel_threshold: float = 1.0e-4
    feature_stat_logging_frequency: int = 0
    density_prediction_weight: float = 0.0
    decoder_image_skip: bool = False

    def __post_init__(self) -> None:
        self.image_size = to_2tuple(self.image_size)
        self.patch_size = to_2tuple(self.patch_size)
        self.stride = to_2tuple(self.stride)
        self.mask_block_size = to_2tuple(self.mask_block_size)
        self.depths = tuple(int(depth) for depth in self.depths)
        self.num_heads = tuple(int(heads) for heads in self.num_heads)
        self.decoder_channels = tuple(int(channels) for channels in self.decoder_channels)

        if len(self.depths) != len(self.num_heads):
            msg = "depths and num_heads must have the same number of stages."
            raise ValueError(msg)

        if len(self.decoder_channels) not in {1, len(self.depths)}:
            msg = "decoder_channels must contain either one value or one value per encoder stage."
            raise ValueError(msg)

        if any(depth <= 0 for depth in self.depths):
            raise ValueError("All encoder depths must be positive integers.")

        if self.predictor_depth <= 0:
            raise ValueError("predictor_depth must be at least 1.")

        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")

        if not 0.0 <= self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in the range [0, 1).")

        if not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError("ema_decay must be in the range [0, 1].")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")

        if self.batch_size <= 0 or self.num_epochs <= 0:
            raise ValueError("batch_size and num_epochs must be positive.")

        if self.warmup < 0:
            raise ValueError("warmup must be non-negative.")

        if self.sigreg_weight < 0.0:
            raise ValueError("sigreg_weight must be non-negative.")

        if self.sigreg_target not in {"dense", "token", "both"}:
            raise ValueError("sigreg_target must be one of: dense, token, both.")

        if self.flat_channel_threshold <= 0.0:
            raise ValueError("flat_channel_threshold must be positive.")

        if self.feature_stat_logging_frequency < 0:
            raise ValueError("feature_stat_logging_frequency must be non-negative.")

        if self.density_prediction_weight < 0.0:
            raise ValueError("density_prediction_weight must be non-negative.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MSJEPAConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> MSJEPAConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return MSJEPAConfig.from_dict(data)
