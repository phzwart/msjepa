from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .config import MSJEPAConfig, load_config
from .decoder import DenseFeatureDecoder
from .ema import initialize_teacher_from_student, update_ema
from .masking import apply_token_mask
from .positional import AbsolutePositionEmbedding2D
from .predictor import DensePredictor
from .swin_encoder import SwinEncoder
from .tokenizer import PatchGridMetadata, PatchTokenizer
from .utils import compute_patch_grid


@dataclass(slots=True)
class MSJEPABranchOutput:
    latent_token_features: Tensor
    dense_feature_map: Tensor
    adapted_dense_feature_map: Tensor | None
    token_grid_metadata: PatchGridMetadata
    token_grid: Tensor
    token_mask: Tensor | None = None


@dataclass(slots=True)
class MSJEPAOutput:
    student: MSJEPABranchOutput
    teacher: MSJEPABranchOutput


class MSJEPABranch(nn.Module):
    def __init__(self, config: MSJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = PatchTokenizer(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            patch_size=config.patch_size,
            stride=config.stride,
        )

        grid_size, _ = compute_patch_grid(config.image_size, config.patch_size, config.stride)
        self.absolute_positional_embedding = (
            AbsolutePositionEmbedding2D(embed_dim=config.embed_dim, grid_size=grid_size)
            if config.use_absolute_positional_embedding
            else None
        )

        self.encoder = SwinEncoder(
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            drop_path=config.drop_path,
            use_relative_position_bias=config.use_relative_position_bias,
        )
        self.decoder = DenseFeatureDecoder(
            encoder_channels=self.encoder.output_channels,
            decoder_channels=config.decoder_channels,
            out_channels=config.dense_feature_dim,
            in_channels=config.in_channels,
            use_image_skip=config.decoder_image_skip,
        )
        self.predictor = DensePredictor(
            feature_dim=config.dense_feature_dim,
            hidden_dim=config.predictor_hidden_dim,
            depth=config.predictor_depth,
        )
        # Optional auxiliary head: dense map -> input channels (e.g. density); used when density_prediction_weight > 0
        self.density_head = nn.Conv2d(config.dense_feature_dim, config.in_channels, kernel_size=1)

    def forward(self, images: Tensor, token_mask: Tensor | None = None, use_predictor: bool = True) -> MSJEPABranchOutput:
        token_grid, metadata = self.tokenizer(images)

        if token_mask is not None:
            token_grid = apply_token_mask(token_grid, token_mask)

        if self.absolute_positional_embedding is not None:
            token_grid = token_grid + self.absolute_positional_embedding(token_grid)

        encoder_features = self.encoder(token_grid)
        latent_token_features = encoder_features[-1].flatten(2).transpose(1, 2).contiguous()
        dense_feature_map = self.decoder(
            encoder_features,
            output_size=metadata.original_size,
            image=images if self.config.decoder_image_skip else None,
        )
        adapted_dense_feature_map = self.predictor(dense_feature_map) if use_predictor else None

        return MSJEPABranchOutput(
            latent_token_features=latent_token_features,
            dense_feature_map=dense_feature_map,
            adapted_dense_feature_map=adapted_dense_feature_map,
            token_grid_metadata=metadata,
            token_grid=token_grid,
            token_mask=token_mask,
        )


class MSJEPA(nn.Module):
    def __init__(self, config: MSJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.student = MSJEPABranch(config)
        self.teacher = copy.deepcopy(self.student)
        initialize_teacher_from_student(self.student, self.teacher)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MSJEPA":
        return cls(load_config(path))

    def update_teacher(self, decay: float | None = None) -> None:
        update_ema(self.student, self.teacher, decay if decay is not None else self.config.ema_decay)

    def forward_student(self, images: Tensor, token_mask: Tensor | None = None) -> MSJEPABranchOutput:
        return self.student(images, token_mask=token_mask, use_predictor=True)

    def forward_teacher(self, images: Tensor, token_mask: Tensor | None = None) -> MSJEPABranchOutput:
        with torch.no_grad():
            return self.teacher(images, token_mask=token_mask, use_predictor=False)

    def forward(
        self,
        images: Tensor,
        student_token_mask: Tensor | None = None,
        teacher_images: Tensor | None = None,
        teacher_token_mask: Tensor | None = None,
    ) -> MSJEPAOutput:
        teacher_images = images if teacher_images is None else teacher_images
        student_output = self.forward_student(images, token_mask=student_token_mask)
        teacher_output = self.forward_teacher(teacher_images, token_mask=teacher_token_mask)
        return MSJEPAOutput(
            student=student_output,
            teacher=teacher_output,
        )
