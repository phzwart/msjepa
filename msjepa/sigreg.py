from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn


@dataclass(slots=True)
class FeatureHealthStats:
    """Lightweight diagnostics for collapse-prone feature distributions."""

    channel_variance: Tensor
    channel_mean: Tensor
    channel_abs_mean: Tensor
    near_flat_mask: Tensor
    dead_mask: Tensor
    feature_mean: float
    feature_std: float
    variance_histogram_counts: list[int]
    variance_histogram_edges: list[float]

    def to_dict(self) -> dict[str, float | int | list[int] | list[float]]:
        """Return JSON-friendly diagnostics for logging and checkpoint metadata."""
        return {
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "near_flat_channel_fraction": float(self.near_flat_mask.float().mean().item()),
            "near_flat_channel_percent": float(self.near_flat_mask.float().mean().mul(100.0).item()),
            "dead_channel_fraction": float(self.dead_mask.float().mean().item()),
            "dead_channel_percent": float(self.dead_mask.float().mean().mul(100.0).item()),
            "channel_variance_mean": float(self.channel_variance.mean().item()),
            "channel_variance_min": float(self.channel_variance.min().item()),
            "channel_variance_median": float(self.channel_variance.median().item()),
            "channel_variance_max": float(self.channel_variance.max().item()),
            "variance_histogram_counts": self.variance_histogram_counts,
            "variance_histogram_edges": self.variance_histogram_edges,
        }


def _flatten_feature_channels(features: Tensor) -> Tensor:
    if features.ndim == 4:
        return features.permute(1, 0, 2, 3).reshape(features.shape[1], -1)
    if features.ndim == 3:
        return features.permute(2, 0, 1).reshape(features.shape[2], -1)
    if features.ndim == 2:
        return features.transpose(0, 1).contiguous()
    raise ValueError(f"Expected a [B, C, H, W], [B, N, C], or [B, C] tensor, got {tuple(features.shape)}.")


def compute_feature_health_stats(
    features: Tensor,
    flat_channel_threshold: float,
    histogram_bins: int = 8,
) -> FeatureHealthStats:
    """Summarize feature health across batch and spatial/token dimensions."""
    flattened = _flatten_feature_channels(features.detach())
    channel_mean = flattened.mean(dim=1)
    channel_variance = flattened.var(dim=1, unbiased=False)
    channel_abs_mean = flattened.abs().mean(dim=1)

    near_flat_mask = channel_variance <= flat_channel_threshold
    dead_mask = near_flat_mask & (channel_abs_mean <= flat_channel_threshold)

    histogram_max = max(float(channel_variance.max().item()), flat_channel_threshold * 10.0, 1.0e-6)
    histogram = torch.histc(channel_variance.float().cpu(), bins=histogram_bins, min=0.0, max=histogram_max)
    histogram_edges = torch.linspace(0.0, histogram_max, steps=histogram_bins + 1)

    return FeatureHealthStats(
        channel_variance=channel_variance,
        channel_mean=channel_mean,
        channel_abs_mean=channel_abs_mean,
        near_flat_mask=near_flat_mask,
        dead_mask=dead_mask,
        feature_mean=float(features.mean().item()),
        feature_std=float(features.std(unbiased=False).item()),
        variance_histogram_counts=[int(value) for value in histogram.tolist()],
        variance_histogram_edges=[float(value) for value in histogram_edges.tolist()],
    )


class SIGRegRegularizer(nn.Module):
    """SIGReg-inspired anti-collapse regularizer for learned features.

    The current version favors healthy per-channel variance, small global channel
    means, and non-trivial activation energy. It is intentionally modular so the
    exact formulation can evolve without touching the training loop.

    TODO: Revisit the weighting scheme once longer training traces are available.
    TODO: Explore target-aware variants that condition the regularizer on mask layout.
    """

    def __init__(
        self,
        flat_channel_threshold: float = 1.0e-4,
        variance_weight: float = 1.0,
        mean_weight: float = 0.1,
        activation_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if flat_channel_threshold <= 0.0:
            raise ValueError("flat_channel_threshold must be positive.")

        self.flat_channel_threshold = flat_channel_threshold
        self.variance_weight = variance_weight
        self.mean_weight = mean_weight
        self.activation_weight = activation_weight

    def forward(self, features: Tensor) -> Tensor:
        flattened = _flatten_feature_channels(features)
        channel_mean = flattened.mean(dim=1)
        channel_variance = flattened.var(dim=1, unbiased=False)
        channel_abs_mean = flattened.abs().mean(dim=1)

        variance_penalty = F.relu(self.flat_channel_threshold - channel_variance).mean()
        mean_penalty = channel_mean.pow(2).mean()
        activation_penalty = F.relu(self.flat_channel_threshold - channel_abs_mean).mean()

        return (
            self.variance_weight * variance_penalty
            + self.mean_weight * mean_penalty
            + self.activation_weight * activation_penalty
        )


def resolve_sigreg_feature_targets(
    sigreg_target: str,
    dense_features: Tensor | None = None,
    token_features: Tensor | None = None,
) -> dict[str, Tensor]:
    """Select which student-side features receive SIGReg."""
    target = sigreg_target.lower()
    selected: dict[str, Tensor] = {}

    if target in {"dense", "both"}:
        if dense_features is None:
            raise ValueError("dense_features must be provided when sigreg_target includes dense.")
        selected["dense"] = dense_features

    if target in {"token", "both"}:
        if token_features is None:
            raise ValueError("token_features must be provided when sigreg_target includes token.")
        selected["token"] = token_features

    if not selected:
        raise ValueError(f"Unsupported sigreg_target: {sigreg_target}.")

    return selected


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1.0e-8:
        return np.zeros_like(array, dtype=np.uint8)
    normalized = (array - min_value) / (max_value - min_value)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def save_mask_visualization(token_mask: Tensor, path: str | Path) -> None:
    """Save a simple grayscale rendering of the first token mask."""
    mask = token_mask.detach().float().cpu()
    if mask.ndim == 4:
        mask = mask[0, 0]
    elif mask.ndim == 3:
        mask = mask[0]
    else:
        raise ValueError(f"Expected [B, 1, H, W] or [B, H, W] mask, got {tuple(token_mask.shape)}.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.numpy() * 255.0).astype(np.uint8)).save(output_path)


def save_mean_feature_map(features: Tensor, path: str | Path) -> None:
    """Save the first dense feature map collapsed over channels."""
    if features.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] tensor, got {tuple(features.shape)}.")

    feature_map = features[0].mean(dim=0).detach().cpu().numpy()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_normalize_to_uint8(feature_map)).save(output_path)


def save_channel_variance_summary(stats: FeatureHealthStats, path: str | Path) -> None:
    """Persist channel variance diagnostics as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = stats.to_dict()
    summary["per_channel_variance"] = [float(value) for value in stats.channel_variance.detach().cpu().tolist()]
    summary["per_channel_abs_mean"] = [float(value) for value in stats.channel_abs_mean.detach().cpu().tolist()]
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def save_dense_feature_tensor_sample(features: Tensor, path: str | Path) -> None:
    """Save the first dense feature tensor for offline inspection."""
    if features.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] tensor, got {tuple(features.shape)}.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features[0].detach().cpu(), output_path)


def save_diagnostic_artifacts(
    output_dir: str | Path,
    prefix: str,
    dense_features: Tensor,
    stats: FeatureHealthStats,
    token_mask: Tensor | None = None,
) -> dict[str, str]:
    """Save lightweight visual diagnostics for debugging collapse and masking."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}
    if token_mask is not None:
        mask_path = output_path / f"{prefix}_mask.png"
        save_mask_visualization(token_mask, mask_path)
        saved_paths["mask"] = str(mask_path)

    mean_map_path = output_path / f"{prefix}_mean_feature.png"
    variance_summary_path = output_path / f"{prefix}_variance_summary.json"
    dense_tensor_path = output_path / f"{prefix}_dense_feature.pt"

    save_mean_feature_map(dense_features, mean_map_path)
    save_channel_variance_summary(stats, variance_summary_path)
    save_dense_feature_tensor_sample(dense_features, dense_tensor_path)

    saved_paths["mean_feature_map"] = str(mean_map_path)
    saved_paths["variance_summary"] = str(variance_summary_path)
    saved_paths["dense_feature_tensor"] = str(dense_tensor_path)
    return saved_paths
