from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .sigreg import SIGRegRegularizer, resolve_sigreg_feature_targets


class DensePredictionLoss(nn.Module):
    def __init__(self, loss_type: str = "mse", feature_normalization: str = "l2") -> None:
        super().__init__()
        self.loss_type = loss_type.lower()
        self.feature_normalization = feature_normalization.lower()

        if self.loss_type not in {"mse", "cosine", "smooth_l1"}:
            raise ValueError(f"Unsupported prediction loss type: {loss_type}.")

        if self.feature_normalization not in {"l2", "layer_norm", "none"}:
            raise ValueError(f"Unsupported feature normalization: {feature_normalization}.")

    def _normalize(self, features: Tensor) -> Tensor:
        if self.feature_normalization == "none":
            return features
        if self.feature_normalization == "l2":
            return F.normalize(features, dim=1, eps=1.0e-6)

        normalized = F.layer_norm(features.permute(0, 2, 3, 1), (features.shape[1],))
        return normalized.permute(0, 3, 1, 2).contiguous()

    def _loss_map(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        if self.loss_type == "mse":
            return (student_features - teacher_features).pow(2).mean(dim=1, keepdim=True)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(student_features, teacher_features, reduction="none").mean(dim=1, keepdim=True)
        return 1.0 - F.cosine_similarity(student_features, teacher_features, dim=1, eps=1.0e-6).unsqueeze(1)

    def forward(self, student_features: Tensor, teacher_features: Tensor, mask: Tensor | None = None) -> Tensor:
        student_features = self._normalize(student_features)
        teacher_features = self._normalize(teacher_features.detach())
        loss_map = self._loss_map(student_features, teacher_features)

        if mask is None:
            return loss_map.mean()

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()
        if mask.shape[-2:] != loss_map.shape[-2:]:
            mask = F.interpolate(mask, size=loss_map.shape[-2:], mode="nearest")

        if float(mask.sum().item()) == 0.0:
            return loss_map.mean()

        return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)


@dataclass(slots=True)
class LossBreakdown:
    prediction_loss: Tensor
    sigreg_loss: Tensor
    total_loss: Tensor


def compute_sigreg_loss(
    regularizer: SIGRegRegularizer,
    sigreg_target: str,
    dense_features: Tensor | None = None,
    token_features: Tensor | None = None,
) -> Tensor:
    selected_features = resolve_sigreg_feature_targets(
        sigreg_target=sigreg_target,
        dense_features=dense_features,
        token_features=token_features,
    )
    losses = [regularizer(features) for features in selected_features.values()]
    return torch.stack(losses).mean() if len(losses) > 1 else losses[0]


def combine_losses(prediction_loss: Tensor, sigreg_loss: Tensor, sigreg_weight: float) -> LossBreakdown:
    total_loss = prediction_loss + sigreg_weight * sigreg_loss
    return LossBreakdown(
        prediction_loss=prediction_loss,
        sigreg_loss=sigreg_loss,
        total_loss=total_loss,
    )


def feature_statistics(features: Tensor) -> dict[str, float]:
    return {
        "feature_mean": float(features.mean().item()),
        "feature_std": float(features.std(unbiased=False).item()),
    }


def student_teacher_agreement(
    student_features: Tensor,
    teacher_features: Tensor,
    feature_normalization: str = "l2",
) -> float:
    criterion = DensePredictionLoss(loss_type="cosine", feature_normalization=feature_normalization)
    cosine_distance = criterion(student_features, teacher_features)
    return float(1.0 - cosine_distance.item())


def density_prediction_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """Auxiliary MSE loss so dense features are encouraged to carry local density/intensity."""
    return F.mse_loss(predicted, target)
