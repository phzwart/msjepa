from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import MSJEPAConfig, load_config
from .data import build_dataloader
from .losses import DensePredictionLoss, combine_losses, compute_sigreg_loss, student_teacher_agreement
from .masking import BlockTokenMasker, mask_coverage
from .model import MSJEPA
from .sigreg import compute_feature_health_stats, save_diagnostic_artifacts, SIGRegRegularizer
from .utils import compute_patch_grid


@torch.no_grad()
def run_validation(
    model: MSJEPA,
    dataloader: DataLoader[torch.Tensor],
    criterion: DensePredictionLoss,
    sigreg: SIGRegRegularizer,
    masker: BlockTokenMasker,
    device: torch.device,
    config: MSJEPAConfig,
    epoch: int = 0,
) -> dict[str, float]:
    model.student.eval()
    model.teacher.eval()

    totals = {
        "prediction_loss": 0.0,
        "sigreg_loss": 0.0,
        "total_loss": 0.0,
        "feature_mean": 0.0,
        "feature_std": 0.0,
        "mask_coverage": 0.0,
        "student_teacher_agreement": 0.0,
        "near_flat_channel_percent": 0.0,
        "dead_channel_percent": 0.0,
        "channel_variance_mean": 0.0,
        "channel_variance_min": 0.0,
        "channel_variance_median": 0.0,
        "channel_variance_max": 0.0,
    }
    histogram_counts: list[int] | None = None
    histogram_edges: list[float] | None = None

    grid_size, _ = compute_patch_grid(config.image_size, config.patch_size, config.stride)
    total_batches = 0
    diagnostics_dir = Path(config.checkpoint_dir) / "diagnostics" / "val"

    for step, images in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        seed = config.mask_seed + epoch * max(1, len(dataloader)) + step
        token_mask = masker.generate(images.shape[0], grid_size=grid_size, device=device, seed=seed)
        outputs = model(images, student_token_mask=token_mask)
        student_features = outputs.student.adapted_dense_feature_map
        if student_features is None:
            raise RuntimeError("Student branch must produce adapted dense features for validation.")

        prediction_loss = criterion(student_features, outputs.teacher.dense_feature_map, mask=token_mask)
        sigreg_loss = compute_sigreg_loss(
            regularizer=sigreg,
            sigreg_target=config.sigreg_target,
            dense_features=student_features,
            token_features=outputs.student.latent_token_features,
        )
        loss_terms = combine_losses(prediction_loss, sigreg_loss, config.sigreg_weight)
        dense_stats = compute_feature_health_stats(student_features, flat_channel_threshold=config.flat_channel_threshold)
        dense_stats_dict = dense_stats.to_dict()
        agreement = student_teacher_agreement(
            student_features,
            outputs.teacher.dense_feature_map,
            feature_normalization=criterion.feature_normalization,
        )

        totals["prediction_loss"] += float(loss_terms.prediction_loss.item())
        totals["sigreg_loss"] += float(loss_terms.sigreg_loss.item())
        totals["total_loss"] += float(loss_terms.total_loss.item())
        totals["feature_mean"] += dense_stats_dict["feature_mean"]
        totals["feature_std"] += dense_stats_dict["feature_std"]
        totals["mask_coverage"] += mask_coverage(token_mask)
        totals["student_teacher_agreement"] += agreement
        totals["near_flat_channel_percent"] += dense_stats_dict["near_flat_channel_percent"]
        totals["dead_channel_percent"] += dense_stats_dict["dead_channel_percent"]
        totals["channel_variance_mean"] += dense_stats_dict["channel_variance_mean"]
        totals["channel_variance_min"] += dense_stats_dict["channel_variance_min"]
        totals["channel_variance_median"] += dense_stats_dict["channel_variance_median"]
        totals["channel_variance_max"] += dense_stats_dict["channel_variance_max"]

        if histogram_counts is None:
            histogram_counts = dense_stats_dict["variance_histogram_counts"]
            histogram_edges = dense_stats_dict["variance_histogram_edges"]
        else:
            histogram_counts = [
                current + new for current, new in zip(histogram_counts, dense_stats_dict["variance_histogram_counts"])
            ]

        if step == 0:
            save_diagnostic_artifacts(
                output_dir=diagnostics_dir,
                prefix=f"epoch{epoch + 1:03d}_sample",
                dense_features=student_features,
                stats=dense_stats,
                token_mask=token_mask,
            )
        total_batches += 1

    if total_batches == 0:
        return totals

    averaged = {key: value / total_batches for key, value in totals.items()}
    if histogram_counts is not None and histogram_edges is not None:
        averaged["variance_histogram_counts"] = histogram_counts
        averaged["variance_histogram_edges"] = histogram_edges
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MSJEPA on a folder-based dataset.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--data-root", type=Path, required=True, help="Validation data root.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint to load.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cpu or cuda.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = MSJEPA(config).to(device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])

    dataloader = build_dataloader(
        root=args.data_root,
        image_size=config.image_size,
        in_channels=config.in_channels,
        batch_size=config.batch_size,
        train=False,
    )
    criterion = DensePredictionLoss(
        loss_type=config.prediction_loss_type,
        feature_normalization=config.feature_normalization,
    )
    sigreg = SIGRegRegularizer(flat_channel_threshold=config.flat_channel_threshold)
    masker = BlockTokenMasker(
        mask_ratio=config.mask_ratio,
        block_size=config.mask_block_size,
        seed=config.mask_seed,
    )
    metrics = run_validation(model, dataloader, criterion, sigreg, masker, device, config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
