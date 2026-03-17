from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import amp
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .config import MSJEPAConfig, load_config
from .data import build_dataloader
from .losses import DensePredictionLoss, combine_losses, compute_sigreg_loss, density_prediction_loss
from .masking import BlockTokenMasker, mask_coverage
from .model import MSJEPA
from .sigreg import compute_feature_health_stats, save_diagnostic_artifacts, SIGRegRegularizer
from .utils import compute_patch_grid
from .validate import run_validation


def build_optimizer(model: MSJEPA, config: MSJEPAConfig) -> Optimizer:
    parameters = [parameter for parameter in model.student.parameters() if parameter.requires_grad]
    optimizer_name = config.optimizer.lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")


def build_scheduler(
    optimizer: Optimizer,
    config: MSJEPAConfig,
    steps_per_epoch: int,
) -> LambdaLR | None:
    scheduler_name = config.scheduler.lower()
    if scheduler_name == "none":
        return None

    total_steps = max(1, config.num_epochs * max(1, steps_per_epoch))
    warmup_steps = config.warmup * max(1, steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if scheduler_name == "constant":
            return 1.0
        raise ValueError(f"Unsupported scheduler: {config.scheduler}.")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    checkpoint_path: Path,
    model: MSJEPA,
    optimizer: Optimizer,
    scheduler: LambdaLR | None,
    config: MSJEPAConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": config.to_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def train_epoch(
    model: MSJEPA,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    optimizer: Optimizer,
    scheduler: LambdaLR | None,
    criterion: DensePredictionLoss,
    sigreg: SIGRegRegularizer,
    masker: BlockTokenMasker,
    scaler: amp.GradScaler,
    device: torch.device,
    config: MSJEPAConfig,
    epoch: int,
) -> dict[str, float]:
    model.student.train()
    model.teacher.eval()

    grid_size, _ = compute_patch_grid(config.image_size, config.patch_size, config.stride)
    totals = {
        "prediction_loss": 0.0,
        "sigreg_loss": 0.0,
        "total_loss": 0.0,
        "mask_coverage": 0.0,
        "near_flat_channel_percent": 0.0,
        "dead_channel_percent": 0.0,
    }
    log_interval = max(1, len(dataloader) // 10)
    use_amp = config.use_amp and device.type == "cuda"
    diagnostics_frequency = config.feature_stat_logging_frequency
    diagnostics_dir = Path(config.checkpoint_dir) / "diagnostics" / "train"

    for step, images in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        seed = config.mask_seed + epoch * max(1, len(dataloader)) + step
        token_mask = masker.generate(images.shape[0], grid_size=grid_size, device=device, seed=seed)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, student_token_mask=token_mask)
            student_features = outputs.student.adapted_dense_feature_map
            if student_features is None:
                raise RuntimeError("Student branch must produce adapted dense features during training.")
            prediction_loss = criterion(student_features, outputs.teacher.dense_feature_map, mask=token_mask)
            sigreg_loss = compute_sigreg_loss(
                regularizer=sigreg,
                sigreg_target=config.sigreg_target,
                dense_features=student_features,
                token_features=outputs.student.latent_token_features,
            )
            loss_terms = combine_losses(prediction_loss, sigreg_loss, config.sigreg_weight)
            total_loss = loss_terms.total_loss
            if getattr(config, "density_prediction_weight", 0.0) > 0:
                density_pred = model.student.density_head(outputs.student.dense_feature_map)
                total_loss = total_loss + config.density_prediction_weight * density_prediction_loss(density_pred, images)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.update_teacher(config.ema_decay)
        if scheduler is not None:
            scheduler.step()

        dense_stats = compute_feature_health_stats(student_features, flat_channel_threshold=config.flat_channel_threshold)

        totals["prediction_loss"] += float(loss_terms.prediction_loss.item())
        totals["sigreg_loss"] += float(loss_terms.sigreg_loss.item())
        totals["total_loss"] += float(total_loss.item())
        totals["mask_coverage"] += mask_coverage(token_mask)
        totals["near_flat_channel_percent"] += float(dense_stats.to_dict()["near_flat_channel_percent"])
        totals["dead_channel_percent"] += float(dense_stats.to_dict()["dead_channel_percent"])

        should_log_diagnostics = diagnostics_frequency > 0 and (
            step == 0 or (step + 1) % diagnostics_frequency == 0 or (step + 1) == len(dataloader)
        )
        if should_log_diagnostics:
            save_diagnostic_artifacts(
                output_dir=diagnostics_dir,
                prefix=f"epoch{epoch + 1:03d}_step{step + 1:05d}",
                dense_features=student_features,
                stats=dense_stats,
                token_mask=token_mask,
            )

        if (step + 1) % log_interval == 0 or step == 0 or (step + 1) == len(dataloader):
            stats_dict = dense_stats.to_dict()
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                        "prediction_loss": float(loss_terms.prediction_loss.item()),
                        "sigreg_loss": float(loss_terms.sigreg_loss.item()),
                        "total_loss": float(loss_terms.total_loss.item()),
                        "mask_coverage": mask_coverage(token_mask),
                        "near_flat_channel_percent": stats_dict["near_flat_channel_percent"],
                        "dead_channel_percent": stats_dict["dead_channel_percent"],
                        "channel_variance_mean": stats_dict["channel_variance_mean"],
                    }
                )
            )

    return {key: value / max(1, len(dataloader)) for key, value in totals.items()}


def fit(
    config: MSJEPAConfig,
    train_root: Path,
    val_root: Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_dataloader(
        root=train_root,
        image_size=config.image_size,
        in_channels=config.in_channels,
        batch_size=config.batch_size,
        train=True,
    )
    val_loader = (
        build_dataloader(
            root=val_root,
            image_size=config.image_size,
            in_channels=config.in_channels,
            batch_size=config.batch_size,
            train=False,
        )
        if val_root is not None
        else None
    )

    model = MSJEPA(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
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
    scaler = amp.GradScaler(device="cuda", enabled=config.use_amp and device.type == "cuda")

    checkpoint_dir = Path(config.checkpoint_dir)
    best_val_loss = float("inf")
    history: dict[str, Any] = {"train": [], "val": []}

    for epoch in range(config.num_epochs):
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            sigreg=sigreg,
            masker=masker,
            scaler=scaler,
            device=device,
            config=config,
            epoch=epoch,
        )
        history["train"].append(train_metrics)

        validation_metrics = None
        if val_loader is not None:
            validation_metrics = run_validation(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                sigreg=sigreg,
                masker=masker,
                device=device,
                config=config,
                epoch=epoch,
            )
            history["val"].append(validation_metrics)

        metrics = {"train": train_metrics, "val": validation_metrics}
        print(json.dumps({"epoch": epoch + 1, **metrics}, indent=2))

        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, scheduler, config, epoch, metrics)
        if validation_metrics is not None and validation_metrics["prediction_loss"] < best_val_loss:
            best_val_loss = validation_metrics["prediction_loss"]
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, config, epoch, metrics)

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MSJEPA with masked dense latent prediction.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--train-root", type=Path, required=True, help="Training data root directory.")
    parser.add_argument("--val-root", type=Path, default=None, help="Optional validation data root directory.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cpu or cuda.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fit(config=config, train_root=args.train_root, val_root=args.val_root, device=device)


if __name__ == "__main__":
    main()
