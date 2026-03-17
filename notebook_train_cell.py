# Single notebook cell: train MSJEPA with your train_loader / val_loader.
# Assumes: config, model, train_loader, val_loader are already defined.
# Optional: set num_epochs and checkpoint_dir before running, or they come from config.

import math
from pathlib import Path

import torch
from torch import amp

from msjepa import MSJEPAConfig
from msjepa.losses import DensePredictionLoss, combine_losses, compute_sigreg_loss, density_prediction_loss, student_teacher_agreement
from msjepa.masking import BlockTokenMasker, mask_coverage
from msjepa.model import MSJEPA
from msjepa.sigreg import SIGRegRegularizer, compute_feature_health_stats
from msjepa.utils import compute_patch_grid


def _to_images(batch):
    """Unwrap (x, y) or (x,) to x so your dataloader format works."""
    return batch[0] if isinstance(batch, (list, tuple)) else batch


class _ImageLoader:
    """Thin wrapper so train/val loops get images only."""
    def __init__(self, loader): self.loader = loader
    def __iter__(self): return (_to_images(b) for b in self.loader)
    def __len__(self): return len(self.loader)


# --- Setup (use your existing config and model) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
model = model.to(device)
num_epochs = getattr(config, "num_epochs", 10)
checkpoint_dir = Path(getattr(config, "checkpoint_dir", "checkpoints"))
checkpoint_dir.mkdir(parents=True, exist_ok=True)

optimizer = torch.optim.AdamW(
    [p for p in model.student.parameters() if p.requires_grad],
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
steps_per_epoch = max(1, len(train_loader))
total_steps = num_epochs * steps_per_epoch
warmup_steps = config.warmup * steps_per_epoch

def _lr_lambda(step):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
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
# AMP only on CUDA; MPS uses float32 by default
use_amp = getattr(config, "use_amp", True) and device.type == "cuda"
scaler = amp.GradScaler(device="cuda", enabled=use_amp)

train_loader_images = _ImageLoader(train_loader)
val_loader_images = _ImageLoader(val_loader)
grid_size, _ = compute_patch_grid(config.image_size, config.patch_size, config.stride)

# --- Training loop (one cycle = one epoch; print and save every epoch) ---
for epoch in range(num_epochs):
    model.student.train()
    model.teacher.eval()
    train_totals = {"prediction_loss": 0.0, "sigreg_loss": 0.0, "density_loss": 0.0, "total_loss": 0.0, "mask_coverage": 0.0}
    n_train = 0
    density_weight = getattr(config, "density_prediction_weight", 0.0)

    for step, images in enumerate(train_loader_images):
        images = images.to(device, non_blocking=True)
        seed = config.mask_seed + epoch * max(1, len(train_loader)) + step
        token_mask = masker.generate(images.shape[0], grid_size=grid_size, device=device, seed=seed)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, student_token_mask=token_mask)
            student_features = outputs.student.adapted_dense_feature_map
            prediction_loss = criterion(student_features, outputs.teacher.dense_feature_map, mask=token_mask)
            sigreg_loss = compute_sigreg_loss(sigreg, config.sigreg_target, dense_features=student_features, token_features=outputs.student.latent_token_features)
            loss_terms = combine_losses(prediction_loss, sigreg_loss, config.sigreg_weight)
            total_loss = loss_terms.total_loss
            density_loss_val = torch.tensor(0.0, device=images.device)
            if density_weight > 0:
                density_pred = model.student.density_head(outputs.student.dense_feature_map)
                density_loss_val = density_prediction_loss(density_pred, images)
                total_loss = total_loss + density_weight * density_loss_val

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.update_teacher(config.ema_decay)
        scheduler.step()

        train_totals["prediction_loss"] += loss_terms.prediction_loss.item()
        train_totals["sigreg_loss"] += loss_terms.sigreg_loss.item()
        train_totals["density_loss"] += density_loss_val.item()
        train_totals["total_loss"] += total_loss.item()
        train_totals["mask_coverage"] += mask_coverage(token_mask)
        n_train += 1

    n_train = max(1, n_train)
    train_metrics = {k: v / n_train for k, v in train_totals.items()}

    # Validation
    model.student.eval()
    model.teacher.eval()
    val_totals = {"prediction_loss": 0.0, "sigreg_loss": 0.0, "total_loss": 0.0, "agreement": 0.0}
    n_val = 0
    with torch.no_grad():
        for step, images in enumerate(val_loader_images):
            images = images.to(device, non_blocking=True)
            seed = config.mask_seed + (epoch + 1000) * max(1, len(val_loader)) + step
            token_mask = masker.generate(images.shape[0], grid_size=grid_size, device=device, seed=seed)
            outputs = model(images, student_token_mask=token_mask)
            student_features = outputs.student.adapted_dense_feature_map
            prediction_loss = criterion(student_features, outputs.teacher.dense_feature_map, mask=token_mask)
            sigreg_loss = compute_sigreg_loss(sigreg, config.sigreg_target, dense_features=student_features, token_features=outputs.student.latent_token_features)
            loss_terms = combine_losses(prediction_loss, sigreg_loss, config.sigreg_weight)
            agreement = student_teacher_agreement(student_features, outputs.teacher.dense_feature_map, feature_normalization=config.feature_normalization)
            val_totals["prediction_loss"] += loss_terms.prediction_loss.item()
            val_totals["sigreg_loss"] += loss_terms.sigreg_loss.item()
            val_totals["total_loss"] += loss_terms.total_loss.item()
            val_totals["agreement"] += agreement
            n_val += 1
    n_val = max(1, n_val)
    val_metrics = {k: v / n_val for k, v in val_totals.items()}

    # Print every cycle (epoch)
    print(f"epoch {epoch + 1}/{num_epochs}  train loss {train_metrics['total_loss']:.4f}  val loss {val_metrics['total_loss']:.4f}  val agreement {val_metrics['agreement']:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}")

    # Checkpoint every cycle
    ckpt_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.to_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }, ckpt_path)
    print(f"  saved {ckpt_path}")

print("done.")
