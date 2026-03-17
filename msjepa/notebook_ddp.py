"""DDP workers for notebook training. Lives in an importable module so spawn can pickle them."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from msjepa import MSJEPAConfig
from msjepa.data import FolderImageDataset, build_augmentations
from msjepa.losses import (
    DensePredictionLoss,
    combine_losses,
    compute_sigreg_loss,
    density_prediction_loss,
    student_teacher_agreement,
)
from msjepa.masking import BlockTokenMasker, mask_coverage
from msjepa.model import MSJEPA
from msjepa.sigreg import SIGRegRegularizer
from msjepa.utils import compute_patch_grid


def _to_images(batch):
    return batch[0] if isinstance(batch, (list, tuple)) else batch


class _ImageLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return (_to_images(b) for b in self.loader)

    def __len__(self):
        return len(self.loader)


def load_pt_train_val(
    data_pt_path: str | Path,
    val_pt_path: str | Path | None = None,
    val_fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load train and val tensors from .pt file(s)."""
    data = torch.load(Path(data_pt_path), map_location="cpu", weights_only=False)
    if val_pt_path is not None:
        val_data = torch.load(Path(val_pt_path), map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "train" in data:
            train_t = data["train"]
        elif isinstance(data, torch.Tensor):
            train_t = data
        else:
            train_t = data.get("images") or data.get("data")
        if isinstance(val_data, dict) and "val" in val_data:
            val_t = val_data["val"]
        elif isinstance(val_data, torch.Tensor):
            val_t = val_data
        else:
            val_t = val_data.get("images") or val_data.get("data")
        return train_t, val_t
    if isinstance(data, dict) and "train" in data and "val" in data:
        return data["train"], data["val"]
    if isinstance(data, torch.Tensor):
        n = len(data)
        n_val = max(1, int(n * val_fraction))
        return data[:-n_val], data[-n_val:]
    for key in ("images", "data"):
        if isinstance(data, dict) and key in data:
            t = data[key]
            n = len(t)
            n_val = max(1, int(n * val_fraction))
            return t[:-n_val], t[-n_val:]
    raise ValueError(
        "data_pt_path must point to a .pt containing a tensor [N,C,H,W] or a dict with "
        "'train'/'val', 'images', or 'data'. For separate val file set val_pt_path."
    )


def _run_ddp_training_loop(
    rank: int,
    world_size: int,
    device: torch.device,
    config: MSJEPAConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_sampler: DistributedSampler,
    num_epochs: int,
    checkpoint_dir: str | Path,
) -> None:
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
    use_amp = getattr(config, "use_amp", True)
    scaler = amp.GradScaler(device="cuda", enabled=use_amp)
    train_loader_images = _ImageLoader(train_loader)
    val_loader_images = _ImageLoader(val_loader)
    grid_size, _ = compute_patch_grid(config.image_size, config.patch_size, config.stride)
    density_weight = getattr(config, "density_prediction_weight", 0.0)
    ckpt_dir = Path(checkpoint_dir)
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"DDP training started, {num_epochs} epochs", flush=True)

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.student.train()
        model.teacher.eval()
        train_totals = {"prediction_loss": 0.0, "sigreg_loss": 0.0, "density_loss": 0.0, "total_loss": 0.0, "mask_coverage": 0.0}
        n_train = 0

        for step, images in enumerate(train_loader_images):
            images = images.to(device, non_blocking=True)
            seed = config.mask_seed + epoch * max(1, len(train_loader)) + step
            token_mask = masker.generate(images.shape[0], grid_size=grid_size, device=device, seed=seed)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images, student_token_mask=token_mask)
                student_features = outputs.student.adapted_dense_feature_map
                prediction_loss = criterion(student_features, outputs.teacher.dense_feature_map, mask=token_mask)
                sigreg_loss = compute_sigreg_loss(sigreg, config.sigreg_target, dense_features=student_features, token_features=outputs.student.latent_token_features)
                loss_terms = combine_losses(prediction_loss, sigreg_loss, config.sigreg_weight)
                total_loss = loss_terms.total_loss
                density_loss_val = torch.tensor(0.0, device=images.device)
                if density_weight > 0:
                    density_pred = model.student.module.density_head(outputs.student.dense_feature_map)
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
        for key in ("prediction_loss", "sigreg_loss", "total_loss", "agreement"):
            t = torch.tensor([val_totals[key]], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            val_totals[key] = t.item()
        n_val_t = torch.tensor([n_val], dtype=torch.int64, device=device)
        dist.all_reduce(n_val_t, op=dist.ReduceOp.SUM)
        n_val_global = n_val_t.item()
        val_metrics = {k: val_totals[k] / n_val_global for k in ("prediction_loss", "sigreg_loss", "total_loss", "agreement")}

        if rank == 0:
            print(f"epoch {epoch + 1}/{num_epochs}  train loss {train_metrics['total_loss']:.4f}  val loss {val_metrics['total_loss']:.4f}  val agreement {val_metrics['agreement']:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1}.pt"
            state = model.state_dict()
            if hasattr(model.student, "module"):
                state = {k.replace("student.module.", "student.", 1): v for k, v in state.items()}
            torch.save({
                "epoch": epoch + 1,
                "model": state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config.to_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

    dist.destroy_process_group()


def _set_dist_env(rank: int, world_size: int, master_addr: str = "127.0.0.1", master_port: int = 29500) -> None:
    """Set env vars required for env:// rendezvous when using multiprocessing.spawn (e.g. from a notebook)."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", master_addr)
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(master_port))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)


def run_ddp_worker_folders(
    rank: int,
    world_size: int,
    config_dict: dict,
    train_root: str | Path,
    val_root: str | Path,
    num_epochs: int,
    checkpoint_dir: str | Path,
) -> None:
    """DDP worker with folder-based data. Used by spawn; must live in an importable module."""
    _set_dist_env(rank, world_size)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    config = MSJEPAConfig.from_dict(config_dict)
    model = MSJEPA(config)
    model.student = DDP(
        model.student.to(device),
        device_ids=[rank],
        find_unused_parameters=True,  # e.g. density head when density_prediction_weight=0
    )
    model.teacher = model.teacher.to(device)

    train_dataset = FolderImageDataset(
        root=train_root,
        image_size=config.image_size,
        in_channels=config.in_channels,
        augmentations=build_augmentations(True),
    )
    val_dataset = FolderImageDataset(
        root=val_root,
        image_size=config.image_size,
        in_channels=config.in_channels,
        augmentations=build_augmentations(False),
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    _run_ddp_training_loop(
        rank=rank, world_size=world_size, device=device, config=config, model=model,
        train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler,
        num_epochs=num_epochs, checkpoint_dir=checkpoint_dir,
    )


def run_ddp_worker_pt(
    rank: int,
    world_size: int,
    config_dict: dict,
    data_pt_path: str | Path,
    val_pt_path: str | Path | None,
    val_fraction: float,
    num_epochs: int,
    checkpoint_dir: str | Path,
) -> None:
    """DDP worker with data from .pt file(s). Used by spawn; must live in an importable module."""
    _set_dist_env(rank, world_size)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    config = MSJEPAConfig.from_dict(config_dict)
    model = MSJEPA(config)
    model.student = DDP(
        model.student.to(device),
        device_ids=[rank],
        find_unused_parameters=True,  # e.g. density head when density_prediction_weight=0
    )
    model.teacher = model.teacher.to(device)

    train_t, val_t = load_pt_train_val(data_pt_path, val_pt_path=val_pt_path, val_fraction=val_fraction)
    train_dataset = TensorDataset(train_t)
    val_dataset = TensorDataset(val_t)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    _run_ddp_training_loop(
        rank=rank, world_size=world_size, device=device, config=config, model=model,
        train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler,
        num_epochs=num_epochs, checkpoint_dir=checkpoint_dir,
    )
