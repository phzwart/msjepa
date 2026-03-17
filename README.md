# MSJEPA

MSJEPA is a practical PyTorch project for self-supervised dense representation learning on images. Training is performed only on learned dense features with masked latent prediction; there is no pixel reconstruction target, no MAE-style image decoder loss, and no image-level supervision.

## Architecture overview

- In-model tokenizer with configurable `patch_size` and `stride`
- Single-scale Swin-style encoder with optional absolute positional embeddings and relative position bias
- Light U-Net-style decoder that upsamples latent features into full-resolution dense feature maps
- Student-side predictor MLP for latent adaptation
- Stop-gradient teacher branch updated with EMA
- SIGReg-inspired anti-collapse regularization and dense feature diagnostics

During training:

- masking is applied on the token grid after tokenization
- the student consumes masked tokens
- the teacher consumes the full image by default
- supervision compares student predicted dense features against teacher dense features
- an additional SIGReg-style term discourages flat channels, dead channels, and trivial feature maps

## Stage 2 contents

Implemented:

- block token masking with configurable mask ratio and block size
- teacher-student model wrapper with EMA updates
- dense latent prediction losses: `mse`, `cosine`, and `smooth_l1`
- optional feature normalization: `l2`, `layer_norm`, or `none`
- SIGReg-inspired feature regularization on dense features, token features, or both
- flat-channel and dead-channel diagnostics with lightweight artifact saving hooks
- folder-based dataset for image files and `.npy` / `.npz` arrays
- minimal train and validation entrypoints with checkpointing and logging

Still intentionally omitted:

- pixel reconstruction losses
- MAE-style decoder targets
- qlty tokenization
- multi-scale pathways

## Quick start

```python
import torch

from msjepa import MSJEPA, load_config

config = load_config("configs/debug.yaml")
model = MSJEPA(config)
images = torch.randn(2, config.in_channels, *config.image_size)
outputs = model(images)

print(outputs.student.latent_token_features.shape)
print(outputs.student.dense_feature_map.shape)
print(outputs.student.adapted_dense_feature_map.shape)
print(outputs.teacher.dense_feature_map.shape)
```

## Training

Train on a folder of images:

```bash
python -m msjepa.train --config configs/default.yaml --train-root /path/to/train --val-root /path/to/val
```

Run validation from a checkpoint:

```bash
python -m msjepa.validate --config configs/default.yaml --data-root /path/to/val --checkpoint checkpoints/default/best.pt
```

Validation reports:

- prediction loss
- SIGReg loss
- total loss
- feature mean and std
- mask coverage
- student-teacher agreement
- near-flat and dead-channel percentages
- channel variance summaries and lightweight histograms

## Why SIGReg

Dense latent prediction can drift into unhealthy regimes even when the teacher-student loss is numerically stable. Common failure modes include:

- flat channels with almost no variance across batch and space
- dead channels with tiny variance and near-zero activation magnitude
- trivial dense maps that satisfy the prediction objective without carrying useful spatial structure

The current `msjepa.sigreg` module adds a simple, modular anti-collapse penalty that pushes channel variance away from zero, keeps channel means from drifting too far, and discourages vanishing activation energy. The exact formula is intentionally isolated so it can be refined later without changing the rest of the training code.

## Diagnostics

Validation and periodic training logs monitor feature health on the student dense features. The project tracks:

- per-channel variance across batch and space
- percent near-flat channels
- dead-channel fraction
- dense feature mean and std
- lightweight variance histograms

When `feature_stat_logging_frequency` is greater than zero, the code also saves simple artifacts under the checkpoint directory, including token masks, mean feature maps, per-channel variance summaries, and sample dense feature tensors.

## Stability examples

Conservative dense-only regularization:

```yaml
sigreg_weight: 0.05
sigreg_target: dense
flat_channel_threshold: 0.0001
feature_stat_logging_frequency: 50
```

More aggressive monitoring while debugging collapse:

```yaml
sigreg_weight: 0.1
sigreg_target: both
flat_channel_threshold: 0.0005
feature_stat_logging_frequency: 1
```

## Notes

- Default tokenization uses `patch_size=8` and `stride=8`
- `stride < patch_size` enables overlapping patches
- Supervision is applied only in learned dense feature space
- No pixel reconstruction loss is used anywhere in the training path
