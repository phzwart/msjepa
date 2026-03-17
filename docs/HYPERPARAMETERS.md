# MSJEPA configuration and hyperparameters

This document describes every option in `MSJEPAConfig` (and thus in YAML configs such as `configs/default.yaml`), what it controls, and how changing it affects the model or training.

---

## 1. Input and tokenization

| Parameter       | Type            | Default      | What it changes |
|----------------|-----------------|--------------|-----------------|
| **image_size** | `(int, int)`    | `(224, 224)` | Spatial size of input images `(H, W)`. Must be consistent with your data. Token grid size is derived from this, `patch_size`, and `stride`. |
| **in_channels**| `int`           | `3`          | Number of input channels (e.g. 3 for RGB). Used by the patch tokenizer to compute patch dimension and by the first conv/unfold. |
| **patch_size** | `(int, int)`    | `(8, 8)`     | Height and width of each patch. Smaller values give more tokens and higher resolution at the cost of compute and memory. |
| **stride**     | `(int, int)`    | `(8, 8)`     | Stride of the sliding window for patch extraction. `stride < patch_size` yields overlapping patches and more tokens; `stride == patch_size` is non-overlapping. |

**Notes:** The tokenizer produces a grid of tokens. Grid dimensions are computed so that patches fit (with optional padding). No pixel decoder or reconstruction loss is used; only this token grid is used for masking and encoding.

---

## 2. Encoder (Swin-style backbone)

| Parameter                          | Type              | Default        | What it changes |
|------------------------------------|-------------------|----------------|-----------------|
| **embed_dim**                      | `int`             | `96`           | Token embedding dimension and channel size of the first encoder stage. Each subsequent stage doubles: stage dims are `embed_dim * 2^0, 2^1, 2^2, 2^3` (e.g. 96, 192, 384, 768 for 4 stages). Larger values increase capacity and compute. |
| **depths**                         | `tuple[int, ...]` | `(2, 2, 6, 2)` | Number of Swin Transformer blocks in each of the four stages. Length must equal `num_heads`. Deeper stages add capacity; typically the third stage is deepest. |
| **num_heads**                      | `tuple[int, ...]` | `(3, 6, 12, 24)` | Number of attention heads per stage. Length must equal `depths`. Stage dimension must be divisible by `num_heads` (stage dim = `embed_dim * 2^stage_idx`). |
| **window_size**                    | `int`             | `7`            | Side length of each non-overlapping window for windowed self-attention. Must be positive. Smaller windows reduce memory and long-range interaction; token grid size must be compatible with windowing (padding is applied internally where needed). |
| **mlp_ratio**                      | `float`           | `4.0`          | Ratio of the MLP hidden dimension to the stage dimension in each Swin block (hidden = `dim * mlp_ratio`). Controls feed-forward capacity. |
| **dropout**                        | `float`           | `0.0`          | Dropout rate used in the encoder (attention and MLP). |
| **drop_path**                      | `float`           | `0.1`          | Stochastic depth (drop path) probability applied to residual branches in the encoder. Helps regularization in deeper networks. |
| **use_absolute_positional_embedding** | `bool`         | `False`        | If `True`, add learned 2D absolute positional embeddings to the token grid before the encoder. |
| **use_relative_position_bias**     | `bool`            | `True`         | If `True`, use learned relative position bias inside each attention window (Swin-style). Usually kept `True`. |

---

## 3. Decoder and dense features

| Parameter               | Type              | Default                    | What it changes |
|-------------------------|-------------------|----------------------------|-----------------|
| **decoder_channels**    | `tuple[int, ...]` | `(384, 256, 192, 128)`     | Channel sizes in the U-Net-style decoder, one per encoder stage (order matches reversed encoder stages). Can be a single integer to use the same size for all stages. Decoder fuses multi-scale encoder features and upsamples to the final dense map. |
| **dense_feature_dim**   | `int`             | `128`                      | Number of channels in the final dense feature map produced by both student and teacher. This is the representation dimension used for the prediction loss and (optionally) SIGReg. |

---

## 4. Student predictor

| Parameter                | Type  | Default | What it changes |
|--------------------------|-------|---------|-----------------|
| **predictor_hidden_dim** | `int` | `256`   | Hidden channel size in the predictor MLP (1×1 conv layers). Only used when `predictor_depth > 1`. |
| **predictor_depth**      | `int` | `2`     | Number of 1×1 conv layers in the predictor (at least 1). The student’s dense features are passed through this before comparing to the teacher; the teacher has no predictor. |

---

## 5. Masking (student branch)

| Parameter         | Type            | Default    | What it changes |
|-------------------|-----------------|------------|-----------------|
| **mask_ratio**    | `float`         | `0.6`      | Target fraction of tokens to mask (in [0, 1)). More masking increases difficulty and can improve learning; too high can hurt. |
| **mask_block_size** | `(int, int)`  | `(4, 4)`   | Height and width of each masked block. Block masks are placed randomly until the mask reaches approximately `mask_ratio` of the grid. Larger blocks give contiguous masked regions. |
| **mask_seed**     | `int`           | `42`       | Random seed for mask generation. Can be overridden per call for reproducibility or variation. |

---

## 6. Teacher and EMA

| Parameter    | Type  | Default | What it changes |
|-------------|-------|---------|-----------------|
| **ema_decay** | `float` | `0.996` | Exponential moving average decay for the teacher: `teacher = decay * teacher + (1 - decay) * student`. Closer to 1 updates the teacher more slowly; lower values make the teacher track the student faster. Must be in [0, 1]. |

---

## 7. Loss and feature normalization

| Parameter                  | Type   | Default | What it changes |
|----------------------------|--------|---------|-----------------|
| **prediction_loss_type**   | `str`  | `"mse"` | Loss between student predicted dense features and teacher dense features. Options: `"mse"`, `"cosine"`, `"smooth_l1"`. |
| **feature_normalization**  | `str`  | `"l2"`  | Normalization applied to student and teacher features before computing the prediction loss. Options: `"l2"` (L2-normalize per spatial position), `"layer_norm"`, `"none"`. |

---

## 7b. Density prediction (auxiliary)

| Parameter                     | Type   | Default | What it changes |
|------------------------------|--------|---------|-----------------|
| **density_prediction_weight**| `float`| `0.0`   | Weight for an auxiliary MSE loss: the student’s dense feature map is passed through a small 1×1 head to predict the input image (same resolution). **Use this when dense maps are too smooth or do not align with local density/intensity.** Non-zero values (e.g. `0.1`–`0.5`) encourage the dense representation to preserve local density so features map to input values. `0` disables. |
| **decoder_image_skip**       | `bool` | `False` | If `True`, the decoder receives the input image and fuses it (resized to output size) with the decoder features before the final head. **Use when dense maps are too smooth and you want structure/density to follow the input.** The dense map then has a direct path to carry local density; combine with `density_prediction_weight` for best effect. |

---

## 8. SIGReg (anti-collapse regularization)

| Parameter                    | Type   | Default   | What it changes |
|-----------------------------|--------|-----------|-----------------|
| **sigreg_weight**           | `float`| `0.05`    | Global weight for the SIGReg loss (added to the prediction loss). Higher values push harder against flat/dead channels and trivial feature maps. |
| **sigreg_target**           | `str`  | `"dense"` | Which student features to regularize: `"dense"` (dense feature map only), `"token"` (encoder token features only), or `"both"`. |
| **flat_channel_threshold**  | `float`| `1.0e-4`  | Threshold below which a channel’s variance (or activation magnitude) is considered “flat” or “dead” in SIGReg and in diagnostics. Tuning this affects how aggressively collapse is penalized and how flat/dead channels are reported. |

---

## 9. Optimizer and schedule

| Parameter        | Type   | Default  | What it changes |
|-----------------|--------|----------|-----------------|
| **optimizer**   | `str`  | `"adamw"`| Optimizer for the student. Supported: `"adamw"`, `"sgd"` (momentum 0.9). |
| **learning_rate** | `float` | `1.0e-4` | Peak learning rate (after warmup when using cosine). |
| **weight_decay** | `float` | `0.05`   | Weight decay (L2) applied to the student parameters. |
| **batch_size**   | `int`  | `8`      | Training batch size (used by the training script’s dataloader). |
| **num_epochs**   | `int`  | `100`    | Number of training epochs. |
| **scheduler**    | `str`  | `"cosine"` | LR schedule after warmup: `"cosine"` (cosine decay to 0), `"constant"`, or `"none"` (no scheduler). |
| **warmup**       | `int`  | `10`     | Number of epochs of linear warmup from 0 to `learning_rate` before the main schedule. |

---

## 10. Checkpoints and logging

| Parameter                        | Type   | Default | What it changes |
|----------------------------------|--------|---------|-----------------|
| **checkpoint_dir**               | `str`  | `"checkpoints"` | Directory where the training script saves checkpoints and, if enabled, diagnostic artifacts. |
| **use_amp**                      | `bool` | `True`  | Whether to use automatic mixed precision (AMP) when running on CUDA. Reduces memory and can speed up training. |
| **feature_stat_logging_frequency** | `int` | `0`     | How often (in epochs) to compute and log feature health stats and save diagnostic artifacts (e.g. variance summaries, mask visualizations). `0` disables. Non-zero values (e.g. `1`, `50`) help monitor collapse and debug. |

---

## Quick reference: config → component

- **Tokenization / grid:** `image_size`, `in_channels`, `patch_size`, `stride`
- **Encoder capacity:** `embed_dim`, `depths`, `num_heads`, `window_size`, `mlp_ratio`, `dropout`, `drop_path`
- **Position:** `use_absolute_positional_embedding`, `use_relative_position_bias`
- **Decoder / output:** `decoder_channels`, `dense_feature_dim`, `decoder_image_skip`
- **Predictor:** `predictor_hidden_dim`, `predictor_depth`
- **Masking:** `mask_ratio`, `mask_block_size`, `mask_seed`
- **Teacher:** `ema_decay`
- **Loss:** `prediction_loss_type`, `feature_normalization`
- **Density alignment:** `density_prediction_weight`
- **SIGReg:** `sigreg_weight`, `sigreg_target`, `flat_channel_threshold`
- **Training:** `optimizer`, `learning_rate`, `weight_decay`, `batch_size`, `num_epochs`, `scheduler`, `warmup`
- **Infra:** `checkpoint_dir`, `use_amp`, `feature_stat_logging_frequency`

Validation and training scripts use the same config; only the entrypoint (train vs validate) and CLI args (e.g. `--checkpoint`, `--data-root`) differ.

---

## If feature maps are too smooth or not aligned with density

Try in order:

1. **`decoder_image_skip: true`** – Gives the decoder direct access to the input image so the dense map can carry structure/density. Often the biggest change.
2. **`density_prediction_weight: 0.2`–`0.5`** – Auxiliary loss so the dense map predicts the input; encourages channels to reflect local intensity.
3. **`feature_normalization: "none"`** – Keeps feature dynamic range instead of L2-normalizing; can make spatial variation more visible.
4. **Higher `mask_ratio`** (e.g. 0.65–0.7) – Harder prediction task can yield sharper, more structured features (if training stays stable).
