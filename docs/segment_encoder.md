# SoftCroppedSegmentEncoder

## Architecture

```
image (B, 3, H, W) + soft_mask (B, H, W)
        │
        ├──► [If feed_mask_to_backbone=True] concat → (B, 4, H, W)
        │    └──► backbone → feature_maps (B, D, H', W')
        │
        ├──► [If feed_mask_to_backbone=False] image only → (B, 3, H, W)
        │    └──► backbone → feature_maps (B, D, H', W')
        │
        ▼
feature_maps (B, D, H', W') + soft_mask (B, H, W)
        │
        ├──► Interpolate soft_mask to (H', W')
        ├──► Flatten → (B, N, D) and (B, N, 1)
        ├──► Weight features by mask, normalise by mask sum
        ├──► Project to bilinear_dim → (B, N, bilinear_dim)
        ├──► Bilinear pool F^T F → (B, bilinear_dim, bilinear_dim)
        ├──► Signed sqrt + L2 normalise
        └──► Flatten → (B, bilinear_dim²) → (B, output_dim)
```

## Backbone Options

| Name | embed_dim | Patch Size | Feature Spacing | Pretrained On | Notes |
|------|-----------|-----------|-----------------|---------------|-------|
| `resnet18` | 512 | pixel | 32 (layer4) | ImageNet-1K | CNN baseline |
| `resnet50` | 2048 | pixel | 32 (layer4) | ImageNet-1K | Higher capacity |
| `dinov2_vits14` | 384 | 14px | 14px | DINOv2 (142M) | Self-supervised, small |
| `dinov2_vitb14` | 768 | 14px | 14px | DINOv2 (142M) | **Recommended — best value** |
| `dinov2_vitl14` | 1024 | 14px | 14px | DINOv2 (142M) | Largest ViT variant |

## Configuration Dimensions

### 1. `feed_mask_to_backbone` (default: `True`)

Controls how the soft mask interacts with the backbone.

| Value | Backbone Input | Mask Effect on Backbone |
|-------|---------------|------------------------|
| `True` (default) | 4 channels (RGB + mask) | Backbone processes mask as modality during feature extraction |
| `False` | 3 channels (RGB only) | Backbone processes clean RGB; mask only affects bilinear pooling |

- **`True`** is needed if `train_backbone=True` (fine-tuning) so the backbone can learn mask-aware representations.
- **`False`** uses standard 3-channel pretrained weights with no 4th-channel expansion — the backbone is never exposed to the mask during feature extraction. The mask is only applied as spatial weights in the bilinear pool (Option C). This is the clean ablation baseline.
- With `train_backbone=False` (frozen backbone) and `feed_mask_to_backbone=True`, the 4th channel contributes only a static additive bias (the 4th conv weight is tiny). The real benefit of Option A only materialises with `train_backbone=True`.

### 2. `train_backbone` (default: `False`)

Controls whether backbone weights are trainable.

| Value | Backbone Weights | Effect |
|-------|-----------------|--------|
| `False` (default) | Frozen (`requires_grad=False`) | Pretrained features preserved; only `proj_bilinear` and `projection` train |
| `True` | Trainable (`requires_grad=True`) | Full fine-tuning of backbone + head layers |

- **`False`**: Use when the pretrained features are already good (standard transfer learning).
- **`True`**: Enables the backbone to learn from the 4th mask channel. Recommended to use a lower learning rate for backbone parameters than for head parameters (e.g., 100× smaller).

### 3. `bilinear_dim` (default: `128`)

Projects backbone features from `embed_dim` down to `bilinear_dim` before computing the covariance matrix.

- Output size is always `bilinear_dim²` (16384 for default 128).
- Larger values retain more feature diversity before the outer product but increase computation.
- `128` is a good middle ground; scale up for larger backbones (DinoV2-B: 768-dim) if compute allows.

### 4. ResNet Feature Layer

Only applicable for `resnet18` and `resnet50` backbones.

Currently hard-coded to `layer4` (stride 32 → feature spacing of 32 pixels). Can optionally use `layer3` (stride 16) for higher spatial resolution at the cost of less semantic depth.

## Ablation Experiments

### Experiment Protocol

Run experiments with the following combinations and compare on depth prediction MAE and horizon classification accuracy.

| # | Backbone | feed_mask | train_backbone | bilinear_dim | Expected Notes |
|---|----------|-----------|----------------|--------------|----------------|
| 1 | resnet18 | True | False | 64 | Current baseline |
| 2 | resnet18 | False | False | 64 | Ablation: mask in backbone vs not |
| 3 | resnet18 | True | True | 64 | Fine-tune with 4-channel input |
| 4 | dinov2_vits14 | True | False | 128 | DinoV2-S baseline |
| 5 | dinov2_vitb14 | True | False | 256 | **Recommended start point** |
| 6 | dinov2_vitb14 | False | False | 256 | DinoV2-B ablation: mask in backbone vs not |
| 7 | dinov2_vitb14 | True | True | 256 | Fine-tune DinoV2-B |
| 8 | dinov2_vitl14 | True | False | 384 | Large model reference |

### What to Measure

- **Depth prediction**: MAE on depth marker predictions
- **Horizon classification**: Accuracy on horizon type predictions
- **Per-segment features**: t-SNE/PCA of segment embeddings to check class separation
- **Feature map statistics**: Mean/spread of feature values in masked vs non-masked regions
- **Training time**: Seconds per epoch for each configuration

## Notes

- DinoV2 variants pad inputs internally to the next multiple of `patch_size` (14px) to avoid reshape errors. Padded regions have zero mask values and black pixels, contributing minimally to bilinear pooling.
- The bilinear pooling normalisation (`mask_sum`) correctly handles variable segment sizes by normalising before the outer product.
- The `_masked_bilinear_pool` method handles all-zero segment masks self-containedly: it returns zero features for segments where the mask is entirely empty, preventing non-zero noise from leaking into the representation (the bilinear pool would otherwise compute `bias @ bias.T` on all-zero input).
- Empty segment positions in a batch are handled per-sample — one sample can have a valid mask while another has an empty mask at the same segment position, and they are encoded independently. If all samples in a batch have an empty mask at a position, all receive zero features.