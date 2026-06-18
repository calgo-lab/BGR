# SoilNet SAM Variant

This document describes the SAM-backed SoilNet variant introduced for the no-geotemporal end-to-end pipeline.

## Goal

The new model keeps the existing SoilNet output contract:

- depth marker predictions
- per-segment tabular predictions
- per-segment horizon embeddings

The main change is the visual path:

- replace the ResNet image backbone with a pretrained Segment Anything Model (SAM)
- replace hard segment crops and random patching with soft vertical segment masks
- keep the graph-embedding-based horizon loss used by the existing `_embed_` experiments

## Architecture

1. Unpad each image with the batch mask.
2. Resize the image to the SAM input resolution.
3. Encode the image with the pretrained SAM image encoder.
4. Pool the full feature map to predict depth markers.
5. Generate smooth vertical masks from the depth markers.
6. Pool masked segment features from the SAM feature map.
7. Optionally pass segment features through a transformer encoder for cross-segment context.
8. Predict tabular values per segment.
9. Build horizon embeddings from the concatenated segment and tabular features.

## Soft Masks

Segment boundaries are represented as continuous masks over the feature-map height.

- Each segment gets a smooth transition zone at the lower and upper boundary.
- The default overlap is 10% of the segment height.
- The masks are normalized so neighboring segments blend rather than hard-cut.

This is intended to preserve context across horizon boundaries, which is important in soil profiles where adjacent layers influence the target horizon label.

## Fine-tuning

The SAM wrapper freezes the full encoder by default and unfreezes:

- the last few transformer blocks
- the image encoder neck

This keeps the model usable as a pretrained foundation model while still allowing task-specific adaptation.

## Hyperparameters

Important defaults in the new experiment:

- `sam_model_type`: `vit_b`
- `sam_trainable_layers`: `4`
- `sam_input_size`: `1024`
- `segment_overlap_pct`: `0.10`
- `boundary_smoothing`: `cosine`
- `segment_pooling_mode`: `masked_avg`
- `segment_attention_layers`: `2`
- `segment_attention_heads`: `4`
- `bilinear_rank`: `128`

If you want to compare against bilinear pooling, set `segment_pooling_mode` to `bilinear`.
If you want to ablate cross-segment attention, set `segment_attention_layers` to `0`.

You must provide a valid SAM checkpoint path via either:

- `hyperparameters["sam_checkpoint"]`
- the `SAM_CHECKPOINT` environment variable

## Experiment Type

Register and run the new experiment with:

- `end2end_sam_embed_nogeotemps`

## Notes

- The experiment intentionally omits geotemporal inputs.
- The horizon task still uses embeddings and cosine loss, matching the existing `_embed_` experiments.
- The code path is designed to stay compatible with the current training and evaluation loops.
