#!/bin/sh

# Softcropping ablation with seed ensemble for error bars
# Tests ResNet18, ResNet50, and DINOv2 variants

# Base configuration
BATCH_SIZE=16
NUM_EPOCHS=20
N_SEEDS=5
WANDB_PROJECT=BGR_SoftcropAblation

# ResNet18 with seed ensemble
echo "Running: Softcrop ResNet18 (${N_SEEDS} seeds)"
python BGR/main.py \
    --batch_size=$BATCH_SIZE \
    --num_epochs=$NUM_EPOCHS \
    --wandb_project_name=$WANDB_PROJECT \
    --experiment_type=end2end_softcrop_nogeotemps \
    --segment_encoder_backbone=resnet18 \
    --seed_ensemble_size=$N_SEEDS \
    --seed_start=2025

# ResNet50 with seed ensemble
echo "Running: Softcrop ResNet50 (${N_SEEDS} seeds)"
python BGR/main.py \
    --batch_size=$BATCH_SIZE \
    --num_epochs=$NUM_EPOCHS \
    --wandb_project_name=$WANDB_PROJECT \
    --experiment_type=end2end_softcrop_nogeotemps \
    --segment_encoder_backbone=resnet50 \
    --seed_ensemble_size=$N_SEEDS \
    --seed_start=2025

# DINOv2-Base with seed ensemble
echo "Running: Softcrop DINOv2-B14 (${N_SEEDS} seeds)"
python BGR/main.py \
    --batch_size=$BATCH_SIZE \
    --num_epochs=$NUM_EPOCHS \
    --wandb_offline \
    --wandb_project_name=$WANDB_PROJECT \
    --experiment_type=end2end_softcrop_nogeotemps \
    --segment_encoder_backbone=dinov2_vitb14 \
    --seed_ensemble_size=$N_SEEDS \
    --seed_start=2025

# Baseline comparison: Random patches with seed ensemble
echo "Running: Baseline - Random Patches (${N_SEEDS} seeds)"
python BGR/main.py \
    --batch_size=$BATCH_SIZE \
    --num_epochs=$NUM_EPOCHS \
    --wandb_project_name=$WANDB_PROJECT \
    --experiment_type=end2end_lstm_resnet_embed_nogeotemps \
    --seed_ensemble_size=$N_SEEDS \
    --seed_start=2025

echo "All ensemble experiments complete!"