#!/bin/sh

# Softcropping ablation with seed ensemble for error bars
# Tests ResNet18, ResNet50, and DINOv2 variants

# Base configuration
BATCH_SIZE=4
NUM_EPOCHS=100
N_SEEDS=5
WANDB_PROJECT=BGR_SoftcropAblation

# Baseline comparison: Random patches with seed ensemble
echo "Running: Baseline - Random Patches (${N_SEEDS} seeds)"
python BGR/main.py \
    --batch_size=$BATCH_SIZE \
    --num_epochs=$NUM_EPOCHS \
    --wandb_project_name=$WANDB_PROJECT \
    --wandb_online \
    --wandb_plot_logging \
    --experiment_type=end2end_lstm_resnet_embed_nogeotemps \
    --seed_ensemble_size=$N_SEEDS \
    --seed_start=2025 \
    --teacher_forcing_approach=linear \
    --teacher_forcing_stop_epoch=5

# ResNet18 with seed ensemble
echo "Running: Softcrop ResNet18 (${N_SEEDS} seeds)"
python BGR/main.py \
    --batch_size=$BATCH_SIZE \
    --num_epochs=$NUM_EPOCHS \
    --wandb_project_name=$WANDB_PROJECT \
    --wandb_online \
    --experiment_type=end2end_softcrop_nogeotemps \
    --segment_encoder_backbone=resnet18 \
    --seed_ensemble_size=$N_SEEDS \
    --seed_start=2025 \
    --teacher_forcing_approach=linear \
    --teacher_forcing_stop_epoch=5

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

# DINOv2-Small with seed ensemble
echo "Running: Softcrop DINOv2-S14 (${N_SEEDS} seeds)"
python BGR/main.py \
    --batch_size=$BATCH_SIZE \
    --num_epochs=$NUM_EPOCHS \
    --wandb_project_name=$WANDB_PROJECT \
    --wandb_online \
    --wandb_plot_logging \
    --experiment_type=end2end_softcrop_nogeotemps \
    --segment_encoder_backbone=dinov2_vits14 \
    --seed_ensemble_size=$N_SEEDS \
    --seed_start=2025 \
    --teacher_forcing_approach=linear \
    --teacher_forcing_stop_epoch=5

# DINOv2-Base with seed ensemble
echo "Running: Softcrop DINOv2-B14 (${N_SEEDS} seeds)"
python BGR/main.py \
    --batch_size=$BATCH_SIZE \
    --num_epochs=$NUM_EPOCHS \
    --wandb_project_name=$WANDB_PROJECT \
    --wandb_online \
    --wandb_plot_logging \
    --experiment_type=end2end_softcrop_nogeotemps \
    --segment_encoder_backbone=dinov2_vitb14 \
    --seed_ensemble_size=$N_SEEDS \
    --seed_start=2025 \
    --teacher_forcing_approach=linear \
    --teacher_forcing_stop_epoch=5

# # DINOv2-Large with seed ensemble
echo "Running: Softcrop DINOv2-L14 (${N_SEEDS} seeds)"
python BGR/main.py \
    --batch_size=$BATCH_SIZE \
    --num_epochs=$NUM_EPOCHS \
    --wandb_project_name=$WANDB_PROJECT \
    --wandb_online \
    --wandb_plot_logging \
    --experiment_type=end2end_softcrop_nogeotemps \
    --segment_encoder_backbone=dinov2_vitl14 \
    --seed_ensemble_size=$N_SEEDS \
    --seed_start=2025 \
    --teacher_forcing_approach=linear \
    --teacher_forcing_stop_epoch=5

echo "All ensemble experiments complete!"