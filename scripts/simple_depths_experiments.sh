#!/bin/sh
python BGR/main.py --batch_size=16 --num_epochs=100 --wandb_online --wandb_plot_logging --wandb_project_name=BGR_SimpleDepthsBaselineExperiments --experiment_type=simple_depths_geotmp

python BGR/main.py --batch_size=16 --num_epochs=100 --wandb_online --wandb_plot_logging --wandb_project_name=BGR_SimpleDepthsBaselineExperiments --experiment_type=simple_depths_geotmp_resnet