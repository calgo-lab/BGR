#!/bin/sh

#python BGR/main.py --batch_size=8 --num_epochs=1 --train_val_test_frac=[0.1,0.1,0.8] --wandb_online --wandb_plot_logging --wandb_project_name=BGR_End2EndExperiments --experiment_type=end2end_lstm_embed

python BGR/main.py --batch_size=8 --num_epochs=1 --wandb_online --wandb_plot_logging --wandb_project_name=BGR_End2EndExperiments --experiment_type=end2end_lstm_embed