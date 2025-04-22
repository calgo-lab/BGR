#!/bin/sh
python BGR/main.py --batch_size=4 --num_epochs=100 --wandb_online --wandb_plot_logging --wandb_project_name=BGR_SimpleHorClsBaselineExperiments --experiment_type=simple_horizon_classification_lstm_geotmp_mlp_tab_mlp_resnet

python BGR/main.py --batch_size=4 --num_epochs=100 --wandb_online --wandb_plot_logging --wandb_project_name=BGR_SimpleHorClsBaselineExperiments --experiment_type=simple_horizon_classification_lstm_embed_geotmp_mlp_tab_mlp_resnet

python BGR/main.py --batch_size=2 --num_epochs=100 --wandb_online --wandb_plot_logging --wandb_project_name=BGR_SimpleTabularsBaselineExperiments --experiment_type=simple_tabulars_geotmp