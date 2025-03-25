# Run the experiments
python BGR/main.py --batch_size=16 --num_epochs=100 --wandb_online --wandb_plot_logging --wandb_project_name=BGR_SimpleHorClsBaselineExperiments --experiment_type=simple_horizon_classification

python BGR/main.py --batch_size=16 --num_epochs=100 --wandb_online --wandb_plot_logging --wandb_project_name=BGR_SimpleHorClsBaselineExperiments --experiment_type=simple_horizon_classification_embed

python BGR/main.py --batch_size=16 --num_epochs=100 --wandb_online --wandb_plot_logging --wandb_project_name=BGR_SimpleHorClsBaselineExperiments --experiment_type=simple_horizon_classification_embed_geotmp

python BGR/main.py --batch_size=16 --num_epochs=100 --wandb_online --wandb_plot_logging --wandb_project_name=BGR_SimpleHorClsBaselineExperiments --experiment_type=simple_horizon_classification_embed_geotmp_mlp

python BGR/main.py --batch_size=16 --num_epochs=100 --wandb_online --wandb_plot_logging --wandb_project_name=BGR_SimpleHorClsBaselineExperiments --experiment_type=simple_horizon_classification_embed_geotmp_mlp_tab_mlp

python BGR/main.py --batch_size=16 --num_epochs=100 --wandb_online --wandb_plot_logging --wandb_project_name=BGR_SimpleHorClsBaselineExperiments --experiment_type=simple_horizon_classification_lstm_embed_geotmp_mlp_tab_mlp