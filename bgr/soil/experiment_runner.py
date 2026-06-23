import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import wandb
import os
import random
import matplotlib.pyplot as plt

from bgr.soil.training_args import TrainingArgs
from bgr.soil.experiments import get_experiment
from bgr.soil.data.horizon_tabular_data import HorizonDataProcessor

class ExperimentRunner:
    """
    The ExperimentRunner class is responsible for managing and executing the training, validation, and testing 
    of a machine learning model. It handles the creation of the model, the execution of the training process, 
    the evaluation on the validation set, and the final testing.
    """
    
    def __init__(
        self,
        experiment_type: str,
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame, 
        test_data: pd.DataFrame,
        dataprocessor: HorizonDataProcessor,
        target: str,
        wandb_project_name : str,
        seed: int = None,
        wandb_plot_logging: bool = False
    ):
        """
        Initializes the ExperimentRunner with the given parameters.
        """
        
        self.experiment_type = experiment_type
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.dataprocessor = dataprocessor
        self.target = target
        self.wandb_project_name = wandb_project_name
        self.seed = seed
        self.wandb_plot_logging = wandb_plot_logging
    
    def run_inference(
        self,
        training_args: TrainingArgs,
        model_file_path: str,
        model_output_dir: str,
        timestamp: str,
        wandb_offline: bool = False
    ):
        """
        Runs inference on the test data using a pre-trained model.

        Args:
            training_args (TrainingArgs): The training arguments.
            model_file_path (str): The path to the pre-trained model file.
            model_output_dir (str): The directory for experiment output.
            datetime (str): The timestamp for the experiment.
            wandb_offline (bool): If True, wandb will be initialized in offline mode.

        Returns:
            dict: The test metrics.
        """
        try:
            # Get the experiment according to the specified type
            experiment = get_experiment(self.experiment_type, training_args, self.target, self.dataprocessor)
            
            # Initialize wandb
            self._init_wandb(wandb_offline, model_output_dir, timestamp)
            
            # Load the model
            model = experiment.get_model()
            self._load_model(model_file_path, model)
            
            # Test the model
            test_metrics = experiment.test(model, self.test_data, model_output_dir, self.wandb_plot_logging)
            wandb.log(test_metrics)
            
            return test_metrics
        finally:
            if wandb.run is not None:
                wandb.run.finish()
            torch.cuda.empty_cache()
    
    def run_train_val_test(
        self,
        training_args: TrainingArgs,
        model_output_dir: str,
        timestamp: str,
        wandb_offline: bool = False
    ):
        """
        Runs the training, validation, and testing of the model.

        Args:
            training_args (TrainingArgs): The training arguments.
            model_output_dir (str): The directory to save the trained model.
            datetime (str): The timestamp for the experiment.
            wandb_offline (bool): If True, wandb will be initialized in offline mode.

        Returns:
            dict: The combined metrics from training, validation, and testing.
        """
        try:
            # Get the experiment according to the specified type
            experiment = get_experiment(self.experiment_type, training_args, self.target, self.dataprocessor)
            
            # Initialize wandb
            self._init_wandb(wandb_offline, model_output_dir, timestamp)
            wandb.config.update(training_args.__dict__)
            
            # Set the seed
            if self.seed is not None:
                self._set_seed(self.seed)
            
            # Train, validate and test the model
            model, metrics = experiment.train_and_validate(self.train_data, self.val_data, model_output_dir)
            
            # Save the model
            self._save_model(model, model_output_dir)
            
            # Test the model
            test_metrics = experiment.test(model, self.test_data, model_output_dir, self.wandb_plot_logging)
            wandb.log(test_metrics)
            
            # Plot the losses
            experiment.plot_losses(model_output_dir, self.wandb_plot_logging)
            
            metrics.update(test_metrics)
            
            return metrics
        finally:
            if wandb.run is not None:
                wandb.run.finish()
            torch.cuda.empty_cache()
    
    def _init_wandb(self, wandb_offline: bool, model_output_dir: str, timestamp: str) -> None:
        """
        Initializes the wandb for the experiment.
        
        Args:
            wandb_offline (bool): If True, wandb will be initialized in offline mode.
        """
        
        wandb.init(project=self.wandb_project_name, dir=model_output_dir, name=f"{self.experiment_type}_{timestamp}", mode = 'offline' if wandb_offline else 'online')
            
        wandb.config.update({
            "experiment_type": self.experiment_type,
            "seed": self.seed
        })
    
    def _load_model(self, model_file_path: str, model: nn.Module) -> nn.Module:
        """
        Loads the model from the model_file_path and returns the model.
        
        Args:
            model_file_path (str): The path to the model file.
            model (nn.Module): The model to load the state_dict into.
        
        Returns:
            nn.Module: The model with the loaded state_dict.
        """
        
        model.load_state_dict(torch.load(model_file_path))
        return
        
    def _save_model(self, model: nn.Module, model_output_dir: str) -> None:
        """
        Saves the model to the model_output_dir.
        
        Args:
            model (nn.Module): The model to save.
            model_output_dir (str): The directory to save the model.
        """
        
        torch.save(model.state_dict(), os.path.join(model_output_dir, "model.pt"))

    def run_with_seed(
        self,
        training_args: TrainingArgs,
        model_output_base: str,
        timestamp: str,
        seed: int,
        wandb_offline: bool = False
    ) -> dict:
        """
        Run single experiment with specific seed. Used by seed ensemble.

        Uses the same data splits (train/val/test) as the parent experiment,
        only the model random seed changes for variance estimation.

        Args:
            training_args: Training arguments.
            model_output_base: Base output directory for ensemble.
            timestamp: Timestamp string.
            seed: Random seed for this run.
            wandb_offline: If True, wandb will be initialized in offline mode.

        Returns:
            dict: Results including metrics and history for this seed.
        """
        seed_dir = os.path.join(model_output_base, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        self._set_seed(seed)

        exp_training_args = TrainingArgs.create_from_args(training_args)
        exp_training_args.init_default_callbacks(seed_dir)

        experiment = get_experiment(self.experiment_type, exp_training_args, self.target, self.dataprocessor)

        group_name = f"{self.experiment_type}_ensemble_{timestamp}"
        self._init_wandb_ensemble(wandb_offline, seed_dir, group_name, seed, timestamp)
        wandb.config.update(training_args.__dict__)

        try:
            model, epoch_metrics = experiment.train_and_validate(self.train_data, self.val_data, seed_dir)

            self._save_model(model, seed_dir)

            test_metrics = experiment.test(model, self.test_data, seed_dir, self.wandb_plot_logging)
            if wandb.run is not None:
                wandb.log(test_metrics)

            experiment.plot_losses(seed_dir, self.wandb_plot_logging)

            result = {
                'seed': seed,
                'final_epoch_metrics': {**epoch_metrics, **test_metrics},
                'history': list(experiment.histories) if hasattr(experiment, 'histories') and experiment.histories else [],
            }

            return result

        finally:
            if wandb.run is not None:
                wandb.run.finish()
            torch.cuda.empty_cache()

    def _init_wandb_ensemble(
        self,
        wandb_offline: bool,
        seed_dir: str,
        group_name: str,
        seed: int,
        timestamp: str
    ) -> None:
        """
        Initialize wandb with grouping for ensemble runs.

        Args:
            wandb_offline: If True, wandb will be initialized in offline mode.
            seed_dir: Output directory for this seed's run.
            group_name: Group name for ensemble runs.
            seed: Random seed for this run.
            timestamp: Timestamp string.
        """
        wandb.init(
            project=self.wandb_project_name,
            dir=seed_dir,
            name=f"{self.experiment_type}_seed{seed}_{timestamp}",
            group=group_name,
            mode='offline' if wandb_offline else 'online'
        )
        wandb.config.update({
            "experiment_type": self.experiment_type,
            "seed": seed,
            "group": group_name
        })

    def _aggregate_metrics(self, all_run_results: list) -> dict:
        """
        Compute mean and std across all seeds for each metric.

        Args:
            all_run_results: List of result dicts from run_with_seed.

        Returns:
            dict: Aggregated metrics with _mean and _std suffixes.
        """
        if not all_run_results:
            return {}

        metric_keys = set()
        for r in all_run_results:
            metric_keys.update(r['final_epoch_metrics'].keys())

        aggregated = {}
        for key in metric_keys:
            values = []
            for r in all_run_results:
                val = r['final_epoch_metrics'].get(key)
                if val is not None and isinstance(val, (int, float, np.integer, np.floating)):
                    values.append(float(val))

            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_n'] = len(values)

        histories = [r['history'] for r in all_run_results if r.get('history')]
        if histories and any(h for h in histories):
            aggregated['history_mean'] = self._aggregate_histories(histories)

        return aggregated

    def _aggregate_histories(self, all_histories: list) -> list:
        """
        Aggregate epoch histories with mean +/- std per epoch.

        Args:
            all_histories: List of per-epoch metric dicts.

        Returns:
            list: List of aggregated per-epoch dicts.
        """
        valid_histories = [h for h in all_histories if h and len(h) > 0]
        if not valid_histories:
            return []

        num_epochs = min(len(h) for h in valid_histories)
        if num_epochs == 0:
            return []

        metric_keys = set()
        for h in valid_histories:
            for epoch_data in h[:num_epochs]:
                metric_keys.update(epoch_data.keys())

        aggregated_history = []
        for epoch_idx in range(num_epochs):
            epoch_agg = {'epoch': epoch_idx + 1}

            for key in metric_keys:
                if key == 'epoch':
                    continue
                values = []
                for h in valid_histories:
                    val = h[epoch_idx].get(key)
                    if val is not None and isinstance(val, (int, float, np.integer, np.floating)):
                        values.append(float(val))

                if values:
                    epoch_agg[f'{key}_mean'] = np.mean(values)
                    epoch_agg[f'{key}_std'] = np.std(values)

            aggregated_history.append(epoch_agg)

        return aggregated_history

    def _log_ensemble_summary_to_wandb(self, aggregated: dict, group_name: str) -> None:
        """
        Log ensemble summary to wandb with a summary figure.

        Args:
            aggregated: Aggregated metrics dict.
            group_name: Name for the plot title.
        """
        try:
            if wandb.run is None:
                return

            from bgr.soil.seed_ensemble_summary import create_ensemble_summary_figure

            available_metrics = _get_available_summary_metrics(aggregated)
            n_seeds = len(available_metrics)

            if hasattr(self, 'experiment_type'):
                exp_type = self.experiment_type
            else:
                exp_type = 'Experiment'

            fig = create_ensemble_summary_figure(
                aggregated,
                exp_type,
                n_seeds=n_seeds
            )
            wandb.log({'ensemble_summary_plot': wandb.Image(fig)})
            plt.close(fig)

            # Use dynamic metric detection
            for key in available_metrics:
                mean_key = f'{key}_mean'
                std_key = f'{key}_std'
                n_key = f'{key}_n'
                if mean_key in aggregated:
                    wandb.summary[f'ensemble_{key}'] = {
                        'mean': aggregated.get(mean_key),
                        'std': aggregated.get(std_key),
                        'n': aggregated.get(n_key, 0)
                    }
        except Exception:
            pass

    def _set_seed(self, seed: int) -> None:
        """
        Sets the seed for all random number generators for full reproducibility.

        Args:
            seed: The seed for all random number generators.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MinimalDataProcessor:
    """
    Lightweight dataprocessor for multiprocessing workers.
    Only holds the state needed for training - no file I/O.
    """

    def __init__(self, state: dict):
        self.embeddings_dict = state['embeddings_dict']
        self.category_maps = state['category_maps']
        self.tabulars_output_dim_dict = state['tabulars_output_dim_dict']
        self.geotemp_img_infos = state.get('geotemp_img_infos', [])


def _run_seed_worker(cfg: dict) -> dict:
    """
    Worker function for a single seed. Module-level for multiprocessing pickling.

    Args:
        cfg: Dictionary containing all configuration for this worker:
            - seed: Random seed
            - gpu_id: GPU device ID
            - train_df, val_df, test_df: DataFrames
            - dataprocessor_state: Minimal state dict
            - experiment_type, training_args_dict, target, timestamp
            - model_output_base, wandb_offline, ensemble_group

    Returns:
        dict: Result with 'seed', 'status', 'metrics'/'error'
    """
    import gc
    import traceback

    try:
        gpu_id = cfg['gpu_id']
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"GPU {gpu_id} not available. Only {torch.cuda.device_count()} GPUs found. "
                f"Valid GPU IDs: {list(range(torch.cuda.device_count()))}"
            )
        torch.cuda.set_device(gpu_id)

        training_args = TrainingArgs.create_from_dict(
            cfg['training_args_dict'],
            device=f'cuda:{gpu_id}'
        )
        training_args.callbacks = None

        dataprocessor = MinimalDataProcessor(cfg['dataprocessor_state'])

        experimenter = ExperimentRunner(
            experiment_type=cfg['experiment_type'],
            train_data=cfg['train_df'],
            val_data=cfg['val_df'],
            test_data=cfg['test_df'],
            dataprocessor=dataprocessor,
            target=cfg['target'],
            wandb_project_name=cfg.get('wandb_project_name', 'BGR'),
            seed=cfg['seed'],
            wandb_plot_logging=cfg['training_args_dict'].get('wandb_plot_logging', False),
        )

        seed_dir = os.path.join(cfg['model_output_base'], f"seed_{cfg['seed']}")
        os.makedirs(seed_dir, exist_ok=True)

        _init_wandb_for_worker(
            cfg['wandb_offline'],
            seed_dir,
            cfg['experiment_type'],
            cfg['seed'],
            cfg['timestamp'],
            cfg['ensemble_group'],
            cfg.get('wandb_project_name', 'BGR'),
        )

        try:
            experimenter._set_seed(cfg['seed'])

            exp_training_args = TrainingArgs.create_from_dict(
                cfg['training_args_dict'],
                device=f'cuda:{gpu_id}'
            )
            exp_training_args.init_default_callbacks(seed_dir)

            experiment = get_experiment(
                cfg['experiment_type'],
                exp_training_args,
                cfg['target'],
                dataprocessor
            )

            model, epoch_metrics = experiment.train_and_validate(
                cfg['train_df'],
                cfg['val_df'],
                seed_dir
            )

            experimenter._save_model(model, seed_dir)

            test_metrics = experiment.test(
                model,
                cfg['test_df'],
                seed_dir,
                cfg['training_args_dict'].get('wandb_plot_logging', False)
            )

            if wandb.run is not None:
                wandb.log(test_metrics)

            experiment.plot_losses(seed_dir, cfg['training_args_dict'].get('wandb_plot_logging', False))

            return {
                'seed': cfg['seed'],
                'status': 'success',
                'metrics': {**epoch_metrics, **test_metrics},
                'history': list(experiment.histories) if hasattr(experiment, 'histories') and experiment.histories else [],
            }

        finally:
            if wandb.run is not None:
                wandb.run.finish()
            del model
            gc.collect()
            torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return {'seed': cfg['seed'], 'status': 'skipped', 'error': 'OOM'}

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        return {
            'seed': cfg['seed'],
            'status': 'error',
            'error': f'{type(e).__name__}: {e}',
            'traceback': traceback.format_exc(),
        }


def _init_wandb_for_worker(
    wandb_offline: bool,
    seed_dir: str,
    experiment_type: str,
    seed: int,
    timestamp: str,
    group_name: str,
    wandb_project_name: str
) -> None:
    """Initialize wandb in worker process."""
    wandb.init(
        project=wandb_project_name,
        dir=seed_dir,
        name=f"{experiment_type}_seed{seed}_{timestamp}",
        group=group_name,
        mode='offline' if wandb_offline else 'online'
    )
    wandb.config.update({
        "experiment_type": experiment_type,
        "seed": seed,
        "group": group_name
    })


def _get_available_summary_metrics(aggregated: dict) -> list:
    """
    Dynamically detect available metrics from aggregated dict.
    
    Filters for metrics that have both _mean and _std values.
    Prioritizes common metrics (accuracy, IoU, loss) but includes any metric.
    
    Args:
        aggregated: Aggregated metrics dict with _mean and _std values
        
    Returns:
        List of metric base names that have both mean and std
    """
    metric_names = set()
    for key in aggregated.keys():
        if key.endswith('_mean'):
            base = key[:-5]  # Remove '_mean'
            std_key = f'{base}_std'
            if std_key in aggregated:
                metric_names.add(base)
    
    # Priority order for display
    priority_patterns = ['accuracy', 'iou', 'f1', 'loss', 'precision', 'recall']
    priority_metrics = []
    other_metrics = []
    
    for m in sorted(metric_names):
        if any(p in m.lower() for p in priority_patterns):
            priority_metrics.append(m)
        else:
            other_metrics.append(m)
    
    return priority_metrics + sorted(other_metrics)


def create_summary_run(
    aggregated: dict,
    experiment_type: str,
    n_seeds: int,
    summary_group: str,
    output_dir: str,
    wandb_offline: bool,
    wandb_project_name: str
) -> None:
    """
    Create wandb summary run with ensemble statistics.

    Args:
        aggregated: Aggregated metrics dict with _mean and _std values
        experiment_type: Name of the experiment
        n_seeds: Number of successful seeds
        summary_group: Group name for the summary run
        output_dir: Directory to save local wandb files
        wandb_offline: If True, wandb will be in offline mode
        wandb_project_name: WandB project name
    """
    wandb.init(
        project=wandb_project_name,
        name=f"{experiment_type}_ensemble_summary",
        group=summary_group,
        mode='offline' if wandb_offline else 'online'
    )

    wandb.config.update({'n_seeds': n_seeds, 'experiment_type': experiment_type})

    # Dynamically detect available metrics
    available_metrics = _get_available_summary_metrics(aggregated)

    summary_data = {}
    for key in available_metrics:
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        if mean_key in aggregated:
            summary_data[f'ensemble_{key}_mean'] = aggregated[mean_key]
            summary_data[f'ensemble_{key}_std'] = aggregated[std_key]

    if summary_data:
        wandb.summary.update(summary_data)

    fig, ax = plt.subplots(figsize=(10, max(6, len(available_metrics) * 0.8)))

    valid_metrics = []
    for key in available_metrics:
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        if mean_key in aggregated:
            display_name = key.replace('test_', 'Test ').replace('val_', 'Val ').replace('_', ' ').title()
            valid_metrics.append((
                display_name,
                aggregated[mean_key],
                aggregated.get(std_key, 0)
            ))

    if valid_metrics:
        labels = [m[0] for m in valid_metrics]
        means = [m[1] for m in valid_metrics]
        stds = [m[2] for m in valid_metrics]

        ax.barh(labels, means, xerr=stds, capsize=5, color='steelblue', alpha=0.7)
        ax.set_xlabel('Value')
        ax.set_title(f'{experiment_type} Ensemble (n={n_seeds})')
        ax.grid(axis='x', alpha=0.3)

        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(mean + std + 0.01, i, f'{mean:.3f} ± {std:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    wandb.log({'ensemble_summary': wandb.Image(fig)})

    wandb.finish()
    plt.close(fig)