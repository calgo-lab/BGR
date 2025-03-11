import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import wandb
import datetime

from bgr.soil.training_args import TrainingArgs
from bgr.soil.experiments import get_experiment

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
            target: str,
            wandb_project_name : str,
            seed: int = None,
            wandb_image_logging: bool = False
        ):
        """
        Initializes the ExperimentRunner with the given parameters.
        """
        
        self.experiment_type = experiment_type
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.target = target
        self.wandb_project_name = wandb_project_name
        self.seed = seed
        self.wandb_image_logging = wandb_image_logging
        
    def run_train_val_test(
            self,
            training_args: TrainingArgs,
            model_output_dir: str,
            datetime: str,
            wandb_offline: bool = False
        ):
        try:
            # Get the experiment according to the specified type
            experiment = get_experiment(self.experiment_type, self.target)
            
            # Initialize wandb
            self._init_wandb(wandb_offline, model_output_dir, datetime)
            
            # Set the seed
            if self.seed is not None:
                self._set_seed(self.seed)
            
            # Train, validate and test the model
            model, metrics = experiment.train_and_validate(self.train_data, self.val_data, training_args, model_output_dir)
            wandb.log(metrics)
            
            # Plot the losses
            experiment.plot_losses(model_output_dir, self.wandb_image_logging)
            
            # Save the model
            self._save_model(model, model_output_dir)
            
            # Test the model
            test_metrics = experiment.test(model, self.test_data, training_args, model_output_dir)
            wandb.log(test_metrics)
            
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
    
    def _set_seed(self, seed : int) -> None:
        """
        Sets the seed for the random number generators in numpy and torch.

        Args:
            seed (int): The seed for the random number generators.
        """
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        
    def _save_model(self, model: nn.Module, model_output_dir: str) -> None:
        """
        Saves the model to the model_output_dir.
        
        Args:
            model (nn.Module): The model to save.
        """
        
        torch.save(model.state_dict(), model_output_dir)