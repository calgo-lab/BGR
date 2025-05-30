import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import wandb
import os

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