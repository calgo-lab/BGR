from argparse import Namespace
import torch
from bgr.soil.callbacks import EarlyStopping, ModelCheckpoint
from bgr.soil.experiments import get_experiment_hyperparameters

class TrainingArgs:
    """TODO: Docstring for TrainingArgs."""
    
    def __init__(self,
        model_output_dir: str = "model_output",
        learning_rate: int = 1e-4,
        weight_decay: int = 1e-2,
        dropout: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 16,
        num_epochs: int = 20,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        callbacks = None,
        save_checkpoints = True,
        use_early_stopping = True,
        early_stopping_patience = 5,
        early_stopping_min_delta = 1e-4,
        hyperparameters: dict = {}
        ):
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.device = device
        self.callbacks = callbacks
        self.save_checkpoints = save_checkpoints
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.hyperparameters = hyperparameters
        
        if self.callbacks is None:
            self.init_default_callbacks(model_output_dir)
    
    @staticmethod
    def create_from_args(args : Namespace) -> 'TrainingArgs':
        training_args = TrainingArgs()
        
        # Set the general parameters for the model
        for var_name in training_args.__dict__.keys():
            if hasattr(args, var_name):
                new_value = getattr(args, var_name)
                
                # if the value is not None, update the attribute
                if new_value is not None:
                    setattr(training_args, var_name, new_value)
        
        if hasattr(args, "model_output_dir"):
            model_output_dir = getattr(args, "model_output_dir")
            training_args.init_default_callbacks(model_output_dir)
        
        # Set the hyperparameters for the experiment
        if hasattr(args, "experiment_type"):
            experiment_type = getattr(args, "experiment_type")
            training_args.hyperparameters = get_experiment_hyperparameters(experiment_type)
            
            for var_name in training_args.hyperparameters.keys():
                if hasattr(args, var_name):
                    new_value = getattr(args, var_name)
                    
                    # if the value is not None, update the attribute
                    if new_value is not None:
                        training_args.hyperparameters[var_name] = new_value
        
        return training_args
    
    @staticmethod
    def create_from_dict(args_dict: dict, device: str = 'cuda') -> 'TrainingArgs':
        """
        Create TrainingArgs from a dictionary (used by parallel workers).
        
        Args:
            args_dict: Dictionary of training arguments.
            device: Target device string (e.g., 'cuda:0', 'cuda:1').
        
        Returns:
            TrainingArgs: Configured training arguments.
        """
        training_args = TrainingArgs(device=device)
        
        for var_name, value in args_dict.items():
            if var_name == 'device':
                continue
            if var_name == 'hyperparameters' and isinstance(value, dict):
                training_args.hyperparameters = value
            elif var_name == 'callbacks':
                continue
            elif var_name in ['learning_rate', 'weight_decay', 'dropout', 'batch_size',
                              'num_workers', 'num_epochs', 'save_checkpoints',
                              'use_early_stopping', 'early_stopping_patience',
                              'early_stopping_min_delta']:
                if value is not None:
                    setattr(training_args, var_name, value)
        
        return training_args
    
    @staticmethod
    def create_from_args_with_device(args: Namespace, device: str) -> 'TrainingArgs':
        """
        Create TrainingArgs from args with a specific device.
        
        Args:
            args: Namespace with command line arguments.
            device: Target device string (e.g., 'cuda:0').
        
        Returns:
            TrainingArgs: Configured training arguments with specified device.
        """
        training_args = TrainingArgs.create_from_args(args)
        training_args.device = device
        return training_args
    
    def init_default_callbacks(self, model_output_dir : str) -> None:
        self.callbacks = []
        
        if self.use_early_stopping:
            self.callbacks.append(
                EarlyStopping(
                    patience=self.early_stopping_patience,
                    min_delta=self.early_stopping_min_delta,
                    verbose=True,
                    monitor="val_loss",
                    mode="min"
                )
            )
        
        if self.save_checkpoints:
            self.callbacks.append(
                ModelCheckpoint(
                    save_dir=model_output_dir,
                    monitor="val_loss",
                    mode="min", 
                    verbose=True
                )
            )