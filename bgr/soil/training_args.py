from argparse import Namespace
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError
from bgr.soil.callbacks import EarlyStopping, ModelCheckpoint

class TrainingArgs:
    """TODO: Docstring for TrainingArgs."""
    
    def __init__(self,
        model_output_dir: str = "model_output",
        learning_rate: int = 1e-3,
        dropout: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 4,
        num_epochs: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        optimizer_cls = torch.optim.Adam,
        lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
        torch_metric = MeanSquaredError(),
        loss_fn = nn.MSELoss(),
        wandb_logger = None,
        callbacks = None,
        save_checkpoints = True,
        use_early_stopping = True,
        early_stopping_patience = 5,
        early_stopping_min_delta = 1e-4
        ):
        
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.device = device
        self.optimizer_cls = optimizer_cls
        self.lr_scheduler_cls = lr_scheduler_cls
        self.torch_metric = torch_metric
        self.loss_fn = loss_fn
        self.wandb_logger = wandb_logger
        self.callbacks = callbacks
        self.save_checkpoints = save_checkpoints
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
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
                    save_path=model_output_dir,
                    monitor="val_loss",
                    mode="min", 
                    verbose=True
                )
            )