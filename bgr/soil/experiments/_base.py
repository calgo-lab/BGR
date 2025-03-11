from abc import ABC, abstractmethod
import pandas as pd
import torch.nn as nn

from bgr.soil.training_args import TrainingArgs

class Experiment(ABC):
    
    @abstractmethod
    def __init__(self, target: str):
        pass
    
    @abstractmethod
    def train_and_validate(self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            training_args: TrainingArgs,
            model_output_dir: str
        ) -> tuple[nn.Module, dict]:
        pass
    
    @abstractmethod
    def test(self,
        model: nn.Module,
        test_df: pd.DataFrame,
        training_args: TrainingArgs,
        model_output_dir: str
    ) -> dict:
        pass
    
    @abstractmethod
    def get_model(self) -> nn.Module:
        pass
    
    @abstractmethod
    def plot_losses(self, model_output_dir: str, wandb_image_logging: bool) -> None:
        pass