from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bgr.soil.training_args import TrainingArgs
    from bgr.soil.data.horizon_tabular_data import HorizonDataProcessor

class Experiment(ABC):
    
    @abstractmethod
    def __init__(self, training_args: 'TrainingArgs', target: str, dataprocessor: 'HorizonDataProcessor'):
        pass
    
    @abstractmethod
    def train_and_validate(self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            model_output_dir: str
        ) -> tuple[nn.Module, dict]:
        pass
    
    @abstractmethod
    def test(self,
        model: nn.Module,
        test_df: pd.DataFrame,
        model_output_dir: str
    ) -> dict:
        pass
    
    @abstractmethod
    def get_model(self) -> nn.Module:
        pass
    
    @abstractmethod
    def plot_losses(self, model_output_dir: str, wandb_image_logging: bool) -> None:
        pass
    
    @staticmethod
    def get_experiment_hyperparameters() -> dict:
        pass