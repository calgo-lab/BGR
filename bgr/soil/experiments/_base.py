from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from typing import TYPE_CHECKING
import re
import numpy as np

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
        model_output_dir: str,
        wandb_image_logging: bool
    ) -> dict:
        pass
    
    @abstractmethod
    def get_model(self) -> nn.Module:
        pass
    
    @abstractmethod
    def plot_losses(self, model_output_dir: str, wandb_image_logging: bool) -> None:
        pass
    
    def _plot_confusion_matrices(self, labels, predictions, emb_dict, model_output_dir, wandb_image_logging=False, mode='test'):
        """
        Plots two confusion matrices - one for all the terminal symbols, one for the (aggregated) main symbols - as heatmaps and saves/logs them.
        
        Args:
            labels (list or array): True labels.
            predictions (list or array): Predicted labels.
            emb_dict (dict): Dictionary with label mappings.
            model_output_dir (str): Directory to save the confusion matrix image.
            wandb_image_logging (bool): Whether to log the image to Weights & Biases.
            mode (str): Mode of the confusion matrix (e.g., 'test', 'train').
            cmap (str): Colormap for the heatmap.
            annotate (bool): Whether to annotate the heatmap with cell values.
        """
        def sort_by_last_capital(label):
            # Find all capital letters in the string
            capitals = re.findall(r'[A-Z]', label)
            # Use the last capital letter if it exists, otherwise fallback to the first character
            return capitals[-1] if capitals else label[0]
        
        def plot_and_save_cm(labels, predictions, possible_labels, agg_level='terminal'):
            # Calculate accuracy score based on the current level's labels/predictions
            accuracy = accuracy_score(labels, predictions)
            
            # Compute confusion matrix
            cm = confusion_matrix(labels, predictions, labels=possible_labels, normalize='true')
            
            # Set up the figure size dynamically for large matrices
            fig_size = max(10, len(possible_labels) * 0.2)
            
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            sns.heatmap(cm, cmap='Blues',
                        xticklabels=possible_labels, yticklabels=possible_labels, 
                        cbar_kws={'shrink': 0.8}, linewidths=0.5, linecolor='lightgray')
            ax.set_aspect('equal')
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            ax.set_title(f'Confusion Matrix {mode} - Accuracy: {accuracy:.4f}', fontsize=14)
            
            # Save the figure
            os.makedirs(model_output_dir, exist_ok=True)
            cm_path = os.path.join(model_output_dir, f'confusion_matrix_{mode}_{agg_level}.pdf')
            plt.savefig(cm_path, bbox_inches='tight', format='pdf')        
            
            # Log to Weights & Biases if enabled
            if wandb_image_logging:
                wandb.log({f"Confusion matrix {mode} {agg_level}": wandb.Image(fig)})
            
            plt.close(fig)

        ### Matrix for all terminal symbols
        labels = [emb_dict['ind2label'][label] for label in labels]
        predictions = [emb_dict['ind2label'][label] for label in predictions]
        possible_labels = sorted(emb_dict['ind2label'], key=sort_by_last_capital)
        plot_and_save_cm(labels, predictions, possible_labels)
        
        ### Matrix for main symbols
        # Aggregate into main symbols (keep the last capital letter in case of mixtures)
        labels = [re.findall(r'[A-Z]', label)[-1] for label in labels]
        predictions = [re.findall(r'[A-Z]', label)[-1] for label in predictions]
        possible_labels = sorted(set(labels))
        plot_and_save_cm(labels, predictions, possible_labels, agg_level='main')
        
    
    @staticmethod
    def get_experiment_hyperparameters() -> dict:
        pass