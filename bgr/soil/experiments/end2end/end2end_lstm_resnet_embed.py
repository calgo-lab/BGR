from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tabulate import tabulate
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bgr.soil.training_args import TrainingArgs

from bgr.soil.utils import pad_tensor
from bgr.soil.modelling.soilnet import SoilNet_LSTM
from bgr.soil.data.horizon_tabular_data import HorizonDataProcessor
from bgr.soil.experiments._base import Experiment
from bgr.soil.modelling.tabulars.tabular_models import SimpleTabularModel
from bgr.soil.metrics import DepthMarkerLoss, TopKHorizonAccuracy, depth_iou, top_k_accuracy_from_indices, precision_recall_at_k
from bgr.soil.data.datasets import ImageTabularEnd2EndDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class End2EndLSTMResNetEmbed(Experiment):
    def __init__(self, training_args: 'TrainingArgs', target: str, dataprocessor: HorizonDataProcessor):
        self.training_args = training_args
        self.dataprocessor = dataprocessor
        self.target = target
        self.trained = False
        
        self.image_normalization = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet statistics
        ])
        
        # Tabular soil features (numerical and categorical)
        self.segments_tabular_num_feature_columns = ['Steine']
        self.segments_tabular_categ_feature_columns = {key : value 
            for key, value in dataprocessor.tabulars_output_dim_dict.items() 
                if key in [
                        'Bodenart',
                        'Bodenfarbe',
                        'Karbonat',
                        'Humusgehaltsklasse',
                        'Durchwurzelung'
                    ]
        }
        self.tabulars_output_dim_dict = dataprocessor.tabulars_output_dim_dict
        
        # Losses and metrics
        # From depths
        self.depth_loss = DepthMarkerLoss(lambda_mono=0, lambda_div=0)
        # From tabs
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.tab_topk = 3
        self.tab_class_average = 'macro'
        # From horizons
        self.label_embeddings_tensor = torch.tensor(self.dataprocessor.embeddings_dict['embedding'], device=self.training_args.device).float()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.hor_topk = 5
        self.hor_topk_acc = lambda k : TopKHorizonAccuracy(self.label_embeddings_tensor, k=k)
        self.hor_class_average = 'macro'
        
        # Retrieve the experiment hyperparameters
        self.hyperparameters = End2EndLSTMResNetEmbed.get_experiment_hyperparameters()
        self.hyperparameters.update(training_args.hyperparameters)
        
        # Initialize dictionary to store lists of stones predictions and true values (for the bisector)
        self.stones_predictions = { "train" : [], "val" : [], "test" : [] }
        self.stones_true_values = { "train" : [], "val" : [], "test" : [] }
        
        # Initialize the labels and predictions dictionaries (for confusion matrix)
        self.hor_possible_labels = list(range(self.label_embeddings_tensor.size(0)))
        self.hor_labels = {'train': None, 'val': None, 'test': None}
        self.hor_predictions = {'train': None, 'val': None, 'test': None}
    
    def train_and_validate(self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_output_dir: str # do we need this?
    ) -> tuple[nn.Module, dict]:
        # Load training and validation sets
        train_dataset = ImageTabularEnd2EndDataset(
            dataframe=train_df,
            normalize=self.image_normalization,
            label_column=self.target,
            geotemp_columns=self.dataprocessor.geotemp_img_infos[:-1], # without img path
            tab_num_columns=self.segments_tabular_num_feature_columns,
            tab_categ_columns=self.segments_tabular_categ_feature_columns
        )
        train_loader = train_dataset.to_dataloader(batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)

        val_dataset = ImageTabularEnd2EndDataset(
            dataframe=val_df,
            normalize=self.image_normalization,
            label_column=self.target,
            geotemp_columns=self.dataprocessor.geotemp_img_infos[:-1], # without img path
            tab_num_columns=self.segments_tabular_num_feature_columns,
            tab_categ_columns=self.segments_tabular_categ_feature_columns
        )
        val_loader = val_dataset.to_dataloader(batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        model = self.get_model()
        model.to(self.training_args.device)
        
        # Training parameters
        lr = self.training_args.learning_rate
        weight_decay = self.training_args.weight_decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.1, min_lr=lr*0.01, verbose=True)
        
        # Training and evaluation loop
        self.histories = []
        for epoch in range(1, self.training_args.num_epochs + 1):
            print("--------------------------------")
            logger.info(f"Epoch {epoch}/{self.training_args.num_epochs}")
            
            # Training loop
            model.train()
            total_train_loss, train_depth_metrics, train_stones_metrics, train_soiltype_metrics, train_soilcolor_metrics, train_carbonate_metrics, train_humus_metrics, train_rooting_metrics, train_horizon_metrics = self._run_model(train_loader, self.training_args.device, model, mode='train', optimizer=optimizer)
            
            # Evaluation loop
            model.eval() # Set model in evaluation mode before running inference
            total_val_loss, val_depth_metrics, val_stones_metrics, val_soiltype_metrics, val_soilcolor_metrics, val_carbonate_metrics, val_humus_metrics, val_rooting_metrics, val_horizon_metrics = self._run_model(val_loader, self.training_args.device, model, mode='val')
            
            # Update metrics for all the tasks and subtasks
            epoch_metrics = {
                'epoch' : epoch,
                'train_loss' : total_train_loss,
                'val_loss' : total_val_loss
            }
            for d in [
                train_depth_metrics,
                train_stones_metrics,
                train_soiltype_metrics,
                train_soilcolor_metrics,
                train_carbonate_metrics,
                train_humus_metrics,
                train_rooting_metrics,
                train_horizon_metrics,
                val_depth_metrics,
                val_stones_metrics,
                val_soiltype_metrics,
                val_soilcolor_metrics,
                val_carbonate_metrics,
                val_humus_metrics,
                val_rooting_metrics,
                val_horizon_metrics
            ]:
                epoch_metrics.update(d)
            
            for callback in self.training_args.callbacks:
                callback(model, epoch_metrics, epoch)
            
            # Log metrics to wandb
            wandb.log(epoch_metrics)
            scheduler.step(total_val_loss)
            
            # Log the current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            self.histories.append(epoch_metrics)
            
            logger.info(
                f"Epoch {epoch}/{self.training_args.num_epochs} Metrics:\n"
                "--------------------------------\n"
                "Training:\n"
                f"- Total Loss: {total_train_loss:.4f}\n"
                f"- Depth Loss: {train_depth_metrics['train_Depth_loss']:.4f}\n"
                f"- Depth IoU: {train_depth_metrics['train_Depth_IoU']:.4f}\n"
                f"- Tabulars:\n"
                f"{self._fancy_print(epoch_metrics, key_prefix='train_')}\n"
                f"- Horizon (Cosine) Loss: {train_horizon_metrics['train_Horizon_cosine_loss']:.4f}\n"
                f"- Horizon Accuracy: {train_horizon_metrics['train_Horizon_accuracy']:.4f}\n"
                f"- Horizon Top-{self.hor_topk} Accuracy: {train_horizon_metrics['train_Horizon_topk_accuracy']:.4f}\n"
                f"- Horizon Precision: {train_horizon_metrics['train_Horizon_precision']:.4f}\n"
                f"- Horizon Recall: {train_horizon_metrics['train_Horizon_recall']:.4f}\n"
                f"- Horizon F1: {train_horizon_metrics['train_Horizon_f1']:.4f}\n"
                f"- Horizon Precision@{self.hor_topk}: {train_horizon_metrics['train_Horizon_precision_at_k']:.4f}\n"
                f"- Horizon Recall@{self.hor_topk}: {train_horizon_metrics['train_Horizon_recall_at_k']:.4f}\n"
                "\nValidation:\n"
                f"- Total Loss: {total_val_loss:.4f}\n"
                f"- Depth Loss: {val_depth_metrics['val_Depth_loss']:.4f}\n"
                f"- Depth IoU: {val_depth_metrics['val_Depth_IoU']:.4f}\n"
                f"- Tabulars:\n"
                f"{self._fancy_print(epoch_metrics, key_prefix='val_')}\n"
                f"- Horizon (Cosine) Loss: {val_horizon_metrics['val_Horizon_cosine_loss']:.4f}\n"
                f"- Horizon Accuracy: {val_horizon_metrics['val_Horizon_accuracy']:.4f}\n"
                f"- Horizon Top-{self.hor_topk} Accuracy: {val_horizon_metrics['val_Horizon_topk_accuracy']:.4f}\n"
                f"- Horizon Precision: {val_horizon_metrics['val_Horizon_precision']:.4f}\n"
                f"- Horizon Recall: {val_horizon_metrics['val_Horizon_recall']:.4f}\n"
                f"- Horizon F1: {val_horizon_metrics['val_Horizon_f1']:.4f}\n"
                f"- Horizon Precision@{self.hor_topk}: {val_horizon_metrics['val_Horizon_precision_at_k']:.4f}\n"
                f"- Horizon Recall@{self.hor_topk}: {val_horizon_metrics['val_Horizon_recall_at_k']:.4f}\n"
                f"Current Learning Rate: {current_lr}\n"
                "--------------------------------"
            )
            
            # Check early stopping
            if self.training_args.use_early_stopping:
                early_stopping = [cb for cb in self.training_args.callbacks if type(cb).__name__ == 'EarlyStopping'][0]
                if early_stopping.should_stop:
                    logger.info("Early stopping activated.")
                    break
            
        self.trained = True
        
        return model, self.histories[-1] # Return the last epoch metrics
            
    def test(self,
        model: nn.Module,
        test_df: pd.DataFrame,
        model_output_dir: str,
        wandb_image_logging: bool
    ) -> dict:
        test_dataset = ImageTabularEnd2EndDataset(
            dataframe=test_df,
            normalize=self.image_normalization,
            label_column=self.target,
            geotemp_columns=self.dataprocessor.geotemp_img_infos[:-1], # without img path
            tab_num_columns=self.segments_tabular_num_feature_columns,
            tab_categ_columns=self.segments_tabular_categ_feature_columns
        )
        
        test_loader = test_dataset.to_dataloader(batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        model.to(self.training_args.device)
        
        print("--------------------------------")
        # Evaluation loop
        model.eval() # Set model in evaluation mode before running inference
        
        total_test_loss, test_depth_metrics, test_stones_metrics, test_soiltype_metrics, test_soilcolor_metrics, test_carbonate_metrics, test_humus_metrics, test_rooting_metrics, test_horizon_metrics = self._run_model(test_loader, self.training_args.device, model, mode='test')
        
        test_metrics = {'test_loss' : total_test_loss}
        for d in [
            test_depth_metrics,
            test_stones_metrics,
            test_soiltype_metrics,
            test_soilcolor_metrics,
            test_carbonate_metrics,
            test_humus_metrics,
            test_rooting_metrics,
            test_horizon_metrics
        ]:
            test_metrics.update(d)
            
        logger.info(
            "Test Metrics:\n"
            "--------------------------------\n"
            f"- Total Loss: {total_test_loss:.4f}\n"
            f"- Depth Loss: {test_depth_metrics['test_Depth_loss']:.4f}\n"
            f"- Depth IoU: {test_depth_metrics['test_Depth_IoU']:.4f}\n"
            f"- Tabulars:\n"
            f"{self._fancy_print(test_metrics, key_prefix='test_')}\n"
            f"- Horizon (Cosine) Loss: {test_horizon_metrics['test_Horizon_cosine_loss']:.4f}\n"
            f"- Horizon Accuracy: {test_horizon_metrics['test_Horizon_accuracy']:.4f}\n"
            f"- Horizon Top-{self.hor_topk} Accuracy: {test_horizon_metrics['test_Horizon_topk_accuracy']:.4f}\n"
            f"- Horizon Precision: {test_horizon_metrics['test_Horizon_precision']:.4f}\n"
            f"- Horizon Recall: {test_horizon_metrics['test_Horizon_recall']:.4f}\n"
            f"- Horizon F1: {test_horizon_metrics['test_Horizon_f1']:.4f}\n"
            f"- Horizon Precision@{self.hor_topk}: {test_horizon_metrics['test_Horizon_precision_at_k']:.4f}\n"
            f"- Horizon Recall@{self.hor_topk}: {test_horizon_metrics['test_Horizon_recall_at_k']:.4f}\n"
            "--------------------------------"
        )
        
        # Plot confusion matrix for horizon predictions
        if self.hor_labels['train']:
            self._plot_confusion_matrices(labels=self.hor_labels['train'], predictions=self.hor_predictions['train'], emb_dict=self.dataprocessor.embeddings_dict, model_output_dir=model_output_dir, wandb_image_logging=wandb_image_logging, mode='train')
        if self.hor_labels['val']:
            self._plot_confusion_matrices(labels=self.hor_labels['val'], predictions=self.hor_predictions['val'], emb_dict=self.dataprocessor.embeddings_dict, model_output_dir=model_output_dir, wandb_image_logging=wandb_image_logging, mode='val')
        self._plot_confusion_matrices(labels=self.hor_labels['test'], predictions=self.hor_predictions['test'], emb_dict=self.dataprocessor.embeddings_dict, model_output_dir=model_output_dir, wandb_image_logging=wandb_image_logging, mode='test')
        
        return test_metrics
    
    def get_model(self) -> nn.Module:
        return SoilNet_LSTM(
            # Parameters for image encoder, geotemp encoder and depth predictor:
            geo_temp_input_dim        = len(self.dataprocessor.geotemp_img_infos) - 2, # without index and img path
            geo_temp_output_dim       = self.hyperparameters['geotemp_output_dim'],
            image_encoder_output_dim  = self.hyperparameters['image_encoder_output_dim'],
            max_seq_len               = self.hyperparameters['max_seq_len'],
            stop_token                = self.hyperparameters['stop_token'],
            depth_rnn_hidden_dim      = self.hyperparameters['depth_rnn_hidden_dim'],
            img_patch_size            = self.hyperparameters['img_patch_size'],
            segments_random_patches = True, # True = use ResNet, False = use custom CNN
            num_patches_per_segment=self.hyperparameters['num_patches_per_segment'],
            segment_random_patch_size=self.hyperparameters['segment_random_patch_size'],
            
            # Parameters for tabular predictors:
            tabular_output_dim_dict    = self.tabulars_output_dim_dict,
            segment_encoder_output_dim = self.hyperparameters['segment_encoder_output_dim'],
            tab_rnn_hidden_dim         = self.hyperparameters['tab_rnn_hidden_dim'],
            tab_num_lstm_layers        = self.hyperparameters['tab_num_lstm_layers'],
            
            # Parameters for horizon predictor:
            segments_tabular_output_dim = self.hyperparameters['segments_tabular_output_dim'],
            embedding_dim               = np.shape(self.dataprocessor.embeddings_dict['embedding'])[1],
            
            # Parameters for the model:
            teacher_forcing_stop_epoch  = self.hyperparameters['teacher_forcing_stop_epoch'],
            teacher_forcing_approach    = self.hyperparameters['teacher_forcing_approach']
        )
    
    def plot_losses(self, model_output_dir, wandb_image_logging):
        # Extract losses for each subtask from self.histories
        epochs = [epoch_metrics['epoch'] for epoch_metrics in self.histories]
        all_train_losses = {
            'Depth':      [epoch_metrics.get('train_Depth_loss', float('nan')) for epoch_metrics in self.histories],
            'Stones':     [epoch_metrics.get('train_Steine_loss', float('nan')) for epoch_metrics in self.histories],
            'Soil Type':  [epoch_metrics.get('train_Bodenart_loss', float('nan')) for epoch_metrics in self.histories],
            'Soil Color': [epoch_metrics.get('train_Bodenfarbe_loss', float('nan')) for epoch_metrics in self.histories],
            'Carbonate':  [epoch_metrics.get('train_Karbonat_loss', float('nan')) for epoch_metrics in self.histories],
            'Humus':      [epoch_metrics.get('train_Humusgehaltsklasse_loss', float('nan')) for epoch_metrics in self.histories],
            'Rooting':    [epoch_metrics.get('train_Durchwurzelung_loss', float('nan')) for epoch_metrics in self.histories],
            'Horizon':    [epoch_metrics.get('train_Horizon_cosine_loss', float('nan')) for epoch_metrics in self.histories],
            'Total':      [epoch_metrics.get('train_loss', float('nan')) for epoch_metrics in self.histories]
        }
        all_val_losses = {
            'Depth':      [epoch_metrics.get('val_Depth_loss', float('nan')) for epoch_metrics in self.histories],
            'Stones':     [epoch_metrics.get('val_Steine_loss', float('nan')) for epoch_metrics in self.histories],
            'Soil Type':  [epoch_metrics.get('val_Bodenart_loss', float('nan')) for epoch_metrics in self.histories],
            'Soil Color': [epoch_metrics.get('val_Bodenfarbe_loss', float('nan')) for epoch_metrics in self.histories],
            'Carbonate':  [epoch_metrics.get('val_Karbonat_loss', float('nan')) for epoch_metrics in self.histories],
            'Humus':      [epoch_metrics.get('val_Humusgehaltsklasse_loss', float('nan')) for epoch_metrics in self.histories],
            'Rooting':    [epoch_metrics.get('val_Durchwurzelung_loss', float('nan')) for epoch_metrics in self.histories],
            'Horizon':    [epoch_metrics.get('val_Horizon_cosine_loss', float('nan')) for epoch_metrics in self.histories],
            'Total':      [epoch_metrics.get('val_loss', float('nan')) for epoch_metrics in self.histories]
        }
        all_train_metrics = {
            'IoU': [epoch_metrics.get('train_Depth_IoU', float('nan')) for epoch_metrics in self.histories],
            'Accuracy': [epoch_metrics.get('train_Horizon_accuracy', float('nan')) for epoch_metrics in self.histories],
            f'Top-{self.hor_topk} Accuracy': [epoch_metrics.get('train_Horizon_topk_accuracy', float('nan')) for epoch_metrics in self.histories],
            'F1 Score': [epoch_metrics.get('train_Horizon_f1', float('nan')) for epoch_metrics in self.histories]
        }
        all_val_metrics = {
            'IoU': [epoch_metrics.get('val_Depth_IoU', float('nan')) for epoch_metrics in self.histories],
            'Accuracy': [epoch_metrics.get('val_Horizon_accuracy', float('nan')) for epoch_metrics in self.histories],
            f'Top-{self.hor_topk} Accuracy': [epoch_metrics.get('val_Horizon_topk_accuracy', float('nan')) for epoch_metrics in self.histories],
            'F1 Score': [epoch_metrics.get('val_Horizon_f1', float('nan')) for epoch_metrics in self.histories]
        }

        ### Plot losses
        # Create a 3x3 subplot
        fig, axes = plt.subplots(3, 3, figsize=(16, 16))
        fig.suptitle('Training Losses for Subtasks', fontsize=18)

        # Plot each loss in a separate subplot
        for ax, (task, train_losses), val_losses in zip(axes.flatten(), all_train_losses.items(), all_val_losses.values()):
            ax.plot(epochs, train_losses, label=f'Train {task} Loss', color='b', marker='o')
            ax.plot(epochs, val_losses, label=f'Validation {task} Loss', color='r', marker='o')
            ax.set_title(task)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True)
            ax.legend()

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the plot to the specified directory
        plot_path = os.path.join(model_output_dir, 'training_losses.pdf')
        plt.savefig(plot_path, bbox_inches='tight', format='pdf')
        logger.info(f"Loss plot saved to {plot_path}")

        # Optionally log the plot to wandb
        if wandb_image_logging:
            wandb.log({"Training Losses": wandb.Image(fig)})
        plt.close(fig)
        
        ### Plot extra metrics
        # Create a 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Metrics for Horizon Predictions', fontsize=16)

        # Plot each metric in a separate subplot
        for ax, (task, train_metrics), val_metrics in zip(axes.flatten(), all_train_metrics.items(), all_val_metrics.values()):
            ax.plot(epochs, train_metrics, label=f'Train {task}', color='b', marker='o')
            ax.plot(epochs, val_metrics, label=f'Validation {task}', color='r', marker='o')
            ax.set_title(task)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(task)
            ax.grid(True)
            ax.legend()

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the plot to the specified directory
        plot_path = os.path.join(model_output_dir, 'training_metrics.pdf')
        plt.savefig(plot_path, bbox_inches='tight', format='pdf')
        logger.info(f"Metrics plot saved to {plot_path}")

        # Optionally log the plot to wandb
        if wandb_image_logging:
            wandb.log({"Training Metrics": wandb.Image(fig)})
        plt.close(fig)
        
        ### Plot the bisector line for stones predictions
        splits = ['train', 'val', 'test']
        for split in splits:
            true_values = self.stones_true_values[split]
            predicted_values = self.stones_predictions[split]

            # Create scatter plot
            fig = plt.figure(figsize=(8, 8))
            lims = [min(true_values), max(true_values)]
            plt.axes(aspect='equal')
            plt.scatter(true_values, predicted_values, alpha=0.5, label=f'{split.capitalize()} Data', color='blue')
            plt.xlim(lims)
            plt.ylim(lims)
            plt.plot(lims, lims, 'r--', label='Bisector (Ideal)', linewidth=2)
            plt.title(f'True vs Predicted Stones ({split.capitalize()})', fontsize=16)
            plt.xlabel('True Values', fontsize=14)
            plt.ylabel('Predicted Values', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)

            # Save the plot
            plot_path = os.path.join(model_output_dir, f'stones_predictions_{split}.pdf')
            plt.savefig(plot_path, bbox_inches='tight', format='pdf')
            logger.info(f"Stones prediction plot for {split} saved to {plot_path}")

            # Optionally log the plot to wandb
            if wandb_image_logging:
                wandb.log({f"Stones Predictions ({split.capitalize()})": wandb.Image(fig)})

            plt.close()

    def _run_model(self, data_loader, device, model, mode='val', optimizer=None):
        ### Initialize losses and metrics
        run_total_loss = 0.0
        run_depth_loss = 0.0
        run_stones_loss, run_soiltype_loss, run_soilcolor_loss, run_carbonate_loss, run_humus_loss, run_rooting_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        run_horizon_loss = 0.0
        
        iou = 0.0 # Depth IoU computed extra during each batch
        
        horizon_correct = 0 # Horizon accuracy is computed differently than tabs accuracy (through similarity of embeddings)
        horizon_topk_correct = 0

        all_topk_soiltype_predictions, all_topk_soilcolor_predictions, all_topk_carbonate_predictions, all_topk_humus_predictions, all_topk_rooting_predictions, all_topk_horizon_predictions = [], [], [], [], [], []
        
        all_soiltype_labels, all_soilcolor_labels, all_carbonate_labels, all_humus_labels, all_rooting_labels, all_horizon_labels = [], [], [], [], [], []
        
        self.stones_predictions[mode] = []
        self.stones_true_values[mode] = []
        
        # Iterate over batches
        data_loader_tqdm = tqdm(data_loader, desc=f"{mode.capitalize()}", leave=False, unit="batch")
        for batch in data_loader_tqdm:
            padded_images, image_mask, geotemp_features, padded_true_depths, padded_segments_tabulars_labels, padded_true_horizon_indices = batch
            padded_images, image_mask, geotemp_features, padded_true_depths, padded_segments_tabulars_labels, padded_true_horizon_indices = (
                padded_images.to(device),
                image_mask.to(device),
                geotemp_features.to(device),
                padded_true_depths.to(device),
                padded_segments_tabulars_labels.to(device),
                padded_true_horizon_indices.to(device)
            )
            
            if mode == 'train':
                optimizer.zero_grad()
            
            with torch.set_grad_enabled(mode == 'train'):
                
                ### Get true targets for all (sub)tasks
                
                ## True tabs
                # Mask for valid indices
                mask = padded_true_horizon_indices != -1
                # Retrieve true labels
                start_idx = 0
                tabular_sliced = {}
                dict_lengths = { 'Steine' : 1 }
                dict_lengths.update(self.segments_tabular_categ_feature_columns)
                for tab_col, tab_col_dim in dict_lengths.items():
                    end_idx = start_idx + tab_col_dim
                    tabular_sliced[tab_col] = padded_segments_tabulars_labels[:, :, start_idx:end_idx][mask]
                    start_idx = end_idx
                    
                stones_true      = tabular_sliced['Steine']
                soiltype_labels  = torch.argmax(tabular_sliced['Bodenart'], dim=1)
                soilcolor_labels = torch.argmax(tabular_sliced['Bodenfarbe'], dim=1)
                carbonate_labels = torch.argmax(tabular_sliced['Karbonat'], dim=1)
                humus_labels     = torch.argmax(tabular_sliced['Humusgehaltsklasse'], dim=1)
                rooting_labels   = torch.argmax(tabular_sliced['Durchwurzelung'], dim=1)
                
                ## True horizons
                true_horizon_embeddings = torch.stack([torch.tensor(self.dataprocessor.embeddings_dict['embedding'][lab.item()]) for lab in padded_true_horizon_indices.view(-1) if lab != -1]).to(device)
                true_horizon_indices = padded_true_horizon_indices.view(-1)[padded_true_horizon_indices.view(-1) != -1]
                
                
                ### Predictions for all (sub)tasks
                padded_pred_depths, padded_pred_tabulars, padded_pred_horizon_embeddings = model(
                    padded_images,
                    image_mask,
                    geotemp_features[:, 1:], # 'index' column not used in model
                    padded_true_depths,
                    padded_segments_tabulars_labels
                )
                
                ## Filter predictions only at valid positions
                pred_tabulars = {key: value[mask] for key, value in padded_pred_tabulars.items()}
                stones_predictions    = pred_tabulars['Steine']
                soiltype_predictions  = pred_tabulars['Bodenart']
                soilcolor_predictions = pred_tabulars['Bodenfarbe']
                carbonate_predictions = pred_tabulars['Karbonat']
                humus_predictions     = pred_tabulars['Humusgehaltsklasse']
                rooting_predictions   = pred_tabulars['Durchwurzelung']
                
                pred_horizon_embeddings   = torch.stack([pred for pred, lab in zip(padded_pred_horizon_embeddings.view(-1, padded_pred_horizon_embeddings.size(-1)), padded_true_horizon_indices.view(-1)) if lab != -1]).to(device)
                pred_topk_horizon_indices = torch.topk(torch.matmul(pred_horizon_embeddings, self.label_embeddings_tensor.T), k=self.hor_topk, dim=1).indices


                ### Calculate losses
                ## Depth loss
                depth_loss = self.depth_loss(padded_pred_depths, padded_true_depths)
                
                ## Tabular losses
                stones_loss    = self.mse_loss(stones_predictions, stones_true)
                soiltype_loss  = self.cross_entropy_loss(soiltype_predictions, soiltype_labels)
                soilcolor_loss = self.cross_entropy_loss(soilcolor_predictions, soilcolor_labels)
                carbonate_loss = self.cross_entropy_loss(carbonate_predictions, carbonate_labels)
                humus_loss     = self.cross_entropy_loss(humus_predictions, humus_labels)
                rooting_loss   = self.cross_entropy_loss(rooting_predictions, rooting_labels)
                
                ## Horizon loss
                # Normalize pred. embeddings for the cosine loss, true embeddings are already normalized
                pred_horizon_embeddings = F.normalize(pred_horizon_embeddings, p=2, dim=1)
                # Create a dummy "same class" tensor with 1s for the cosine similarity
                same_class = torch.ones(pred_horizon_embeddings.size(0)).to(device)
                horizon_loss = self.cosine_loss(pred_horizon_embeddings, true_horizon_embeddings, same_class)

                ## Total loss (sum of all losses)
                total_loss = 10*depth_loss + stones_loss/10. + soiltype_loss + soilcolor_loss + carbonate_loss + humus_loss + rooting_loss + 10*horizon_loss
                
                if mode == 'train':
                    # Backpropagation
                    total_loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Update running losses
                run_total_loss     += total_loss.item()
                run_depth_loss     += depth_loss.item()
                run_stones_loss    += stones_loss.item()
                run_soiltype_loss  += soiltype_loss.item()
                run_soilcolor_loss += soilcolor_loss.item()
                run_carbonate_loss += carbonate_loss.item()
                run_humus_loss     += humus_loss.item()
                run_rooting_loss   += rooting_loss.item()
                run_horizon_loss   += horizon_loss.item()
                
                # Update depth IoU separately
                iou += depth_iou(padded_pred_depths, padded_true_depths, model.depth_marker_predictor.stop_token)

                # Top-K predictions
                _, topk_soiltype_predictions  = torch.topk(soiltype_predictions, self.tab_topk)
                _, topk_soilcolor_predictions = torch.topk(soilcolor_predictions, self.tab_topk)
                _, topk_carbonate_predictions = torch.topk(carbonate_predictions, self.tab_topk)
                _, topk_humus_predictions     = torch.topk(humus_predictions, self.tab_topk)
                _, topk_rooting_predictions   = torch.topk(rooting_predictions, self.tab_topk)
                
                horizon_correct      += self.hor_topk_acc(1)(pred_horizon_embeddings, true_horizon_indices)
                horizon_topk_correct += self.hor_topk_acc(self.hor_topk)(pred_horizon_embeddings, true_horizon_indices)

                # Add predictions and true values to lists (for bisector)
                self.stones_predictions[mode].append(stones_predictions.detach().cpu())
                self.stones_true_values[mode].append(stones_true.detach().cpu())
                
                # Append topk predictions and labels for Precision@k, Recall@k and F1 score
                all_topk_soiltype_predictions.append(topk_soiltype_predictions.detach().cpu())
                all_topk_soilcolor_predictions.append(topk_soilcolor_predictions.detach().cpu())
                all_topk_carbonate_predictions.append(topk_carbonate_predictions.detach().cpu())
                all_topk_humus_predictions.append(topk_humus_predictions.detach().cpu())
                all_topk_rooting_predictions.append(topk_rooting_predictions.detach().cpu())
                all_topk_horizon_predictions.append(pred_topk_horizon_indices.cpu())

                all_soiltype_labels.append(soiltype_labels.detach().cpu())
                all_soilcolor_labels.append(soilcolor_labels.detach().cpu())
                all_carbonate_labels.append(carbonate_labels.detach().cpu())
                all_humus_labels.append(humus_labels.detach().cpu())
                all_rooting_labels.append(rooting_labels.detach().cpu())
                all_horizon_labels.append(true_horizon_indices.cpu())

                data_loader_tqdm.set_postfix(loss=total_loss.item())

        # Average losses over the batches
        run_total_loss     /= len(data_loader)
        run_depth_loss     /= len(data_loader)
        run_stones_loss    /= len(data_loader)
        run_soiltype_loss  /= len(data_loader)
        run_soilcolor_loss /= len(data_loader)
        run_carbonate_loss /= len(data_loader)
        run_humus_loss     /= len(data_loader)
        run_rooting_loss   /= len(data_loader)
        run_horizon_loss   /= len(data_loader)

        # Average IoU and horizon accuracies separately
        iou /= len(data_loader)
        eval_horizon_acc      = horizon_correct / len(data_loader)
        eval_horizon_topk_acc = horizon_topk_correct / len(data_loader)

        # Concatenate and change dtype of predictions and true values
        self.stones_predictions[mode] = torch.cat(self.stones_predictions[mode]).numpy().flatten().tolist()
        self.stones_true_values[mode] = torch.cat(self.stones_true_values[mode]).numpy().flatten().tolist()
        
        # Concatenate all top-k predictions and true values
        all_topk_soiltype_predictions = torch.cat(all_topk_soiltype_predictions).numpy()
        all_topk_soilcolor_predictions = torch.cat(all_topk_soilcolor_predictions).numpy()
        all_topk_carbonate_predictions = torch.cat(all_topk_carbonate_predictions).numpy()
        all_topk_humus_predictions = torch.cat(all_topk_humus_predictions).numpy()
        all_topk_rooting_predictions = torch.cat(all_topk_rooting_predictions).numpy()
        #all_topk_horizon_predictions   = torch.cat(all_topk_horizon_predictions).numpy() # apply numpy() later in metrics
        all_soiltype_labels = torch.cat(all_soiltype_labels).numpy()
        all_soilcolor_labels = torch.cat(all_soilcolor_labels).numpy()
        all_carbonate_labels = torch.cat(all_carbonate_labels).numpy()
        all_humus_labels = torch.cat(all_humus_labels).numpy()
        all_rooting_labels = torch.cat(all_rooting_labels).numpy()
        all_horizon_labels = torch.cat(all_horizon_labels).numpy()

        # Get top-1 predictions
        top1_soiltype_predictions = all_topk_soiltype_predictions[:, 0]
        top1_soilcolor_predictions = all_topk_soilcolor_predictions[:, 0]
        top1_carbonate_predictions = all_topk_carbonate_predictions[:, 0]
        top1_humus_predictions = all_topk_humus_predictions[:, 0]
        top1_rooting_predictions = all_topk_rooting_predictions[:, 0]
        top1_horizon_predictions = torch.cat(all_topk_horizon_predictions)[:, 0]
        # Plus topk-horizons computed extra
        topk_horizon_predictions = torch.cat(all_topk_horizon_predictions)

        # Possible labels for sklearn metrics
        possible_soiltype_labels = list(range(self.tabulars_output_dim_dict['Bodenart']))
        possible_soilcolor_labels = list(range(self.tabulars_output_dim_dict['Bodenfarbe']))
        possible_carbonate_labels = list(range(self.tabulars_output_dim_dict['Karbonat']))
        possible_humus_labels = list(range(self.tabulars_output_dim_dict['Humusgehaltsklasse']))
        possible_rooting_labels = list(range(self.tabulars_output_dim_dict['Durchwurzelung']))

        # Calculate metrics
        depth_metrics = {
            f'{mode}_Depth_loss' : run_depth_loss,
            f'{mode}_Depth_IoU': iou.detach().cpu().numpy()
        }
        stones_metrics = {
            f'{mode}_Steine_loss': run_stones_loss
        }
        precision_at_k, recal_at_k = precision_recall_at_k(all_soiltype_labels, all_topk_soiltype_predictions, all_labels=possible_soiltype_labels, average=self.tab_class_average)
        soiltype_metrics = {
            f'{mode}_Bodenart_loss': run_soiltype_loss,
            f'{mode}_Bodenart_accuracy': accuracy_score(all_soiltype_labels, top1_soiltype_predictions),
            f'{mode}_Bodenart_f1': f1_score(all_soiltype_labels, top1_soiltype_predictions, labels=possible_soiltype_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Bodenart_precision': precision_score(all_soiltype_labels, top1_soiltype_predictions, labels=possible_soiltype_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Bodenart_recall': recall_score(all_soiltype_labels, top1_soiltype_predictions, labels=possible_soiltype_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Bodenart_top_k_accuracy': top_k_accuracy_from_indices(all_soiltype_labels, all_topk_soiltype_predictions),
            f'{mode}_Bodenart_precision_at_k': precision_at_k,
            f'{mode}_Bodenart_recall_at_k': recal_at_k
        }
        precision_at_k, recal_at_k = precision_recall_at_k(all_soilcolor_labels, all_topk_soilcolor_predictions, all_labels=possible_soilcolor_labels, average=self.tab_class_average)
        soilcolor_metrics = {
            f'{mode}_Bodenfarbe_loss': run_soilcolor_loss,
            f'{mode}_Bodenfarbe_accuracy': accuracy_score(all_soilcolor_labels, top1_soilcolor_predictions),
            f'{mode}_Bodenfarbe_f1': f1_score(all_soilcolor_labels, top1_soilcolor_predictions, labels=possible_soilcolor_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Bodenfarbe_precision': precision_score(all_soilcolor_labels, top1_soilcolor_predictions, labels=possible_soilcolor_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Bodenfarbe_recall': recall_score(all_soilcolor_labels, top1_soilcolor_predictions, labels=possible_soilcolor_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Bodenfarbe_top_k_accuracy': top_k_accuracy_from_indices(all_soilcolor_labels, all_topk_soilcolor_predictions),
            f'{mode}_Bodenfarbe_precision_at_k': precision_at_k,
            f'{mode}_Bodenfarbe_recall_at_k': recal_at_k
        }
        precision_at_k, recal_at_k = precision_recall_at_k(all_carbonate_labels, all_topk_carbonate_predictions, all_labels=possible_carbonate_labels, average=self.tab_class_average)
        carbonate_metrics = {
            f'{mode}_Karbonat_loss': run_carbonate_loss,
            f'{mode}_Karbonat_accuracy': accuracy_score(all_carbonate_labels, top1_carbonate_predictions),
            f'{mode}_Karbonat_f1': f1_score(all_carbonate_labels, top1_carbonate_predictions, labels=possible_carbonate_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Karbonat_precision': precision_score(all_carbonate_labels, top1_carbonate_predictions, labels=possible_carbonate_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Karbonat_recall': recall_score(all_carbonate_labels, top1_carbonate_predictions, labels=possible_carbonate_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Karbonat_top_k_accuracy': top_k_accuracy_from_indices(all_carbonate_labels, all_topk_carbonate_predictions),
            f'{mode}_Karbonat_precision_at_k': precision_at_k,
            f'{mode}_Karbonat_recall_at_k': recal_at_k
        }
        precision_at_k, recal_at_k = precision_recall_at_k(all_humus_labels, all_topk_humus_predictions, all_labels=possible_humus_labels, average=self.tab_class_average)
        humus_metrics = {
            f'{mode}_Humusgehaltsklasse_loss': run_humus_loss,
            f'{mode}_Humusgehaltsklasse_accuracy': accuracy_score(all_humus_labels, top1_humus_predictions),
            f'{mode}_Humusgehaltsklasse_f1': f1_score(all_humus_labels, top1_humus_predictions, labels=possible_humus_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Humusgehaltsklasse_precision': precision_score(all_humus_labels, top1_humus_predictions, labels=possible_humus_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Humusgehaltsklasse_recall': recall_score(all_humus_labels, top1_humus_predictions, labels=possible_humus_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Humusgehaltsklasse_top_k_accuracy': top_k_accuracy_from_indices(all_humus_labels, all_topk_humus_predictions),
            f'{mode}_Humusgehaltsklasse_precision_at_k': precision_at_k,
            f'{mode}_Humusgehaltsklasse_recall_at_k': recal_at_k
        }
        precision_at_k, recal_at_k = precision_recall_at_k(all_rooting_labels, all_topk_rooting_predictions, all_labels=possible_rooting_labels, average=self.tab_class_average)
        rooting_metrics = {
            f'{mode}_Durchwurzelung_loss': run_rooting_loss,
            f'{mode}_Durchwurzelung_accuracy': accuracy_score(all_rooting_labels, top1_rooting_predictions),
            f'{mode}_Durchwurzelung_f1': f1_score(all_rooting_labels, top1_rooting_predictions, labels=possible_rooting_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Durchwurzelung_precision': precision_score(all_rooting_labels, top1_rooting_predictions, labels=possible_rooting_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Durchwurzelung_recall': recall_score(all_rooting_labels, top1_rooting_predictions, labels=possible_rooting_labels, average=self.tab_class_average, zero_division=0),
            f'{mode}_Durchwurzelung_top_k_accuracy': top_k_accuracy_from_indices(all_rooting_labels, all_topk_rooting_predictions),
            f'{mode}_Durchwurzelung_precision_at_k': precision_at_k,
            f'{mode}_Durchwurzelung_recall_at_k': recal_at_k
        }
        precision_at_k, recall_at_k = precision_recall_at_k(all_horizon_labels, topk_horizon_predictions.numpy(), all_labels=self.hor_possible_labels, average=self.hor_class_average)
        horizon_metrics = {
            f'{mode}_Horizon_cosine_loss': run_horizon_loss,
            f'{mode}_Horizon_accuracy': eval_horizon_acc,
            f'{mode}_Horizon_topk_accuracy': eval_horizon_topk_acc,
            f'{mode}_Horizon_precision': precision_score(all_horizon_labels, top1_horizon_predictions.numpy(), labels=self.hor_possible_labels, average=self.hor_class_average, zero_division=0),
            f'{mode}_Horizon_recall': recall_score(all_horizon_labels, top1_horizon_predictions.numpy(), labels=self.hor_possible_labels, average=self.hor_class_average, zero_division=0),
            f'{mode}_Horizon_f1': f1_score(all_horizon_labels, top1_horizon_predictions.numpy(), labels=self.hor_possible_labels, average=self.hor_class_average, zero_division=0),
            f'{mode}_Horizon_precision_at_k': precision_at_k,
            f'{mode}_Horizon_recall_at_k': recall_at_k
        }
        
        # Store labels and predictions for confusion matrix
        self.hor_labels[mode]      = all_horizon_labels
        self.hor_predictions[mode] = top1_horizon_predictions.numpy()

        return run_total_loss, \
            depth_metrics, \
            stones_metrics, \
            soiltype_metrics, \
            soilcolor_metrics, \
            carbonate_metrics, \
            humus_metrics, \
            rooting_metrics, \
            horizon_metrics
    
    def _fancy_print(self, epoch_metrics, key_prefix=''):
        """For printing table of tabular metrics"""
        metrics = {
            "Loss" : "loss",
            "Accuracy" : "accuracy",
            "F1" : "f1",
            "Precision" : "precision",
            "Recall" : "recall",
            f"Top-{self.tab_topk} Acc." : "top_k_accuracy",
            f"Prec. @{self.tab_topk}" : "precision_at_k",
            f"Recall @{self.tab_topk}" : "recall_at_k",
        }
        
        categories = {
            "Stones" : "Steine",
            "Soil Type" : "Bodenart",
            "Soil Color" : "Bodenfarbe",
            "Carbonate" : "Karbonat",
            "Humus" : "Humusgehaltsklasse",
            "Rooting" : "Durchwurzelung"
        }
        
        data = {}
        
        for category, category_key in categories.items():
            data[category] = [epoch_metrics.get(f"{key_prefix}{category_key}_{metric}", float('nan')) for metric in metrics.values()]
            
        table = pd.DataFrame(data, index=metrics.keys())
        table.index.name = None

        # Round the values to 4 decimal places
        table = table.round(4)
        
        return tabulate(table, headers='keys', tablefmt='mixed_grid', showindex=True)
    
    @staticmethod
    def get_experiment_hyperparameters():
        return {
            # Parameters for image encoder, geotemp encoder and depth predictor:
            'geotemp_output_dim': 256,
            'image_encoder_output_dim': 512,
            'max_seq_len': 8,
            'stop_token': 1.0,
            'depth_rnn_hidden_dim': 256,
            'img_patch_size': 512,
            'num_patches_per_segment': 8,
            'segment_random_patch_size': 224,
            
            # Parameters for tabular predictors:
            'segment_encoder_output_dim': 512,
            'tab_rnn_hidden_dim': 1024,
            'tab_num_lstm_layers': 2,
            
            # Parameters for horizon predictor:
            'segments_tabular_output_dim': 256,
            
            # Parameters for model:
            'teacher_forcing_stop_epoch': 5,
            'teacher_forcing_approach': 'linear' # 'linear' or 'binary'
        }
