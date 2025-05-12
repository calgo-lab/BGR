from __future__ import annotations
import logging
import pandas as pd
import torch
import torch.nn as nn
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

from bgr.soil.data.horizon_tabular_data import HorizonDataProcessor
from bgr.soil.experiments._base import Experiment
from bgr.soil.modelling.tabulars.tabular_models import SimpleTabularModel
from bgr.soil.metrics import top_k_accuracy_from_indices, precision_recall_at_k
from bgr.soil.data.datasets import SegmentsTabularDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTabularsGeotemps(Experiment):
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
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.tab_topk = 3
        self.class_average = 'macro'
        
        # Retrieve the experiment hyperparameters
        self.hyperparameters = SimpleTabularsGeotemps.get_experiment_hyperparameters()
        self.hyperparameters.update(training_args.hyperparameters)
        
        # Initialize dictionary to store lists of stones predictions and true values
        self.stones_predictions = { "train" : [], "val" : [], "test" : [] }
        self.stones_true_values = { "train" : [], "val" : [], "test" : [] }
    
    def train_and_validate(self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_output_dir: str
    ) -> tuple[nn.Module, dict]:
        train_dataset = SegmentsTabularDataset(
            dataframe=train_df,
            normalize=self.image_normalization,
            label_column=self.target,
            geotemp_columns=self.dataprocessor.geotemp_img_infos[1:-1], # without index and img path
            tab_num_columns=self.segments_tabular_num_feature_columns,
            tab_categ_columns=self.segments_tabular_categ_feature_columns
        )
        train_loader = DataLoader(train_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)

        val_dataset = SegmentsTabularDataset(
            dataframe=val_df,
            normalize=self.image_normalization,
            label_column=self.target,
            geotemp_columns=self.dataprocessor.geotemp_img_infos[1:-1], # without index and img path
            tab_num_columns=self.segments_tabular_num_feature_columns,
            tab_categ_columns=self.segments_tabular_categ_feature_columns
        )
        val_loader = DataLoader(val_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        model = self.get_model()
        model.to(self.training_args.device)
        
        lr = self.training_args.learning_rate
        weight_decay = self.training_args.weight_decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.1, min_lr=lr*0.01)
        
        self.histories = []
        
        for epoch in range(1, self.training_args.num_epochs + 1):
            print("--------------------------------")
            logger.info(f"Epoch {epoch}/{self.training_args.num_epochs}")
            
            # Training loop
            model.train()
            total_train_loss, train_stones_metrics, train_soiltype_metrics, train_soilcolor_metrics, train_carbonate_metrics, train_humus_metrics, train_rooting_metrics = self._train_model(train_loader, self.training_args.device, model, optimizer)
            
            # Evaluation loop
            model.eval() # Set model in evaluation mode before running inference
            total_val_loss, val_stones_metrics, val_soiltype_metrics, val_soilcolor_metrics, val_carbonate_metrics, val_humus_metrics, val_rooting_metrics = self._evaluate_model(val_loader, self.training_args.device, model)
            
            epoch_metrics = {
                'epoch' : epoch,
                'train_loss' : total_train_loss,
                'val_loss' : total_val_loss
            }
            for d in [
                train_stones_metrics,
                train_soiltype_metrics,
                train_soilcolor_metrics,
                train_carbonate_metrics,
                train_humus_metrics,
                train_rooting_metrics,
                val_stones_metrics,
                val_soiltype_metrics,
                val_soilcolor_metrics,
                val_carbonate_metrics,
                val_humus_metrics,
                val_rooting_metrics
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
                f"{self._fancy_print(epoch_metrics, key_prefix='train_')}\n"
                "\nValidation:\n"
                f"- Total Loss: {total_val_loss:.4f}\n"
                f"{self._fancy_print(epoch_metrics, key_prefix='val_')}\n"
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
        test_dataset = SegmentsTabularDataset(
            dataframe=test_df,
            normalize=self.image_normalization,
            label_column=self.target,
            geotemp_columns=self.dataprocessor.geotemp_img_infos[1:-1], # without index and img path
            tab_num_columns=self.segments_tabular_num_feature_columns,
            tab_categ_columns=self.segments_tabular_categ_feature_columns
        )
        
        test_loader = DataLoader(test_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        model.to(self.training_args.device)
        
        print("--------------------------------")
        # Evaluation loop
        model.eval() # Set model in evaluation mode before running inference
        
        total_test_loss, test_stones_metrics, test_soiltype_metrics, test_soilcolor_metrics, test_carbonate_metrics, test_humus_metrics, test_rooting_metrics = self._evaluate_model(test_loader, self.training_args.device, model, mode='test')
        
        test_metrics = { 'Test Loss' : total_test_loss }
        for d in [
            test_stones_metrics,
            test_soiltype_metrics,
            test_soilcolor_metrics,
            test_carbonate_metrics,
            test_humus_metrics,
            test_rooting_metrics
        ]:
            test_metrics.update(d)
            
        logger.info(
            "Test Metrics:\n"
            "--------------------------------\n"
            f"- Total Loss: {total_test_loss:.4f}\n"
            f"{self._fancy_print(test_metrics, key_prefix='test_')}\n"
            "--------------------------------"
        )
        
        return test_metrics
    
    def get_model(self) -> nn.Module:
        return SimpleTabularModel(
            tabular_output_dim_dict=self.tabulars_output_dim_dict,
            geotemp_input_dim=len(self.dataprocessor.geotemp_img_infos) - 2, # without index and img path
            segment_encoder_output_dim=self.hyperparameters['segment_encoder_output_dim'],
            geotemp_output_dim=self.hyperparameters['geotemp_output_dim'],
            patch_size=self.hyperparameters['patch_size'],
            rnn_hidden_dim=self.hyperparameters['predictor_hidden_dim'],
            num_lstm_layers=self.hyperparameters['num_lstm_layers'],
            predefined_random_patches=False # True = use ResNet, False = use custom CNN
        )
    
    def plot_losses(self, model_output_dir, wandb_image_logging):
        # Extract losses for each subtask from self.histories
        epochs = [epoch_metrics['epoch'] for epoch_metrics in self.histories]
        all_train_losses = {
            'Stones': [epoch_metrics.get('train_Steine_loss', float('nan')) for epoch_metrics in self.histories],
            'Soil Type': [epoch_metrics.get('train_Bodenart_loss', float('nan')) for epoch_metrics in self.histories],
            'Soil Color': [epoch_metrics.get('train_Bodenfarbe_loss', float('nan')) for epoch_metrics in self.histories],
            'Carbonate': [epoch_metrics.get('train_Karbonat_loss', float('nan')) for epoch_metrics in self.histories],
            'Humus': [epoch_metrics.get('train_Humusgehaltsklasse_loss', float('nan')) for epoch_metrics in self.histories],
            'Rooting': [epoch_metrics.get('train_Durchwurzelung_loss', float('nan')) for epoch_metrics in self.histories],
        }
        all_val_losses = {
            'Stones': [epoch_metrics.get('val_Steine_loss', float('nan')) for epoch_metrics in self.histories],
            'Soil Type': [epoch_metrics.get('val_Bodenart_loss', float('nan')) for epoch_metrics in self.histories],
            'Soil Color': [epoch_metrics.get('val_Bodenfarbe_loss', float('nan')) for epoch_metrics in self.histories],
            'Carbonate': [epoch_metrics.get('val_Karbonat_loss', float('nan')) for epoch_metrics in self.histories],
            'Humus': [epoch_metrics.get('val_Humusgehaltsklasse_loss', float('nan')) for epoch_metrics in self.histories],
            'Rooting': [epoch_metrics.get('val_Durchwurzelung_loss', float('nan')) for epoch_metrics in self.histories],
        }

        # Create a 3x2 subplot
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle('Training Losses for Subtasks', fontsize=16)

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
        
        # Plot the bisector line for stones predictions
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
    
    def _train_model(self, train_loader, device, model, optimizer):
        total_train_loss = 0.0
        train_stones_loss, train_soiltype_loss, train_soilcolor_loss, train_carbonate_loss, train_humus_loss, train_rooting_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        all_topk_soiltype_predictions = []
        all_topk_soilcolor_predictions = []
        all_topk_carbonate_predictions = []
        all_topk_humus_predictions = []
        all_topk_rooting_predictions = []
        
        all_soiltype_labels = []
        all_soilcolor_labels = []
        all_carbonate_labels = []
        all_humus_labels = []
        all_rooting_labels = []
        
        self.stones_predictions["train"] = []
        self.stones_true_values["train"] = []
        
        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False, unit="batch")
        for batch in train_loader_tqdm:
            _, segments, padded_segments_tabulars_labels, geotemp_features, horizon_indices = batch # full image not needed
            segments, padded_segments_tabulars_labels, geotemp_features = segments.to(device), padded_segments_tabulars_labels.to(device), geotemp_features.to(device)
            
            # Mask for valid indices
            mask = horizon_indices != -1
            
            optimizer.zero_grad()
            
            padded_pred_tabulars = model(segments, geotemp_features)
            tabular_predictions = {key: value[mask] for key, value in padded_pred_tabulars.items()}
            
            stones_predictions = tabular_predictions['Steine']
            soiltype_predictions = tabular_predictions['Bodenart']
            soilcolor_predictions = tabular_predictions['Bodenfarbe']
            carbonate_predictions = tabular_predictions['Karbonat']
            humus_predictions = tabular_predictions['Humusgehaltsklasse']
            rooting_predictions = tabular_predictions['Durchwurzelung']
            
            # Retrieve true labels
            start_idx = 0
            tabular_sliced = {}
            dict_lengths = { 'Steine' : 1 }
            dict_lengths.update(self.segments_tabular_categ_feature_columns)
            for tab_col, tab_col_dim in dict_lengths.items():
                end_idx = start_idx + tab_col_dim
                tabular_sliced[tab_col] = padded_segments_tabulars_labels[:, :, start_idx:end_idx][mask]
                start_idx = end_idx
                
            stones_true = tabular_sliced['Steine']
            soiltype_labels = torch.argmax(tabular_sliced['Bodenart'], dim=1)
            soilcolor_labels = torch.argmax(tabular_sliced['Bodenfarbe'], dim=1)
            carbonate_labels = torch.argmax(tabular_sliced['Karbonat'], dim=1)
            humus_labels = torch.argmax(tabular_sliced['Humusgehaltsklasse'], dim=1)
            rooting_labels = torch.argmax(tabular_sliced['Durchwurzelung'], dim=1)
            
            # Calculate losses
            stones_loss = self.mse_loss(stones_predictions, stones_true)
            soiltype_loss = self.cross_entropy_loss(soiltype_predictions, soiltype_labels)
            soilcolor_loss = self.cross_entropy_loss(soilcolor_predictions, soilcolor_labels)
            carbonate_loss = self.cross_entropy_loss(carbonate_predictions, carbonate_labels)
            humus_loss = self.cross_entropy_loss(humus_predictions, humus_labels)
            rooting_loss = self.cross_entropy_loss(rooting_predictions, rooting_labels)
            total_loss = stones_loss + soiltype_loss + soilcolor_loss + carbonate_loss + humus_loss + rooting_loss
            
            # Backpropagation
            total_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update running losses
            total_train_loss += total_loss.item()
            train_stones_loss += stones_loss.item()
            train_soiltype_loss += soiltype_loss.item()
            train_soilcolor_loss += soilcolor_loss.item()
            train_carbonate_loss += carbonate_loss.item()
            train_humus_loss += humus_loss.item()
            train_rooting_loss += rooting_loss.item()
            
            # Top-K predictions
            horizon_indices, topk_soiltype_predictions = torch.topk(soiltype_predictions, self.tab_topk)
            horizon_indices, topk_soilcolor_predictions = torch.topk(soilcolor_predictions, self.tab_topk)
            horizon_indices, topk_carbonate_predictions = torch.topk(carbonate_predictions, self.tab_topk)
            horizon_indices, topk_humus_predictions = torch.topk(humus_predictions, self.tab_topk)
            horizon_indices, topk_rooting_predictions = torch.topk(rooting_predictions, self.tab_topk)
            
            # Add predictions and true values to lists
            self.stones_predictions["train"].append(stones_predictions.detach().cpu())
            self.stones_true_values["train"].append(stones_true.detach().cpu())
            
            all_topk_soiltype_predictions.append(topk_soiltype_predictions.detach().cpu())
            all_topk_soilcolor_predictions.append(topk_soilcolor_predictions.detach().cpu())
            all_topk_carbonate_predictions.append(topk_carbonate_predictions.detach().cpu())
            all_topk_humus_predictions.append(topk_humus_predictions.detach().cpu())
            all_topk_rooting_predictions.append(topk_rooting_predictions.detach().cpu())
            
            all_soiltype_labels.append(soiltype_labels.detach().cpu())
            all_soilcolor_labels.append(soilcolor_labels.detach().cpu())
            all_carbonate_labels.append(carbonate_labels.detach().cpu())
            all_humus_labels.append(humus_labels.detach().cpu())
            all_rooting_labels.append(rooting_labels.detach().cpu())
            
            train_loader_tqdm.set_postfix(loss=total_loss.item())
        
        # Average losses over the batches
        total_train_loss /= len(train_loader)
        train_stones_loss /= len(train_loader)
        train_soiltype_loss /= len(train_loader)
        train_soilcolor_loss /= len(train_loader)
        train_carbonate_loss /= len(train_loader)
        train_humus_loss /= len(train_loader)
        train_rooting_loss /= len(train_loader)
        
        # Flatten and concatenate predictions and true values
        self.stones_predictions["train"] = torch.cat(self.stones_predictions["train"]).numpy().flatten().tolist()
        self.stones_true_values["train"] = torch.cat(self.stones_true_values["train"]).numpy().flatten().tolist()
        
        # Concatenate all top-k predictions and true values
        all_topk_soiltype_predictions = torch.cat(all_topk_soiltype_predictions).numpy()
        all_topk_soilcolor_predictions = torch.cat(all_topk_soilcolor_predictions).numpy()
        all_topk_carbonate_predictions = torch.cat(all_topk_carbonate_predictions).numpy()
        all_topk_humus_predictions = torch.cat(all_topk_humus_predictions).numpy()
        all_topk_rooting_predictions = torch.cat(all_topk_rooting_predictions).numpy()
        all_soiltype_labels = torch.cat(all_soiltype_labels).numpy()
        all_soilcolor_labels = torch.cat(all_soilcolor_labels).numpy()
        all_carbonate_labels = torch.cat(all_carbonate_labels).numpy()
        all_humus_labels = torch.cat(all_humus_labels).numpy()
        all_rooting_labels = torch.cat(all_rooting_labels).numpy()
        
        # Get top-1 predictions
        top1_soiltype_predictions = all_topk_soiltype_predictions[:, 0]
        top1_soilcolor_predictions = all_topk_soilcolor_predictions[:, 0]
        top1_carbonate_predictions = all_topk_carbonate_predictions[:, 0]
        top1_humus_predictions = all_topk_humus_predictions[:, 0]
        top1_rooting_predictions = all_topk_rooting_predictions[:, 0]
        
        # Possible labels for sklearn metrics
        possible_soiltype_labels = list(range(self.tabulars_output_dim_dict['Bodenart']))
        possible_soilcolor_labels = list(range(self.tabulars_output_dim_dict['Bodenfarbe']))
        possible_carbonate_labels = list(range(self.tabulars_output_dim_dict['Karbonat']))
        possible_humus_labels = list(range(self.tabulars_output_dim_dict['Humusgehaltsklasse']))
        possible_rooting_labels = list(range(self.tabulars_output_dim_dict['Durchwurzelung']))
                
        # Calculate metrics
        train_stones_metrics = {
            'train_Steine_loss' : train_stones_loss
        }
        train_soiltype_metrics = {
            'train_Bodenart_loss' : train_soiltype_loss,
            'train_Bodenart_accuracy' : accuracy_score(all_soiltype_labels, top1_soiltype_predictions),
            'train_Bodenart_f1' : f1_score(all_soiltype_labels, top1_soiltype_predictions, labels=possible_soiltype_labels, average=self.class_average, zero_division=0),
            'train_Bodenart_precision' : precision_score(all_soiltype_labels, top1_soiltype_predictions, labels=possible_soiltype_labels, average=self.class_average, zero_division=0),
            'train_Bodenart_recall' : recall_score(all_soiltype_labels, top1_soiltype_predictions, labels=possible_soiltype_labels, average=self.class_average, zero_division=0),
            'train_Bodenart_top_k_accuracy' : top_k_accuracy_from_indices(all_soiltype_labels, all_topk_soiltype_predictions),
            'train_Bodenart_precision_at_k' : precision_recall_at_k(all_soiltype_labels, all_topk_soiltype_predictions, all_labels=possible_soiltype_labels, average=self.class_average)[0],
            'train_Bodenart_recall_at_k' : precision_recall_at_k(all_soiltype_labels, all_topk_soiltype_predictions, all_labels=possible_soiltype_labels, average=self.class_average)[1],
        }
        train_soilcolor_metrics = {
            'train_Bodenfarbe_loss' : train_soilcolor_loss,
            'train_Bodenfarbe_accuracy' : accuracy_score(all_soilcolor_labels, top1_soilcolor_predictions),
            'train_Bodenfarbe_f1' : f1_score(all_soilcolor_labels, top1_soilcolor_predictions, labels=possible_soilcolor_labels, average=self.class_average, zero_division=0),
            'train_Bodenfarbe_precision' : precision_score(all_soilcolor_labels, top1_soilcolor_predictions, labels=possible_soilcolor_labels, average=self.class_average, zero_division=0),
            'train_Bodenfarbe_recall' : recall_score(all_soilcolor_labels, top1_soilcolor_predictions, labels=possible_soilcolor_labels, average=self.class_average, zero_division=0),
            'train_Bodenfarbe_top_k_accuracy' : top_k_accuracy_from_indices(all_soilcolor_labels, all_topk_soilcolor_predictions),
            'train_Bodenfarbe_precision_at_k' : precision_recall_at_k(all_soilcolor_labels, all_topk_soilcolor_predictions, all_labels=possible_soilcolor_labels, average=self.class_average)[0],
            'train_Bodenfarbe_recall_at_k' : precision_recall_at_k(all_soilcolor_labels, all_topk_soilcolor_predictions, all_labels=possible_soilcolor_labels, average=self.class_average)[1],
        }
        train_carbonate_metrics = {
            'train_Karbonat_loss' : train_carbonate_loss,
            'train_Karbonat_accuracy' : accuracy_score(all_carbonate_labels, top1_carbonate_predictions),
            'train_Karbonat_f1' : f1_score(all_carbonate_labels, top1_carbonate_predictions, labels=possible_carbonate_labels, average=self.class_average, zero_division=0),
            'train_Karbonat_precision' : precision_score(all_carbonate_labels, top1_carbonate_predictions, labels=possible_carbonate_labels, average=self.class_average, zero_division=0),
            'train_Karbonat_recall' : recall_score(all_carbonate_labels, top1_carbonate_predictions, labels=possible_carbonate_labels, average=self.class_average, zero_division=0),
            'train_Karbonat_top_k_accuracy' : top_k_accuracy_from_indices(all_carbonate_labels, all_topk_carbonate_predictions),
            'train_Karbonat_precision_at_k' : precision_recall_at_k(all_carbonate_labels, all_topk_carbonate_predictions, all_labels=possible_carbonate_labels, average=self.class_average)[0],
            'train_Karbonat_recall_at_k' : precision_recall_at_k(all_carbonate_labels, all_topk_carbonate_predictions, all_labels=possible_carbonate_labels, average=self.class_average)[1],
        }
        train_humus_metrics = {
            'train_Humusgehaltsklasse_loss' : train_humus_loss,
            'train_Humusgehaltsklasse_accuracy' : accuracy_score(all_humus_labels, top1_humus_predictions),
            'train_Humusgehaltsklasse_f1' : f1_score(all_humus_labels, top1_humus_predictions, labels=possible_humus_labels, average=self.class_average, zero_division=0),
            'train_Humusgehaltsklasse_precision' : precision_score(all_humus_labels, top1_humus_predictions, labels=possible_humus_labels, average=self.class_average, zero_division=0),
            'train_Humusgehaltsklasse_recall' : recall_score(all_humus_labels, top1_humus_predictions, labels=possible_humus_labels, average=self.class_average, zero_division=0),
            'train_Humusgehaltsklasse_top_k_accuracy' : top_k_accuracy_from_indices(all_humus_labels, all_topk_humus_predictions),
            'train_Humusgehaltsklasse_precision_at_k' : precision_recall_at_k(all_humus_labels, all_topk_humus_predictions, all_labels=possible_humus_labels, average=self.class_average)[0],
            'train_Humusgehaltsklasse_recall_at_k' : precision_recall_at_k(all_humus_labels, all_topk_humus_predictions, all_labels=possible_humus_labels, average=self.class_average)[1],
        }
        train_rooting_metrics = {
            'train_Durchwurzelung_loss' : train_rooting_loss,
            'train_Durchwurzelung_accuracy' : accuracy_score(all_rooting_labels, top1_rooting_predictions),
            'train_Durchwurzelung_f1' : f1_score(all_rooting_labels, top1_rooting_predictions, labels=possible_rooting_labels, average=self.class_average, zero_division=0),
            'train_Durchwurzelung_precision' : precision_score(all_rooting_labels, top1_rooting_predictions, labels=possible_rooting_labels, average=self.class_average, zero_division=0),
            'train_Durchwurzelung_recall' : recall_score(all_rooting_labels, top1_rooting_predictions, labels=possible_rooting_labels, average=self.class_average, zero_division=0),
            'train_Durchwurzelung_top_k_accuracy' : top_k_accuracy_from_indices(all_rooting_labels, all_topk_rooting_predictions),
            'train_Durchwurzelung_precision_at_k' : precision_recall_at_k(all_rooting_labels, all_topk_rooting_predictions, all_labels=possible_rooting_labels, average=self.class_average)[0],
            'train_Durchwurzelung_recall_at_k' : precision_recall_at_k(all_rooting_labels, all_topk_rooting_predictions, all_labels=possible_rooting_labels, average=self.class_average)[1],
        }
        
        return total_train_loss, train_stones_metrics, train_soiltype_metrics, train_soilcolor_metrics, train_carbonate_metrics, train_humus_metrics, train_rooting_metrics

    def _evaluate_model(self, eval_loader, device, model, mode='val'):
        total_eval_loss = 0.0
        eval_stones_loss, eval_soiltype_loss, eval_soilcolor_loss, eval_carbonate_loss, eval_humus_loss, eval_rooting_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        all_topk_soiltype_predictions = []
        all_topk_soilcolor_predictions = []
        all_topk_carbonate_predictions = []
        all_topk_humus_predictions = []
        all_topk_rooting_predictions = []

        all_soiltype_labels = []
        all_soilcolor_labels = []
        all_carbonate_labels = []
        all_humus_labels = []
        all_rooting_labels = []
        
        self.stones_predictions[mode] = []
        self.stones_true_values[mode] = []
        
        eval_loader_tqdm = tqdm(eval_loader, desc=f"{mode.capitalize()} Evaluation", leave=False, unit="batch")
        for batch in eval_loader_tqdm:
            _, segments, padded_segments_tabulars_labels, geotemp_features, horizon_indices = batch # full image not needed
            segments, padded_segments_tabulars_labels, geotemp_features = segments.to(device), padded_segments_tabulars_labels.to(device), geotemp_features.to(device)

            # Mask for valid indices
            mask = horizon_indices != -1

            with torch.no_grad():
                padded_tabular_predictions = model(segments, geotemp_features)
                tabular_predictions = {key: value[mask] for key, value in padded_tabular_predictions.items()}

                stones_predictions = tabular_predictions['Steine']
                soiltype_predictions = tabular_predictions['Bodenart']
                soilcolor_predictions = tabular_predictions['Bodenfarbe']
                carbonate_predictions = tabular_predictions['Karbonat']
                humus_predictions = tabular_predictions['Humusgehaltsklasse']
                rooting_predictions = tabular_predictions['Durchwurzelung']

                # Retrieve true labels
                start_idx = 0
                tabular_sliced = {}
                dict_lengths = { 'Steine' : 1}
                dict_lengths.update(self.segments_tabular_categ_feature_columns)
                for tab_col, tab_col_dim in dict_lengths.items():
                    end_idx = start_idx + tab_col_dim
                    tabular_sliced[tab_col] = padded_segments_tabulars_labels[:, :, start_idx:end_idx][mask]
                    start_idx = end_idx

                stones_true = tabular_sliced['Steine']
                soiltype_labels = torch.argmax(tabular_sliced['Bodenart'], dim=1)
                soilcolor_labels = torch.argmax(tabular_sliced['Bodenfarbe'], dim=1)
                carbonate_labels = torch.argmax(tabular_sliced['Karbonat'], dim=1)
                humus_labels = torch.argmax(tabular_sliced['Humusgehaltsklasse'], dim=1)
                rooting_labels = torch.argmax(tabular_sliced['Durchwurzelung'], dim=1)

                # Calculate losses
                stones_loss = self.mse_loss(stones_predictions, stones_true)
                soiltype_loss = self.cross_entropy_loss(soiltype_predictions, soiltype_labels)
                soilcolor_loss = self.cross_entropy_loss(soilcolor_predictions, soilcolor_labels)
                carbonate_loss = self.cross_entropy_loss(carbonate_predictions, carbonate_labels)
                humus_loss = self.cross_entropy_loss(humus_predictions, humus_labels)
                rooting_loss = self.cross_entropy_loss(rooting_predictions, rooting_labels)
                total_loss = stones_loss + soiltype_loss + soilcolor_loss + carbonate_loss + humus_loss + rooting_loss
                
                # Update running losses
                total_eval_loss += total_loss.item()
                eval_stones_loss += stones_loss.item()
                eval_soiltype_loss += soiltype_loss.item()
                eval_soilcolor_loss += soilcolor_loss.item()
                eval_carbonate_loss += carbonate_loss.item()
                eval_humus_loss += humus_loss.item()
                eval_rooting_loss += rooting_loss.item()

                # Top-K predictions
                horizon_indices, topk_soiltype_predictions = torch.topk(soiltype_predictions, self.tab_topk)
                horizon_indices, topk_soilcolor_predictions = torch.topk(soilcolor_predictions, self.tab_topk)
                horizon_indices, topk_carbonate_predictions = torch.topk(carbonate_predictions, self.tab_topk)
                horizon_indices, topk_humus_predictions = torch.topk(humus_predictions, self.tab_topk)
                horizon_indices, topk_rooting_predictions = torch.topk(rooting_predictions, self.tab_topk)

                # Add predictions and true values to lists
                self.stones_predictions[mode].append(stones_predictions.detach().cpu())
                self.stones_true_values[mode].append(stones_true.detach().cpu())
                
                # Add predictions and true values to lists
                all_topk_soiltype_predictions.append(topk_soiltype_predictions.detach().cpu())
                all_topk_soilcolor_predictions.append(topk_soilcolor_predictions.detach().cpu())
                all_topk_carbonate_predictions.append(topk_carbonate_predictions.detach().cpu())
                all_topk_humus_predictions.append(topk_humus_predictions.detach().cpu())
                all_topk_rooting_predictions.append(topk_rooting_predictions.detach().cpu())

                all_soiltype_labels.append(soiltype_labels.detach().cpu())
                all_soilcolor_labels.append(soilcolor_labels.detach().cpu())
                all_carbonate_labels.append(carbonate_labels.detach().cpu())
                all_humus_labels.append(humus_labels.detach().cpu())
                all_rooting_labels.append(rooting_labels.detach().cpu())

                eval_loader_tqdm.set_postfix(loss=total_loss.item())

        # Average losses over the batches
        total_eval_loss /= len(eval_loader)
        eval_stones_loss /= len(eval_loader)
        eval_soiltype_loss /= len(eval_loader)
        eval_soilcolor_loss /= len(eval_loader)
        eval_carbonate_loss /= len(eval_loader)
        eval_humus_loss /= len(eval_loader)
        eval_rooting_loss /= len(eval_loader)

        # Concatenate and change dtype of predictions and true values
        self.stones_predictions[mode] = torch.cat(self.stones_predictions[mode]).numpy().flatten().tolist()
        self.stones_true_values[mode] = torch.cat(self.stones_true_values[mode]).numpy().flatten().tolist()
        
        # Concatenate all top-k predictions and true values
        all_topk_soiltype_predictions = torch.cat(all_topk_soiltype_predictions).numpy()
        all_topk_soilcolor_predictions = torch.cat(all_topk_soilcolor_predictions).numpy()
        all_topk_carbonate_predictions = torch.cat(all_topk_carbonate_predictions).numpy()
        all_topk_humus_predictions = torch.cat(all_topk_humus_predictions).numpy()
        all_topk_rooting_predictions = torch.cat(all_topk_rooting_predictions).numpy()
        all_soiltype_labels = torch.cat(all_soiltype_labels).numpy()
        all_soilcolor_labels = torch.cat(all_soilcolor_labels).numpy()
        all_carbonate_labels = torch.cat(all_carbonate_labels).numpy()
        all_humus_labels = torch.cat(all_humus_labels).numpy()
        all_rooting_labels = torch.cat(all_rooting_labels).numpy()

        # Get top-1 predictions
        top1_soiltype_predictions = all_topk_soiltype_predictions[:, 0]
        top1_soilcolor_predictions = all_topk_soilcolor_predictions[:, 0]
        top1_carbonate_predictions = all_topk_carbonate_predictions[:, 0]
        top1_humus_predictions = all_topk_humus_predictions[:, 0]
        top1_rooting_predictions = all_topk_rooting_predictions[:, 0]

        # Possible labels for sklearn metrics
        possible_soiltype_labels = list(range(self.tabulars_output_dim_dict['Bodenart']))
        possible_soilcolor_labels = list(range(self.tabulars_output_dim_dict['Bodenfarbe']))
        possible_carbonate_labels = list(range(self.tabulars_output_dim_dict['Karbonat']))
        possible_humus_labels = list(range(self.tabulars_output_dim_dict['Humusgehaltsklasse']))
        possible_rooting_labels = list(range(self.tabulars_output_dim_dict['Durchwurzelung']))

        # Calculate metrics
        eval_stones_metrics = {
            f'{mode}_Steine_loss': eval_stones_loss
        }
        eval_soiltype_metrics = {
            f'{mode}_Bodenart_loss': eval_soiltype_loss,
            f'{mode}_Bodenart_accuracy': accuracy_score(all_soiltype_labels, top1_soiltype_predictions),
            f'{mode}_Bodenart_f1': f1_score(all_soiltype_labels, top1_soiltype_predictions, labels=possible_soiltype_labels, average=self.class_average, zero_division=0),
            f'{mode}_Bodenart_precision': precision_score(all_soiltype_labels, top1_soiltype_predictions, labels=possible_soiltype_labels, average=self.class_average, zero_division=0),
            f'{mode}_Bodenart_recall': recall_score(all_soiltype_labels, top1_soiltype_predictions, labels=possible_soiltype_labels, average=self.class_average, zero_division=0),
            f'{mode}_Bodenart_top_k_accuracy': top_k_accuracy_from_indices(all_soiltype_labels, all_topk_soiltype_predictions),
            f'{mode}_Bodenart_precision_at_k': precision_recall_at_k(all_soiltype_labels, all_topk_soiltype_predictions, all_labels=possible_soiltype_labels, average=self.class_average)[0],
            f'{mode}_Bodenart_recall_at_k': precision_recall_at_k(all_soiltype_labels, all_topk_soiltype_predictions, all_labels=possible_soiltype_labels, average=self.class_average)[1],
        }
        eval_soilcolor_metrics = {
            f'{mode}_Bodenfarbe_loss': eval_soilcolor_loss,
            f'{mode}_Bodenfarbe_accuracy': accuracy_score(all_soilcolor_labels, top1_soilcolor_predictions),
            f'{mode}_Bodenfarbe_f1': f1_score(all_soilcolor_labels, top1_soilcolor_predictions, labels=possible_soilcolor_labels, average=self.class_average, zero_division=0),
            f'{mode}_Bodenfarbe_precision': precision_score(all_soilcolor_labels, top1_soilcolor_predictions, labels=possible_soilcolor_labels, average=self.class_average, zero_division=0),
            f'{mode}_Bodenfarbe_recall': recall_score(all_soilcolor_labels, top1_soilcolor_predictions, labels=possible_soilcolor_labels, average=self.class_average, zero_division=0),
            f'{mode}_Bodenfarbe_top_k_accuracy': top_k_accuracy_from_indices(all_soilcolor_labels, all_topk_soilcolor_predictions),
            f'{mode}_Bodenfarbe_precision_at_k': precision_recall_at_k(all_soilcolor_labels, all_topk_soilcolor_predictions, all_labels=possible_soilcolor_labels, average=self.class_average)[0],
            f'{mode}_Bodenfarbe_recall_at_k': precision_recall_at_k(all_soilcolor_labels, all_topk_soilcolor_predictions, all_labels=possible_soilcolor_labels, average=self.class_average)[1],
        }
        eval_carbonate_metrics = {
            f'{mode}_Karbonat_loss': eval_carbonate_loss,
            f'{mode}_Karbonat_accuracy': accuracy_score(all_carbonate_labels, top1_carbonate_predictions),
            f'{mode}_Karbonat_f1': f1_score(all_carbonate_labels, top1_carbonate_predictions, labels=possible_carbonate_labels, average=self.class_average, zero_division=0),
            f'{mode}_Karbonat_precision': precision_score(all_carbonate_labels, top1_carbonate_predictions, labels=possible_carbonate_labels, average=self.class_average, zero_division=0),
            f'{mode}_Karbonat_recall': recall_score(all_carbonate_labels, top1_carbonate_predictions, labels=possible_carbonate_labels, average=self.class_average, zero_division=0),
            f'{mode}_Karbonat_top_k_accuracy': top_k_accuracy_from_indices(all_carbonate_labels, all_topk_carbonate_predictions),
            f'{mode}_Karbonat_precision_at_k': precision_recall_at_k(all_carbonate_labels, all_topk_carbonate_predictions, all_labels=possible_carbonate_labels, average=self.class_average)[0],
            f'{mode}_Karbonat_recall_at_k': precision_recall_at_k(all_carbonate_labels, all_topk_carbonate_predictions, all_labels=possible_carbonate_labels, average=self.class_average)[1],
        }
        eval_humus_metrics = {
            f'{mode}_Humusgehaltsklasse_loss': eval_humus_loss,
            f'{mode}_Humusgehaltsklasse_accuracy': accuracy_score(all_humus_labels, top1_humus_predictions),
            f'{mode}_Humusgehaltsklasse_f1': f1_score(all_humus_labels, top1_humus_predictions, labels=possible_humus_labels, average=self.class_average, zero_division=0),
            f'{mode}_Humusgehaltsklasse_precision': precision_score(all_humus_labels, top1_humus_predictions, labels=possible_humus_labels, average=self.class_average, zero_division=0),
            f'{mode}_Humusgehaltsklasse_recall': recall_score(all_humus_labels, top1_humus_predictions, labels=possible_humus_labels, average=self.class_average, zero_division=0),
            f'{mode}_Humusgehaltsklasse_top_k_accuracy': top_k_accuracy_from_indices(all_humus_labels, all_topk_humus_predictions),
            f'{mode}_Humusgehaltsklasse_precision_at_k': precision_recall_at_k(all_humus_labels, all_topk_humus_predictions, all_labels=possible_humus_labels, average=self.class_average)[0],
            f'{mode}_Humusgehaltsklasse_recall_at_k': precision_recall_at_k(all_humus_labels, all_topk_humus_predictions, all_labels=possible_humus_labels, average=self.class_average)[1],
        }
        eval_rooting_metrics = {
            f'{mode}_Durchwurzelung_loss': eval_rooting_loss,
            f'{mode}_Durchwurzelung_accuracy': accuracy_score(all_rooting_labels, top1_rooting_predictions),
            f'{mode}_Durchwurzelung_f1': f1_score(all_rooting_labels, top1_rooting_predictions, labels=possible_rooting_labels, average=self.class_average, zero_division=0),
            f'{mode}_Durchwurzelung_precision': precision_score(all_rooting_labels, top1_rooting_predictions, labels=possible_rooting_labels, average=self.class_average, zero_division=0),
            f'{mode}_Durchwurzelung_recall': recall_score(all_rooting_labels, top1_rooting_predictions, labels=possible_rooting_labels, average=self.class_average, zero_division=0),
            f'{mode}_Durchwurzelung_top_k_accuracy': top_k_accuracy_from_indices(all_rooting_labels, all_topk_rooting_predictions),
            f'{mode}_Durchwurzelung_precision_at_k': precision_recall_at_k(all_rooting_labels, all_topk_rooting_predictions, all_labels=possible_rooting_labels, average=self.class_average)[0],
            f'{mode}_Durchwurzelung_recall_at_k': precision_recall_at_k(all_rooting_labels, all_topk_rooting_predictions, all_labels=possible_rooting_labels, average=self.class_average)[1],
        }

        return total_eval_loss, eval_stones_metrics, eval_soiltype_metrics, eval_soilcolor_metrics, eval_carbonate_metrics, eval_humus_metrics, eval_rooting_metrics
    
    def _fancy_print(self, epoch_metrics, key_prefix=''):
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
            #'num_segment_patches' : 48, # only used with SegmentPatches dataset for ResNetPatch
            'segment_encoder_output_dim': 512,
            'geotemp_output_dim': 256,
            'patch_size': 512,
            'predictor_hidden_dim': 1024,
            'num_lstm_layers': 2
        }
