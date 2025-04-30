from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_ # modifies the tensors in-place (vs clip_grad_norm)
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bgr.soil.training_args import TrainingArgs

from bgr.soil.data.horizon_tabular_data import HorizonDataProcessor
from bgr.soil.experiments import Experiment
from bgr.soil.modelling.horizon.horizon_models import SimpleHorizonClassifierWithEmbeddingsGeotemps
from bgr.soil.metrics import TopKHorizonAccuracy, precision_recall_at_k
from bgr.soil.data.datasets import SegmentsTabularDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleHorizonClassificationEmbeddingsGeotemp(Experiment):
    def __init__(self, training_args: 'TrainingArgs', target: str, dataprocessor: HorizonDataProcessor):
        self.training_args = training_args
        self.target = target
        self.dataprocessor = dataprocessor
        self.trained = False
        
        self.label_embeddings_tensor = torch.tensor(self.dataprocessor.embeddings_dict['embedding'], device=self.training_args.device).float()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.topk = 5
        self.horizon_topk_acc = lambda k : TopKHorizonAccuracy(self.label_embeddings_tensor, k=k)
        self.f1_average = 'macro'
        self.image_normalization = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet statistics
        ])
        
        # Retrieve the experiment hyperparameters
        self.hyperparameters = SimpleHorizonClassificationEmbeddingsGeotemp.get_experiment_hyperparameters()
        self.hyperparameters.update(training_args.hyperparameters)
        
        # Initialize the labels and predictions dictionaries for confusion matrix
        self.possible_labels = list(range(self.label_embeddings_tensor.size(0)))
        self.labels = {'train': None, 'val': None, 'test': None}
        self.predictions = {'train': None, 'val': None, 'test': None}
    
    def train_and_validate(self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_output_dir: str
    ) -> tuple[nn.Module, dict]:
        
        train_dataset = SegmentsTabularDataset(
            dataframe=train_df,
            normalize=self.image_normalization,
            label_column=self.target,
            geotemp_columns=self.dataprocessor.geotemp_img_infos[:-1] # without 'file'
        )
        train_loader = DataLoader(train_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        val_dataset = SegmentsTabularDataset(
            dataframe=val_df,
            normalize=self.image_normalization,
            label_column=self.target,
            geotemp_columns=self.dataprocessor.geotemp_img_infos[:-1] # without 'file'
        )
        val_loader = DataLoader(val_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        model = self.get_model()
        model.to(self.training_args.device)
        
        lr = self.training_args.learning_rate
        weight_decay = self.training_args.weight_decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.1, min_lr=lr*0.01, verbose=True)
        
        self.train_loss_history, self.val_loss_history = [], []
        self.train_acc_history, self.val_acc_history = [], []
        self.train_topk_acc_history, self.val_topk_acc_history = [], []
        self.train_precision_history, self.val_precision_history = [], []
        self.train_recall_history, self.val_recall_history = [], []
        self.train_precision_at_k_history, self.val_precision_at_k_history = [], []
        self.train_recall_at_k_history, self.val_recall_at_k_history = [], []
        self.train_f1_score_history, self.val_f1_score_history = [], []

        for epoch in range(1, self.training_args.num_epochs + 1):
            print("--------------------------------")
            logger.info(f"Epoch {epoch}/{self.training_args.num_epochs}")
            
            # Training loop
            model.train()
            avg_train_loss, avg_train_acc, avg_train_topk_acc, train_precision, train_recall, train_precision_at_k, train_recall_at_k, train_f1_score = self._train_model(train_loader, self.training_args.device, model, optimizer)

            # Evaluation loop
            model.eval() # Set model in evaluation mode before running inference
            avg_val_loss, avg_val_acc, avg_val_topk_acc, val_precision, val_recall, val_precision_at_k, val_recall_at_k, val_f1_score = self._evaluate_model(val_loader, self.training_args.device, model)

            epoch_metrics = {
                'epoch' : epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': avg_train_acc,
                'val_acc': avg_val_acc,
                'train_topk_correct': avg_train_topk_acc,
                'val_topk_acc': avg_val_topk_acc,
                'train_precision': train_precision,
                'val_precision': val_precision,
                'train_recall': train_recall,
                'val_recall': val_recall,
                'train_precision_at_k': train_precision_at_k,
                'val_precision_at_k': val_precision_at_k,
                'train_recall_at_k': train_recall_at_k,
                'val_recall_at_k': val_recall_at_k,
                'train_f1_score': train_f1_score,
                'val_f1_score': val_f1_score
            }
            for callback in self.training_args.callbacks:
                callback(model, epoch_metrics, epoch)
            
            # Log metrics to wandb
            wandb.log(epoch_metrics)
            
            # Apply the scheduler with validation loss
            scheduler.step(avg_val_loss)
            # Log the current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Log metrics
            self.train_loss_history.append(avg_train_loss); self.val_loss_history.append(avg_val_loss)
            self.train_acc_history.append(avg_train_acc); self.val_acc_history.append(avg_val_acc)
            self.train_topk_acc_history.append(avg_train_topk_acc); self.val_topk_acc_history.append(avg_val_topk_acc)
            self.train_precision_history.append(train_precision); self.val_precision_history.append(val_precision)
            self.train_recall_history.append(train_recall); self.val_recall_history.append(val_recall)
            self.train_precision_at_k_history.append(train_precision_at_k); self.val_precision_at_k_history.append(val_precision_at_k)
            self.train_recall_at_k_history.append(train_recall_at_k); self.val_recall_at_k_history.append(val_recall_at_k)
            self.train_f1_score_history.append(train_f1_score); self.val_f1_score_history.append(val_f1_score)

            logger.info(
                f"""
                Epoch {epoch}/{self.training_args.num_epochs} Metrics:
                --------------------------------
                Training:
                - Cosine Loss: {avg_train_loss:.4f}
                - Accuracy: {avg_train_acc:.4f}
                - Top-{self.topk} Accuracy: {avg_train_topk_acc:.4f}
                - Precision: {train_precision:.4f}
                - Recall: {train_recall:.4f}
                - Precision@{self.topk}: {train_precision_at_k:.4f}
                - Recall@{self.topk}: {train_recall_at_k:.4f}
                - F1 Score: {train_f1_score:.4f}
                
                Validation:
                - Cosine Loss: {avg_val_loss:.4f}
                - Accuracy: {avg_val_acc:.4f}
                - Top-{self.topk} Accuracy: {avg_val_topk_acc:.4f}
                - Precision: {val_precision:.4f}
                - Recall: {val_recall:.4f}
                - Precision@{self.topk}: {val_precision_at_k:.4f}
                - Recall@{self.topk}: {val_recall_at_k:.4f}
                - F1 Score: {val_f1_score:.4f}
                
                Current Learning Rate: {current_lr}
                --------------------------------
                """
            )
            
            # Check early stopping
            if self.training_args.use_early_stopping:
                early_stopping = [cb for cb in self.training_args.callbacks if type(cb).__name__ == 'EarlyStopping'][0]
                if early_stopping.should_stop:
                    logger.info("Early stopping activated.")
                    break
        
        self.trained = True
        return_metrics = {
            'Train Cosine Loss' : self.train_loss_history[-1],
            'Validation Cosine Loss' : self.val_loss_history[-1],
            'Train Accuracy' : self.train_acc_history[-1],
            'Validation Accuracy' : self.val_acc_history[-1],
            'Train Top-5 Accuracy' : self.train_topk_acc_history[-1],
            'Validation Top-5 Accuracy' : self.val_topk_acc_history[-1],
            'Train Precision@5' : self.train_precision_at_k_history[-1],
            'Validation Precision@5' : self.val_precision_at_k_history[-1],
            'Train Recall@5' : self.train_recall_at_k_history[-1],
            'Validation Recall@5' : self.val_recall_at_k_history[-1],
            'Train F1 Score' : self.train_f1_score_history[-1],
            'Validation F1 Score' : self.val_f1_score_history[-1]
        }
        
        return model, return_metrics
    
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
            geotemp_columns=self.dataprocessor.geotemp_img_infos[:-1] # without 'file'
        )
        test_loader = DataLoader(test_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        model.to(self.training_args.device)
        
        print("--------------------------------")
        # Evaluation loop
        model.eval() # Set model in evaluation mode before running inference
        avg_test_loss, avg_test_accuracy, avg_test_topk_accuracy, test_precision, test_recall, avg_test_precision_at_k, avg_test_recall_at_k, test_f1_score = self._evaluate_model(test_loader, self.training_args.device, model, mode='test')
        
        test_metrics = {
            'Test Cosine Loss': avg_test_loss,
            'Test Accuracy': avg_test_accuracy,
            'Test Top-5 Accuracy': avg_test_topk_accuracy,
            'Test Precision': test_precision,
            'Test Recall': test_recall,
            'Test Precision@5': avg_test_precision_at_k,
            'Test Recall@5': avg_test_recall_at_k,
            'Test F1 Score': test_f1_score
        }
        
        logger.info(
                f"""
                Test Metrics:
                --------------------------------
                Testing:
                - Cosine Loss: {avg_test_loss:.4f}
                - Accuracy: {avg_test_accuracy:.4f}
                - Top-{self.topk} Accuracy: {avg_test_topk_accuracy:.4f}
                - Precision: {test_precision:.4f}
                - Recall: {test_recall:.4f}
                - Precision@{self.topk}: {avg_test_precision_at_k:.4f}
                - Recall@{self.topk}: {avg_test_recall_at_k:.4f}
                - F1 Score: {test_f1_score:.4f}
                --------------------------------
                """
            )
        
        return test_metrics
    
    def get_model(self) -> nn.Module:
        return SimpleHorizonClassifierWithEmbeddingsGeotemps(
            geo_temp_input_dim=len(self.dataprocessor.geotemp_img_infos) - 2, # without index and img path
            segment_encoder_output_dim=self.hyperparameters['segment_encoder_output_dim'],
            patch_size=self.hyperparameters['patch_size'],
            embedding_dim=np.shape(self.dataprocessor.embeddings_dict['embedding'])[1]
        )
    
    def plot_losses(self, model_output_dir: str, wandb_image_logging: bool) -> None:
        if not self.trained:
            raise ValueError("Model has not been trained yet.")
        
        complete_epochs = len(self.train_loss_history) + 1
        loss_histories = {
            'Cosine': (self.train_loss_history, self.val_loss_history)
        }
        acc_histories = {
            'Accuracy': (self.train_acc_history, self.val_acc_history)
        }
        topk_acc_histories = {
            f'Top-{self.topk} Accuracy': (self.train_topk_acc_history, self.val_topk_acc_history)
        }
        f1_score_histories = {
            'F1 Score': (self.train_f1_score_history, self.val_f1_score_history)
        }
        
        figure = plt.figure(figsize=(15, 7))
        for i, (title, (train_history, val_history)) in enumerate(loss_histories.items()):
            plt.subplot(2, 2, i + 1)
            plt.plot(range(1, complete_epochs), train_history, label=f'Train {title} Loss', marker='o', color='b')
            plt.plot(range(1, complete_epochs), val_history, label=f'Validation {title} Loss', marker='o', color='r')
            plt.title(f'{title} Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
            
        for i, (title, (train_history, val_history)) in enumerate(acc_histories.items()):
            plt.subplot(2, 2, i + 2)
            plt.plot(range(1, complete_epochs), train_history, label=f'Train {title}', marker='o', color='b')
            plt.plot(range(1, complete_epochs), val_history, label=f'Validation {title}', marker='o', color='r')
            plt.title(f'{title}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid()
        
        for i, (title, (train_history, val_history)) in enumerate(topk_acc_histories.items()):
            plt.subplot(2, 2, i + 3)
            plt.plot(range(1, complete_epochs), train_history, label=f'Train {title}', marker='o', color='b')
            plt.plot(range(1, complete_epochs), val_history, label=f'Validation {title}', marker='o', color='r')
            plt.title(f'{title}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid()
            
        for i, (title, (train_history, val_history)) in enumerate(f1_score_histories.items()):
            plt.subplot(2, 2, i + 4)
            plt.plot(range(1, complete_epochs), train_history, label=f'Train {title}', marker='o', color='b')
            plt.plot(range(1, complete_epochs), val_history, label=f'Validation {title}', marker='o', color='r')
            plt.title(f'{title}')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.grid()
        
        plt.tight_layout()
        
        plt.savefig(f'{model_output_dir}/losses_and_accuracies.pdf', bbox_inches='tight', format='pdf')
        if wandb_image_logging:
            wandb.log({"Losses and Accuracies": wandb.Image(figure)})
        
        # Plot confusion matrices
        self._plot_confusion_matrices(labels=self.labels['train'], predictions=self.predictions['train'], emb_dict=self.dataprocessor.embeddings_dict, model_output_dir=model_output_dir, wandb_image_logging=wandb_image_logging, mode='train')
        self._plot_confusion_matrices(labels=self.labels['val'], predictions=self.predictions['val'], emb_dict=self.dataprocessor.embeddings_dict, model_output_dir=model_output_dir, wandb_image_logging=wandb_image_logging, mode='val')
        self._plot_confusion_matrices(labels=self.labels['test'], predictions=self.predictions['test'], emb_dict=self.dataprocessor.embeddings_dict, model_output_dir=model_output_dir, wandb_image_logging=wandb_image_logging, mode='test')
            
    def _train_model(self, train_loader, device, model, optimizer):
        train_loss_total = 0.0
        train_correct = 0
        train_topk_correct = 0
        
        all_topk_predictions = []
        all_labels = []
        
        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_loader_tqdm:
            _, segments, _, geotemp_features, padded_true_horizon_indices = batch # full image and tabular_features not needed
            segments, geotemp_features, padded_true_horizon_indices = segments.to(device), geotemp_features.to(device), padded_true_horizon_indices.to(device)

            optimizer.zero_grad() # otherwise, PyTorch accumulates the gradients during backprop

            # Predict depth markers (as padded tensors)
            padded_pred_horizon_embeddings = model(segments=segments, geo_temp_features=geotemp_features[:, 1:]) # 'index' column not used in model
                
            true_horizon_embeddings = torch.stack([torch.tensor(self.dataprocessor.embeddings_dict['embedding'][lab.item()]) for lab in padded_true_horizon_indices.view(-1) if lab != -1]).to(device)
            pred_horizon_embeddings = torch.stack([pred for pred, lab in zip(padded_pred_horizon_embeddings.view(-1, padded_pred_horizon_embeddings.size(-1)), padded_true_horizon_indices.view(-1)) if lab != -1]).to(device)
            true_horizon_indices = padded_true_horizon_indices.view(-1)[padded_true_horizon_indices.view(-1) != -1]
            pred_topk_horizon_indices = torch.topk(torch.matmul(pred_horizon_embeddings, self.label_embeddings_tensor.T), k=self.topk, dim=1).indices
                
            # Normalize embeddings for the cosine loss, true embeddings are already normalized
            pred_horizon_embeddings = F.normalize(pred_horizon_embeddings, p=2, dim=1)
                
            # Compute individual losses, then sum them together for backprop
            # Create a dummy "same class" tensor with 1s for the cosine similarity
            same_class = torch.ones(pred_horizon_embeddings.size(0)).to(device)
            train_loss = self.cosine_loss(pred_horizon_embeddings, true_horizon_embeddings, same_class)
            train_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Calculate batch losses to total loss
            train_loss_total += train_loss.item()
            train_correct += self.horizon_topk_acc(1)(pred_horizon_embeddings, true_horizon_indices)
            train_topk_correct += self.horizon_topk_acc(self.topk)(pred_horizon_embeddings, true_horizon_indices)
            
            # Append predictions and labels for F1 score
            all_topk_predictions.append(pred_topk_horizon_indices.cpu())
            all_labels.append(true_horizon_indices.cpu())

            train_loader_tqdm.set_postfix(loss=train_loss.item())

        # Average losses over the batches
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_acc = train_correct / len(train_loader)
        avg_train_topk_acc = train_topk_correct / len(train_loader)
        
        # Calculate precision@k, recall@k and F1 score
        labels = torch.cat(all_labels)
        top1_predictions = torch.cat(all_topk_predictions)[:, 0]
        topk_predictions = torch.cat(all_topk_predictions)
        
        train_precision = precision_score(labels.numpy(), top1_predictions.numpy(), labels=self.possible_labels, average=self.f1_average, zero_division=0)
        train_recall = recall_score(labels.numpy(), top1_predictions.numpy(), labels=self.possible_labels, average=self.f1_average, zero_division=0)
        train_precision_at_k, train_recall_at_k = precision_recall_at_k(labels, topk_predictions, all_labels=self.possible_labels, average=self.f1_average)
        train_f1_score = f1_score(labels.numpy(), top1_predictions.numpy(), labels=self.possible_labels, average=self.f1_average, zero_division=0)
        
        # Store labels and predictions for confusion matrix
        self.labels['train'] = labels.numpy()
        self.predictions['train'] = top1_predictions.numpy()
        
        return \
            avg_train_loss, \
            avg_train_acc, \
            avg_train_topk_acc, \
            train_precision, \
            train_recall, \
            train_precision_at_k, \
            train_recall_at_k, \
            train_f1_score

    def _evaluate_model(self, eval_loader, device, model, mode='val'):
        eval_loss_total = 0.0
        eval_correct = 0
        eval_topk_correct = 0
        
        all_topk_predictions = []
        all_labels = []
        
        eval_loader_tqdm = tqdm(eval_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in eval_loader_tqdm:
                _, segments, _, geotemp_features, padded_true_horizon_indices = batch # full image and tabular_features not needed
                segments, geotemp_features, padded_true_horizon_indices = segments.to(device), geotemp_features.to(device), padded_true_horizon_indices.to(device)

                # Predict depth markers (as padded tensors)
                padded_pred_horizon_embeddings = model(segments=segments, geo_temp_features=geotemp_features[:, 1:]) # 'index' column not used in model
                    
                true_horizon_embeddings = torch.stack([torch.tensor(self.dataprocessor.embeddings_dict['embedding'][lab.item()]) for lab in padded_true_horizon_indices.view(-1) if lab != -1]).to(device)
                pred_horizon_embeddings = torch.stack([pred for pred, lab in zip(padded_pred_horizon_embeddings.view(-1, padded_pred_horizon_embeddings.size(-1)), padded_true_horizon_indices.view(-1)) if lab != -1]).to(device)
                true_horizon_indices = padded_true_horizon_indices.view(-1)[padded_true_horizon_indices.view(-1) != -1]
                pred_topk_horizon_indices = torch.topk(torch.matmul(pred_horizon_embeddings, self.label_embeddings_tensor.T), k=self.topk, dim=1).indices
                    
                # Normalize embeddings for the cosine loss, true embeddings are already normalized
                pred_horizon_embeddings = F.normalize(pred_horizon_embeddings, p=2, dim=1)
                    
                # Compute batch losses
                # Create a dummy "same class" tensor with 1s for the cosine similarity
                same_class = torch.ones(pred_horizon_embeddings.size(0)).to(device)
                val_loss = self.cosine_loss(pred_horizon_embeddings, true_horizon_embeddings, same_class)

                # Add batch losses to total loss
                eval_loss_total += val_loss.item()
                eval_correct += self.horizon_topk_acc(1)(pred_horizon_embeddings, true_horizon_indices)
                eval_topk_correct += self.horizon_topk_acc(self.topk)(pred_horizon_embeddings, true_horizon_indices)
                
                # Append predictions and labels for F1 score
                all_topk_predictions.append(pred_topk_horizon_indices.cpu())
                all_labels.append(true_horizon_indices.cpu())
            
            # Average losses over the batches
            avg_eval_loss = eval_loss_total / len(eval_loader)
            avg_eval_acc = eval_correct / len(eval_loader)
            avg_eval_topk_acc = eval_topk_correct / len(eval_loader)
            
            # Calculate precision@k, recall@k and F1 score
            labels = torch.cat(all_labels)
            top1_predictions = torch.cat(all_topk_predictions)[:, 0]
            topk_predictions = torch.cat(all_topk_predictions)
            
            eval_precision = precision_score(labels.numpy(), top1_predictions.numpy(), labels=self.possible_labels, average=self.f1_average, zero_division=0)
            eval_recall = recall_score(labels.numpy(), top1_predictions.numpy(), labels=self.possible_labels, average=self.f1_average, zero_division=0)
            eval_precision_at_k, eval_recall_at_k = precision_recall_at_k(labels, topk_predictions, all_labels=self.possible_labels, average=self.f1_average)
            eval_f1_score = f1_score(labels.numpy(), top1_predictions.numpy(), labels=self.possible_labels, average=self.f1_average, zero_division=0)
            
            # Store labels and predictions for confusion matrix
            self.labels[mode] = labels.numpy()
            self.predictions[mode] = top1_predictions.numpy()
            
        return \
            avg_eval_loss, \
            avg_eval_acc, \
            avg_eval_topk_acc, \
            eval_precision, \
            eval_recall, \
            eval_precision_at_k, \
            eval_recall_at_k, \
            eval_f1_score
    
    @staticmethod
    def get_experiment_hyperparameters():
        return {
            'segment_encoder_output_dim' : 512,
            'patch_size' : 512
        }