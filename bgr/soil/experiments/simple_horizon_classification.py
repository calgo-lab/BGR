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

from bgr.soil.data.horizon_tabular_data import HorizonDataProcessor
from bgr.soil.experiments import Experiment
from bgr.soil.training_args import TrainingArgs
from bgr.soil.modelling.general_models import SimpleHorizonClassifier
from bgr.soil.metrics import TopKHorizonAccuracy
from bgr.soil.data.datasets import SegmentsTabularDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleHorizonClassificationExperiment(Experiment):
    def __init__(self, training_args: TrainingArgs, target: str, dataprocessor: HorizonDataProcessor):
        self.training_args = training_args
        self.target = target
        self.dataprocessor = dataprocessor
        self.trained = False
        
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.topk = 5
        self.horizon_topk_acc = lambda k : TopKHorizonAccuracy(torch.tensor(self.dataprocessor.embeddings_dict['embedding'], device=self.training_args.device).float(), k=k)
        self.image_normalization = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet statistics
        ])
    
    def train_and_validate(self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_output_dir: str
    ) -> tuple[nn.Module, dict]:
        
        train_dataset = SegmentsTabularDataset(
            dataframe=train_df,
            normalize=self.image_normalization,
            label_column='Horizontsymbol_relevant',
            feature_columns=self.dataprocessor.geotemp_img_infos[:-1] # without 'file'
        )
        train_loader = DataLoader(train_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        val_dataset = SegmentsTabularDataset(
            dataframe=val_df,
            normalize=self.image_normalization,
            label_column='Horizontsymbol_relevant',
            feature_columns=self.dataprocessor.geotemp_img_infos[:-1] # without 'file'
        )
        val_loader = DataLoader(val_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        device = self.training_args.device
        
        model = self.get_model()
        model.to(device)
        
        lr = self.training_args.learning_rate
        weight_decay = self.training_args.weight_decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.1, min_lr=lr*0.01, verbose=True)
        
        self.train_loss_history, self.val_loss_history = [], []
        self.train_acc_history, self.val_acc_history = [], []
        self.train_topk_acc_history, self.val_topk_acc_history = [], []

        for epoch in range(self.training_args.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.training_args.num_epochs}")
            
            # Training
            model.train()
            train_loss_total = 0.0
            train_correct = 0
            train_topk_correct = 0
            train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
            for batch in train_loader_tqdm:
                segments, geotemp_features, padded_true_horizon_indices = batch
                segments, geotemp_features, padded_true_horizon_indices = segments.to(device), geotemp_features.to(device), padded_true_horizon_indices.to(device)

                optimizer.zero_grad() # otherwise, PyTorch accumulates the gradients during backprop

                # Predict depth markers (as padded tensors)
                padded_pred_horizon_embeddings = model(segments=segments, geo_temp_features=geotemp_features[:, 1:]) # 'index' column not used in model
                
                true_horizon_embeddings = torch.stack([torch.tensor(self.dataprocessor.embeddings_dict['embedding'][lab.item()]) for lab in padded_true_horizon_indices.view(-1) if lab != -1]).to(device)
                pred_horizon_embeddings = torch.stack([pred for pred, lab in zip(padded_pred_horizon_embeddings.view(-1, padded_pred_horizon_embeddings.size(-1)), padded_true_horizon_indices.view(-1)) if lab != -1]).to(device)
                true_horizon_indices = padded_true_horizon_indices.view(-1)[padded_true_horizon_indices.view(-1) != -1]
                
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

                train_loader_tqdm.set_postfix(loss=train_loss.item())

            # Average losses and iou at the end of the epoch
            avg_train_loss = train_loss_total / len(train_loader)
            avg_train_acc = train_correct / len(train_loader)
            avg_train_topk_acc = train_topk_correct / len(train_loader)

            # Evaluation loop
            model.eval() # Set model in evaluation mode before running inference
            val_loss_total = 0.0
            val_correct = 0
            val_topk_correct = 0
            val_loader_tqdm = tqdm(val_loader, desc="Evaluating", leave=False)
            with torch.no_grad():
                for batch in val_loader_tqdm:
                    segments, geotemp_features, padded_true_horizon_indices = batch
                    segments, geotemp_features, padded_true_horizon_indices = segments.to(device), geotemp_features.to(device), padded_true_horizon_indices.to(device)

                    # Predict depth markers (as padded tensors)
                    padded_pred_horizon_embeddings = model(segments=segments, geo_temp_features=geotemp_features[:, 1:]) # 'index' column not used in model
                    
                    true_horizon_embeddings = torch.stack([torch.tensor(self.dataprocessor.embeddings_dict['embedding'][lab.item()]) for lab in padded_true_horizon_indices.view(-1) if lab != -1]).to(device)
                    pred_horizon_embeddings = torch.stack([pred for pred, lab in zip(padded_pred_horizon_embeddings.view(-1, padded_pred_horizon_embeddings.size(-1)), padded_true_horizon_indices.view(-1)) if lab != -1]).to(device)
                    true_horizon_indices = padded_true_horizon_indices.view(-1)[padded_true_horizon_indices.view(-1) != -1]
                    
                    # Normalize embeddings for the cosine loss, true embeddings are already normalized
                    pred_horizon_embeddings = F.normalize(pred_horizon_embeddings, p=2, dim=1)
                    
                    # Compute batch losses
                    # Create a dummy "same class" tensor with 1s for the cosine similarity
                    same_class = torch.ones(pred_horizon_embeddings.size(0)).to(device)
                    val_loss = self.cosine_loss(pred_horizon_embeddings, true_horizon_embeddings, same_class)

                    # Add batch losses to total loss
                    val_loss_total += val_loss.item()
                    val_correct += self.horizon_topk_acc(1)(pred_horizon_embeddings, true_horizon_indices)
                    val_topk_correct += self.horizon_topk_acc(self.topk)(pred_horizon_embeddings, true_horizon_indices)

            # Average losses and iou at the end of the epoch
            avg_val_loss = val_loss_total / len(val_loader)
            avg_val_acc = val_correct / len(val_loader)
            avg_val_topk_acc = val_topk_correct / len(val_loader)

            epoch_metrics = {
                'train_cosine_loss': avg_train_loss,
                'val_cosine_loss': avg_val_loss,
                'train_acc': avg_train_acc,
                'val_acc': avg_val_acc,
                'train_topk_correct': avg_train_topk_acc,
                'val_topk_acc': avg_val_topk_acc
            }
            for callback in self.training_args.callbacks:
                callback(model, epoch_metrics, epoch)
            
            # Apply the scheduler with validation loss
            scheduler.step(avg_val_loss)
            # Log the current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Log metrics
            self.train_loss_history.append(avg_train_loss); self.val_loss_history.append(avg_val_loss)
            self.train_acc_history.append(avg_train_acc); self.val_acc_history.append(avg_val_acc)
            self.train_topk_acc_history.append(avg_train_topk_acc); self.val_topk_acc_history.append(avg_val_topk_acc)

            logger.info(f"Epoch {epoch+1}, Total Training Cosine Loss: {avg_train_loss:.4f}, Training Acc: {avg_train_acc:.4f}, Training Top-{self.topk} Acc: {avg_train_topk_acc:.4f}")
            logger.info(f"\nTotal Validation Cosine Loss: {avg_val_loss:.4f}, Validation Acc: {avg_val_acc:.4f}, Validation Top-{self.topk} Acc: {avg_val_topk_acc:.4f}")
            logger.info(f"Current LR: {current_lr}")
            
            # Check early stopping
            if self.training_args.use_early_stopping:
                early_stopping = [cb for cb in self.training_args.callbacks if type(cb).__name__ == 'EarlyStopping'][0]
                if early_stopping.should_stop:
                    logger.info("Early stopping activated.")
                    break
        
        self.trained = True
        return model, epoch_metrics
    
    def test(self,
        model: nn.Module,
        test_df: pd.DataFrame,
        model_output_dir: str
    ) -> dict:
        
        test_dataset = SegmentsTabularDataset(
            dataframe=test_df,
            normalize=self.image_normalization,
            label_column='Horizontsymbol_relevant',
            feature_columns=self.dataprocessor.geotemp_img_infos[:-1] # without 'file'
        )
        test_loader = DataLoader(test_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        device = self.training_args.device
        model.to(device)
        
        # Evaluation loop
        model.eval() # Set model in evaluation mode before running inference
        test_loss_total = 0.0
        test_correct = 0
        test_topk_correct = 0
        test_loader_tqdm = tqdm(test_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in test_loader_tqdm:
                segments, geotemp_features, padded_true_horizon_indices = batch
                segments, geotemp_features, padded_true_horizon_indices = segments.to(device), geotemp_features.to(device), padded_true_horizon_indices.to(device)

                # Predict depth markers (as padded tensors)
                padded_pred_horizon_embeddings = model(segments=segments, geo_temp_features=geotemp_features[:, 1:]) # 'index' column not used in model
                
                true_horizon_embeddings = torch.stack([torch.tensor(self.dataprocessor.embeddings_dict['embedding'][lab.item()]) for lab in padded_true_horizon_indices.view(-1) if lab != -1]).to(device)
                pred_horizon_embeddings = torch.stack([pred for pred, lab in zip(padded_pred_horizon_embeddings.view(-1, padded_pred_horizon_embeddings.size(-1)), padded_true_horizon_indices.view(-1)) if lab != -1]).to(device)
                true_horizon_indices = padded_true_horizon_indices.view(-1)[padded_true_horizon_indices.view(-1) != -1]

                # Normalize embeddings for the cosine loss, true embeddings are already normalized
                pred_horizon_embeddings = F.normalize(pred_horizon_embeddings, p=2, dim=1)
                
                # Compute batch losses
                # Create a dummy "same class" tensor with 1s for the cosine similarity
                same_class = torch.ones(pred_horizon_embeddings.size(0)).to(device)
                test_loss = self.cosine_loss(pred_horizon_embeddings, true_horizon_embeddings, same_class)

                # Add batch losses to total loss
                test_loss_total += test_loss.item()
                test_correct += self.horizon_topk_acc(1)(pred_horizon_embeddings, true_horizon_indices)
                test_topk_correct += self.horizon_topk_acc(self.topk)(pred_horizon_embeddings, true_horizon_indices)
        
        test_metrics = {
            'test_cosine_loss': test_loss_total / len(test_loader),
            'test_acc': test_correct / len(test_loader),
            'test_topk_acc': test_topk_correct / len(test_loader)
        }
        
        logger.info(f"\nTotal Test Cosine Loss: {test_metrics['test_cosine_loss']:.4f}, Test Acc: {test_metrics['test_acc']:.4f}, Test Top-{self.topk} Acc: {test_metrics['test_topk_acc']:.4f}")
        
        return test_metrics
    
    def get_model(self) -> nn.Module:
        return SimpleHorizonClassifier(
            geo_temp_input_dim=len(self.dataprocessor.geotemp_img_infos) - 2, # without index and img path
            geo_temp_output_dim=256,
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
        
        plt.tight_layout()
        
        plt.savefig(f'{model_output_dir}/losses_and_accuracies.png')
        if wandb_image_logging:
            wandb.log({"Losses and Accuracies": wandb.Image(figure)})