from __future__ import annotations
import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_ # modifies the tensors in-place (vs clip_grad_norm)
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bgr.soil.training_args import TrainingArgs


from bgr.soil.data.horizon_tabular_data import HorizonDataProcessor
from bgr.soil.experiments import Experiment
from bgr.soil.modelling.depth.depth_models import SimpleDepthModel
from bgr.soil.metrics import DepthMarkerLoss, depth_iou
from bgr.soil.utils import pad_tensor
from bgr.soil.data.datasets import ImagePatchesTabularDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDepthsGeotempsResNet(Experiment):
    def __init__(self, training_args: 'TrainingArgs', target: str, dataprocessor: HorizonDataProcessor):
        self.training_args = training_args
        self.target = target
        self.dataprocessor = dataprocessor
        self.trained = False
        
        self.depth_loss = DepthMarkerLoss(lambda_mono=0, lambda_div=0)
        self.image_normalization = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet statistics
        ])
        
        # Retrieve the experiment hyperparameters
        self.hyperparameters = SimpleDepthsGeotempsResNet.get_experiment_hyperparameters()
        self.hyperparameters.update(training_args.hyperparameters)
    
    def train_and_validate(self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_output_dir: str
    ) -> tuple[nn.Module, dict]:
        
        train_dataset = ImagePatchesTabularDataset(
            dataframe=train_df,
            normalize=self.image_normalization,
            augment=[],
            img_path_column='file',
            label_column=None, # no label column as input; access it instead via 'index' during training
            geotemp_columns=self.dataprocessor.geotemp_img_infos[:-1], # without 'file'
            image_patch_number=self.hyperparameters['num_image_patches']
        )
        train_loader = DataLoader(train_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        val_dataset = ImagePatchesTabularDataset(
            dataframe=val_df,
            normalize=self.image_normalization,
            augment=[],
            img_path_column='file',
            label_column=None, # no label column as input; access it instead via 'index' during training
            geotemp_columns=self.dataprocessor.geotemp_img_infos[:-1], # without 'file'
            image_patch_number=self.hyperparameters['num_image_patches']
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
        self.train_iou_history, self.val_iou_history = [], []

        for epoch in range(1, self.training_args.num_epochs + 1):
            logger.info(f"Epoch {epoch}/{self.training_args.num_epochs}")
            
            # Training loop
            model.train()
            avg_train_loss, avg_train_iou = self._train_model(train_loader, device, model, optimizer)

            # Evaluation loop
            model.eval() # Set model in evaluation mode before running inference
            avg_val_loss, avg_val_iou = self._evaluate_model(val_loader, device, model)
            
            epoch_metrics = {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_iou': avg_train_iou,
                'val_iou': avg_val_iou
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
            self.train_iou_history.append(avg_train_iou.detach().cpu().numpy()); self.val_iou_history.append(avg_val_iou.detach().cpu().numpy())

            logger.info(
                f"""
                Epoch {epoch}/{self.training_args.num_epochs} Metrics:
                --------------------------------
                Training:
                - Depth Loss: {avg_train_loss:.4f}
                - IoU: {avg_train_iou:.4f}
                
                Validation:
                - Depth Loss: {avg_val_loss:.4f}
                - IoU: {avg_val_iou:.4f}
                
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
            'Train Depth Loss': self.train_loss_history[-1],
            'Validation Depth Loss': self.val_loss_history[-1],
            'Train IoU': self.train_iou_history[-1],
            'Validation IoU': self.val_iou_history[-1]
        }
        return model, return_metrics
    
    def test(self,
        model: nn.Module,
        test_df: pd.DataFrame,
        model_output_dir: str
    ) -> dict:
        
        test_dataset = ImagePatchesTabularDataset(
            dataframe=test_df,
            normalize=self.image_normalization,
            augment=[],
            img_path_column='file',
            label_column=None, # no label column as input; access it instead via 'index' during training
            geotemp_columns=self.dataprocessor.geotemp_img_infos[:-1], # without 'file'
            image_patch_number=self.hyperparameters['num_image_patches']
        )
        test_loader = DataLoader(test_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        device = self.training_args.device
        model.to(device)
        
        print("--------------------------------")
        model.eval() # Set model in evaluation mode before running inference
        avg_test_loss, avg_test_iou = self._evaluate_model(test_loader, device, model)
        
        test_metrics = {
            'Test Depth Loss': avg_test_loss,
            'Test IoU': avg_test_iou
        }
        
        logger.info(
                f"""
                Test Metrics:
                --------------------------------
                Testing:
                - Depth Loss: {avg_test_loss:.4f}
                - IoU: {avg_test_iou:.4f}
                --------------------------------
                """
            )
        
        return test_metrics
    
    def get_model(self) -> nn.Module:
        return SimpleDepthModel(
            geo_temp_input_dim=len(self.dataprocessor.geotemp_img_infos) - 2, # without index and img path
            geo_temp_output_dim=self.hyperparameters['geotemp_output_dim'],
            image_encoder_output_dim=self.hyperparameters['image_encoder_output_dim'],
            max_seq_len=self.hyperparameters['max_seq_len'],
            stop_token=self.hyperparameters['stop_token'],
            rnn_hidden_dim=self.hyperparameters['rnn_hidden_dim'],
            patch_size=self.hyperparameters['patch_size'],
            predefined_random_patches=True # True = use ResNet, False = use custom CNN
        )
    
    def plot_losses(self, model_output_dir: str, wandb_image_logging: bool) -> None:
        if not self.trained:
            raise ValueError("Model has not been trained yet.")
        
        complete_epochs = len(self.train_loss_history) + 1
        loss_histories = {
            'Depth': (self.train_loss_history, self.val_loss_history)
        }
        iou_histories = {
            'IoU': (self.train_iou_history, self.val_iou_history)
        }
        
        figure = plt.figure(figsize=(10, 5))
        for i, (title, (train_history, val_history)) in enumerate(loss_histories.items()):
            plt.subplot(1, 2, i + 1)
            plt.plot(range(1, complete_epochs), train_history, label=f'Train {title} Loss', marker='o', color='b')
            plt.plot(range(1, complete_epochs), val_history, label=f'Validation {title} Loss', marker='o', color='r')
            plt.title(f'{title} Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
        
        for i, (title, (train_history, val_history)) in enumerate(iou_histories.items()):
            plt.subplot(1, 2, i + 2)
            plt.plot(range(1, complete_epochs), train_history, label=f'Train {title}', marker='o', color='b')
            plt.plot(range(1, complete_epochs), val_history, label=f'Validation {title}', marker='o', color='r')
            plt.title(f'{title} Scores')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.legend()
            plt.grid()
        plt.tight_layout()
        
        plt.savefig(f'{model_output_dir}/losses_and_iou_scores.pdf', bbox_inches='tight', format='pdf')
        if wandb_image_logging:
            wandb.log({"Losses and IoU Scores": wandb.Image(plt)})
    
    def _train_model(self, train_loader, device, model, optimizer):
        train_loss_total = 0.0
        train_iou = 0.0
        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_loader_tqdm:
            images, geotemp_features = batch
            images, geotemp_features = images.to(device), geotemp_features.to(device)

            optimizer.zero_grad() # otherwise, PyTorch accumulates the gradients during backprop

            # Get corresponding true depth markers and morphological features via index column in df (the first value in every row in geotemp)
            # Note: the code accounts for duplicate indexes resulting after the augmentations in the ImageTabularDataset class (is there a better way than duplicating indexes during augmentation?)
            true_depths = []
            batch_indices = geotemp_features.cpu().numpy()[:, 0]
            for idx in batch_indices:
                true_depths.append(train_loader.dataset.dataframe.loc[train_loader.dataset.dataframe['index'] == idx, 'Untergrenze'].values[0])

            # Turn list of depths into a padded tensor and also return mask of valid positions
            padded_true_depths = pad_tensor(true_depths,
                                            max_seq_len=model.depth_marker_predictor.max_seq_len,
                                            stop_token=model.depth_marker_predictor.stop_token,
                                            device=device)

            # Predict depth markers (as padded tensors)
            pred_depths = model(images=images, geo_temp=geotemp_features[:, 1:]) # 'index' column not used in model

            # Compute individual losses, then sum them together for backprop
            train_loss = self.depth_loss(pred_depths, padded_true_depths)
            train_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Calculate batch losses to total loss
            train_loss_total += train_loss.item()

            # Calculate IoU
            train_iou += depth_iou(pred_depths, padded_true_depths, model.stop_token)

            train_loader_tqdm.set_postfix(loss=train_loss.item())

        # Average losses and iou at the end of the epoch
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        
        return avg_train_loss, avg_train_iou
    
    def _evaluate_model(self, val_loader, device, model):
        val_loss_total = 0.0
        val_iou = 0.0
        val_loader_tqdm = tqdm(val_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in val_loader_tqdm:
                images, geotemp_features = batch
                images, geotemp_features = images.to(device), geotemp_features.to(device)

                # Get corresponding true depth markers via index column in df (see training step above)
                true_depths = []
                batch_indices = geotemp_features.cpu().numpy()[:, 0]
                for idx in batch_indices:
                    true_depths.append(val_loader.dataset.dataframe.loc[val_loader.dataset.dataframe['index'] == idx, 'Untergrenze'].values[0])

                # Turn list of depths into a padded tensor and also return mask of valid positions
                padded_true_depths = pad_tensor(true_depths,
                                                max_seq_len=model.depth_marker_predictor.max_seq_len,
                                                stop_token=model.depth_marker_predictor.stop_token,
                                                device=device)

                # Predict depth markers (as padded tensors) and morphological features
                pred_depths = model(images=images, geo_temp=geotemp_features[:, 1:]) # 'index' column not used in model

                # Compute batch losses
                val_loss = self.depth_loss(pred_depths, padded_true_depths)

                # Add batch losses to total loss
                val_loss_total += val_loss.item()

                # Calculate IoU
                val_iou += depth_iou(pred_depths, padded_true_depths, model.stop_token)

        # Average losses and iou at the end of the epoch
        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        return avg_val_loss, avg_val_iou
            
    @staticmethod
    def get_experiment_hyperparameters() -> dict:
        return {
            'num_image_patches' : 48, # only used with SegmentPatches dataset for ResNetPatch
            'geotemp_output_dim': 256,
            'image_encoder_output_dim': 512,
            'max_seq_len': 8,
            'stop_token': 1.0,
            'rnn_hidden_dim': 256,
            'patch_size': 512
        }