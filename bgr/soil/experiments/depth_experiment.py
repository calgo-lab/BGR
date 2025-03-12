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

from bgr.soil.data.horizon_tabular_data import HorizonDataProcessor
from bgr.soil.experiments import Experiment
from bgr.soil.training_args import TrainingArgs
from bgr.soil.modelling.general_models import HorizonSegmenter
from bgr.soil.metrics import DepthMarkerLoss, depth_iou
from bgr.soil.utils import pad_tensor
from bgr.soil.data.datasets import ImageTabularDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthExperiment(Experiment):
    def __init__(self, training_args: TrainingArgs, target: str, dataprocessor: HorizonDataProcessor):
        self.training_args = training_args
        self.target = target
        self.dataprocessor = dataprocessor
        self.trained = False
        
        self.depth_loss = DepthMarkerLoss(lambda_mono=0, lambda_div=0)
        self.image_normalization = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet statistics
        ])
    
    def train_and_validate(self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_output_dir: str
    ) -> tuple[nn.Module, dict]:
        
        train_dataset = ImageTabularDataset(
            dataframe=train_df,
            normalize=self.image_normalization,
            augment=[],
            image_path='file',
            label=None, # no label column as input; access it instead via 'index' during training
            feature_columns=self.dataprocessor.geotemp_img_infos[:-1] # without 'file'
        )
        train_loader = DataLoader(train_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        val_dataset = ImageTabularDataset(
            dataframe=val_df,
            normalize=self.image_normalization,
            augment=[],
            image_path='file',
            label=None, # no label column as input; access it instead via 'index' during training
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
        self.train_iou_history, self.val_iou_history = [], []

        for epoch in range(self.training_args.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.training_args.num_epochs}")
            
            # Training
            model.train()
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
                    true_depths.append(train_df.loc[train_df['index'] == idx, 'Untergrenze'].values[0])

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

            # Evaluation loop
            model.eval() # Set model in evaluation mode before running inference
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
                        true_depths.append(val_df.loc[val_df['index'] == idx, 'Untergrenze'].values[0])

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

            epoch_metrics = {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_iou': avg_train_iou,
                'val_iou': avg_val_iou
            }
            for callback in self.training_args.callbacks:
                callback(model, epoch_metrics, epoch)
            
            # Apply the scheduler with validation loss
            scheduler.step(avg_val_loss)
            # Log the current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Log metrics
            self.train_loss_history.append(avg_train_loss); self.val_loss_history.append(avg_val_loss)
            self.train_iou_history.append(avg_train_iou); self.val_iou_history.append(avg_val_iou)

            logger.info(f"Epoch {epoch+1}, Total Training Depth Loss: {avg_train_loss:.4f}, Training IoU: {avg_train_iou:.4f}")
            logger.info(f"\nTotal Validation Depth Loss: {avg_val_loss:.4f}, Validation IoU: {avg_val_iou:.4f}")
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
        
        test_dataset = ImageTabularDataset(
            dataframe=test_df,
            normalize=self.image_normalization,
            augment=[],
            image_path='file',
            label=None, # no label column as input; access it instead via 'index' during training
            feature_columns=self.dataprocessor.geotemp_img_infos[:-1] # without 'file'
        )
        test_loader = DataLoader(test_dataset, batch_size=self.training_args.batch_size, shuffle=True, num_workers=self.training_args.num_workers, drop_last=True)
        
        device = self.training_args.device
        model.to(device)
        
        model.eval() # Set model in evaluation mode before running inference
        test_loss_total = 0.0
        test_iou = 0.0
        test_loader_tqdm = tqdm(test_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in test_loader_tqdm:
                images, geotemp_features = batch
                images, geotemp_features = images.to(device), geotemp_features.to(device)

                # Get corresponding true depth markers via index column in df (see training step above)
                true_depths = []
                batch_indices = geotemp_features.cpu().numpy()[:, 0]
                for idx in batch_indices:
                    true_depths.append(test_df.loc[test_df['index'] == idx, 'Untergrenze'].values[0])

                # Turn list of depths into a padded tensor and also return mask of valid positions
                padded_true_depths = pad_tensor(true_depths,
                                                max_seq_len=model.depth_marker_predictor.max_seq_len,
                                                stop_token=model.depth_marker_predictor.stop_token,
                                                device=device)

                # Predict depth markers (as padded tensors) and morphological features
                pred_depths = model(images=images, geo_temp=geotemp_features[:, 1:]) # 'index' column not used in model

                # Compute batch losses
                test_loss = self.depth_loss(pred_depths, padded_true_depths)

                # Add batch losses to total loss
                test_loss_total += test_loss.item()

                # Calculate IoU
                test_iou += depth_iou(pred_depths, padded_true_depths, model.stop_token)
        
        test_metrics = {
            'test_loss': test_loss_total / len(test_loader),
            'test_iou': test_iou / len(test_loader)
        }
        
        logger.info(f"\nTotal Test Depth Loss: {test_metrics['test_loss']:.4f}, Test IoU: {test_metrics['test_iou']:.4f}")
        
        return test_metrics
    
    def get_model(self) -> nn.Module:
        return HorizonSegmenter(
            geo_temp_input_dim=len(self.dataprocessor.geotemp_img_infos) - 2, # without index and img path
            geo_temp_output_dim=256,
            max_seq_len=8, # 8 is the longest number of horizons in one image
            stop_token=1.0,
            rnn_hidden_dim=256
        )
    
    def plot_losses(self, model_output_dir: str, wandb_image_logging: bool) -> None:
        if not self.trained:
            raise ValueError("Model has not been trained yet.")
        
        complete_epochs = len(self.train_loss_history) + 1
        loss_histories = {
            'Depth': (self.train_loss_history, self.val_loss_history)
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
        plt.tight_layout()
        
        plt.savefig(f'{model_output_dir}/losses.png')
        if wandb_image_logging:
            wandb.log({"Losses": wandb.Image(plt)})