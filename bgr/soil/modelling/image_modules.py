import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

class MaskedResNetImageEncoder(nn.Module):
    """
    Encodes a batch of padded images using a CNN backbone and masked average pooling.

    Args:
        backbone_name (str): Name of the torchvision model to use as backbone (e.g., 'resnet50').
        pretrained (bool): Whether to load pretrained weights for the backbone.
        output_embedding_dim (Optional[int]): If provided, adds a final linear layer
                                                to project features to this dimension.
    """
    def __init__(self, resnet_version='18', pretrained: bool = True, output_embedding_dim: Optional[int] = None):
        super().__init__()

        # Load the pre-trained backbone
        if resnet_version == '18':
            backbone = models.resnet18(weights=models.resnet.ResNet18_Weights.DEFAULT if pretrained else None)
            self.feature_dim = backbone.fc.in_features # Get feature dim before final layer
            # Remove the final classification layer and adaptive pool layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        else:
            backbone = models.resnet50(weights=models.resnet.ResNet50_Weights.DEFAULT if pretrained else None)
            self.feature_dim = backbone.fc.in_features # Get feature dim before final layer
            # Remove the final classification layer and adaptive pool layer (pooling will be done with the image_mask)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # Optional final projection layer
        self.projection = nn.Linear(self.feature_dim, output_embedding_dim) if output_embedding_dim else nn.Identity()
        self.output_dim = output_embedding_dim if output_embedding_dim else self.feature_dim


    def forward(self, padded_image: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for masked image encoding.

        Args:
            padded_image (torch.Tensor): Batch of padded images (N, C, H, W).
            image_mask (torch.Tensor): Batch of masks (N, 1, H, W), True where valid.

        Returns:
            torch.Tensor: Batch of image embeddings (N, output_dim).
        """
        # 1. Get features from the backbone
        # -> Shape: (batch_size, channels, height, width)
        features = self.backbone(padded_image)
        batch_size, channels, height_features, width_features = features.shape

        # 2. Downsample the mask to match feature map dimensions
        # Ensure mask is float for interpolation
        mask_float = image_mask.float()
        # Use nearest neighbor interpolation to keep mask sharp
        downsampled_mask = F.interpolate(mask_float, size=(height_features, width_features), mode='nearest')

        # 3. Perform masked average pooling
        # Zero out features in padded regions
        masked_features = features * downsampled_mask # Broadcasting (N, C_feat, Hf, Wf) * (N, 1, Hf, Wf)

        # Sum features over spatial dimensions
        feature_sum = masked_features.sum(dim=[2, 3]) # Shape: (N, C_feat)

        # Count number of valid (unmasked) spatial locations per feature map
        # Summing the float mask (0s and 1s) gives the count
        valid_pixels_count = downsampled_mask.sum(dim=[2, 3]) # Shape: (N, 1)

        # Avoid division by zero if a mask is all False (shouldn't happen with valid images)
        valid_pixels_count = valid_pixels_count.clamp(min=1e-6)

        # Calculate masked average
        # Shape: (N, C_feat) / (N, 1) -> (N, C_feat)
        masked_avg_features = feature_sum / valid_pixels_count

        # 4. Optional projection
        # -> Shape: (N, output_dim)
        output_embedding = self.projection(masked_avg_features)

        return output_embedding

class ResNetEncoder(nn.Module):
    def __init__(self, resnet_version = '18'):
        super(ResNetEncoder, self).__init__()
        if resnet_version == '18':
            self.cnn = models.resnet18(weights=models.resnet.ResNet18_Weights.DEFAULT)
        else:
            self.cnn = models.resnet50(weights=models.resnet.ResNet50_Weights.DEFAULT)
        self.num_img_features = self.cnn.fc.in_features # store before replacing classification head with identity (512 for resnet18)
        self.cnn.fc = nn.Identity()  # Removing the final classification layer

    def forward(self, x):
        return self.cnn(x)
    
class ResNetPatchEncoder(nn.Module):
    def __init__(self, output_dim, resnet_version='18'):
        super(ResNetPatchEncoder, self).__init__()
        if resnet_version == '18':
            self.cnn = models.resnet18(weights=models.resnet.ResNet18_Weights.DEFAULT)
        elif resnet_version == '50':
            self.cnn = models.resnet50(weights=models.resnet.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported ResNet version. Choose either '18' or '50'.")
        
        self.num_img_features = output_dim
        self.cnn.fc = nn.Identity()

    def forward(self, x):
        # x-shape (batch_size, num_patches, channels, height, width)
        batch_size, num_patches, channels, height, width = x.shape

        # Reshape to treat each patch as a separate sample in the batch
        patches = x.contiguous().view(batch_size * num_patches, channels, height, width)

        # Pass each patch through the ResNet backbone
        patch_features = self.cnn(patches)  # Shape: (batch_size * num_patches, cnn_output_dim)

        # Reshape back to (batch_size, num_patches, cnn_output_dim)
        patch_features = patch_features.view(batch_size, num_patches, -1)

        # Aggregate features across patches
        aggregated_features = torch.mean(patch_features, dim=1)  # Shape: (batch_size, cnn_output_dim)

        # Optionally project to the desired output_dim
        if aggregated_features.size(-1) != self.num_img_features:
            aggregated_features = nn.Linear(aggregated_features.size(-1), self.num_img_features)(aggregated_features)

        return aggregated_features  # Shape: (batch_size, output_dim)
        
class HDCNNEncoder(nn.Module):
    def __init__(self, input_channels=3, output_dim=512):
        super(HDCNNEncoder, self).__init__()

        self.num_img_features = output_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, output_dim // 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim // 16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(output_dim // 16, output_dim // 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim // 8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(output_dim // 8, output_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim // 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(output_dim // 4, output_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 512)
        return x


class PatchCNNEncoder(nn.Module):
    def __init__(self, input_channels=3, output_dim=512, patch_size=128, patch_stride=64):
        super(PatchCNNEncoder, self).__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride  # Allow overlapping patches
        self.num_img_features = output_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, output_dim//16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim//16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(output_dim//16, output_dim//8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim//8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(output_dim//8, output_dim//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim//4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(output_dim//4, output_dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim//2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(output_dim//2, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Reduce to (batch_size, 512, 1, 1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_stride).unfold(3, self.patch_size, self.patch_stride)
        patches = patches.contiguous().view(-1, channels, self.patch_size, self.patch_size)

        patch_features = self.conv_layers(patches)
        patch_features = self.global_pool(patch_features)
        patch_features = torch.flatten(patch_features, 1)

        # Combine all patch features into a single image feature vector
        image_feature = torch.mean(patch_features.view(batch_size, -1, patch_features.size(-1)), dim=1)

        return image_feature
