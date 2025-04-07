import torch
import torch.nn as nn
from torchvision import models

class ResNetEncoder(nn.Module):
    def __init__(self, resnet_version = '18'):
        super(ResNetEncoder, self).__init__()
        if resnet_version == '18':
            self.cnn = models.resnet18(pretrained=True)
        else:
            self.cnn = models.resnet50(pretrained=True)
        self.num_img_features = self.cnn.fc.in_features # store before replacing classification head with identity (512 for resnet18)
        self.cnn.fc = nn.Identity()  # Removing the final classification layer

    def forward(self, x):
        return self.cnn(x)
    
class ResNetPatchEncoder(nn.Module):
    def __init__(self, output_dim, resnet_version='18'):
        super(ResNetPatchEncoder, self).__init__()
        if resnet_version == '18':
            self.cnn = models.resnet18(pretrained=True)
        elif resnet_version == '50':
            self.cnn = models.resnet50(pretrained=True)
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
