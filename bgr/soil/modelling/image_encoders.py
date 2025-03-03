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


class HDCNNEncoder(nn.Module):
    def __init__(self, input_channels=3, output_dim=512):
        super(HDCNNEncoder, self).__init__()

        self.num_img_features = output_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 512)
        x = self.fc(x)
        return x
