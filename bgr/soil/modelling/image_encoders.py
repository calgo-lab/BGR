import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.num_img_features = self.cnn.fc.in_features # store before replacing classification head with identity (512 for resnet18)
        self.cnn.fc = nn.Identity()  # Removing the final classification layer

    def forward(self, x):
        return self.cnn(x)