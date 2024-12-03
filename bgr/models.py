import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor


class ImageTabularModel(nn.Module):
    def __init__(self, vision_backbone, num_tabular_features, num_classes):
        super(ImageTabularModel, self).__init__()

        # Load pretrained DINOv2-Model from Hugging Face
        self.vision_backbone = AutoModel.from_pretrained(vision_backbone)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(vision_backbone)

        # MLP for the tabular data
        self.fc_tabular = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        # Combined Fully Connected Layers
        self.fc_combined = nn.Sequential(
            nn.Linear(self.vision_backbone.config.hidden_size + 32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, tabular_features):
        # Extract image features
        image_features = self.vision_backbone(pixel_values=image).last_hidden_state[:, 0, :] # Use [CLS] token representation

        # Process tabular features with the MLP
        tabular_features_processed = self.fc_tabular(tabular_features)

        # Combine image and tabular features
        combined_features = torch.cat((image_features, tabular_features_processed), dim=1)

        # Final prediction
        output = self.fc_combined(combined_features)
        return output