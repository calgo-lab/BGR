import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel, AutoFeatureExtractor


class ImageTabularModel(nn.Module):
    """
    Simple baseline model that combines ViT image features with tabular features
    """
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


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.num_img_features = self.cnn.fc.in_features # store before replacing classification head with identity (512 for resnet18)
        self.cnn.fc = nn.Identity()  # Removing the final classification layer

    def forward(self, x):
        return self.cnn(x)


# Geospatial and Temporal Encoder for features in df_loc
class GeoTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GeoTemporalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class DepthMarkerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_seq_len):
        super(DepthMarkerPredictor, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Predicting one depth marker at a time
        self.max_seq_len = max_seq_len

    def forward(self, x, teacher_forcing=False, target_seq=None):
        #batch_size = x.size(0)
        outputs = []
        hidden = None

        # Autoregressive generation
        for t in range(self.max_seq_len):
            if t == 0 or not teacher_forcing:
                input_step = x if t == 0 else outputs[-1].detach()
            else:
                input_step = target_seq[:, t-1].unsqueeze(1)  # Use ground truth for teacher forcing

            output, hidden = self.rnn(input_step, hidden)
            depth_marker = self.fc(output).squeeze(-1)  # Shape: (batch_size, 1)
            outputs.append(depth_marker)

        return torch.stack(outputs, dim=1)  # Shape: (batch_size, max_seq_len)


class TabularPropertyPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TabularPropertyPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class HorizonClassifier(nn.Module):
    def __init__(self, geo_temp_input_dim, geo_temp_hidden_dim, rnn_hidden_dim):#, tabular_output_dim):
        super(HorizonClassifier, self).__init__()
        self.image_encoder = ImageEncoder()
        self.geo_temp_encoder = GeoTemporalEncoder(geo_temp_input_dim, geo_temp_hidden_dim)
        self.depth_marker_predictor = DepthMarkerPredictor(self.image_encoder.num_img_features + geo_temp_hidden_dim, rnn_hidden_dim)
        #self.tabular_property_predictor = TabularPropertyPredictor(image_feature_dim + geo_temp_hidden_dim, tabular_output_dim)

    def forward(self, image, geo_temp):
        image_features = self.image_encoder(image)
        geo_temp_features = self.geo_temp_encoder(geo_temp)
        combined_features = torch.cat([image_features, geo_temp_features], dim=-1)
        depth_markers = self.depth_marker_predictor(combined_features)
        #tabular_properties = self.tabular_property_predictor(combined_features)
        return depth_markers#, tabular_properties


