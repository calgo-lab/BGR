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
    def __init__(self, input_dim, output_dim):
        super(GeoTemporalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class DepthMarkerPredictor(nn.Module):
    def __init__(self, input_dim, transformer_dim=128, num_heads=4, num_layers=2, max_seq_len=10, stop_token=100):
        super(DepthMarkerPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads), num_layers=num_layers
        )
        self.predictor = nn.Linear(transformer_dim, 1) # Predict depth per step
        self.max_seq_len = max_seq_len
        self.stop_token = stop_token

    def forward(self, x):
        x = self.fc(x).unsqueeze(0).repeat(self.max_seq_len, 1, 1)
        x = self.transformer(x)
        x = self.predictor(x).squeeze(-1)  # (max_seq_len, batch_size)

        """
        # Mask outputs based on stop token
        outputs, masks = [], []
        for batch_idx in range(depth_predictions.size(1)):
            depth_list, mask = [], []
            for step_idx in range(depth_predictions.size(0)):
                value = depth_predictions[step_idx, batch_idx]#.item()
                if value.item() >= self.stop_token:
                    break
                depth_list.append(value)
                mask.append(1)

            # Padding with stop token
            pad_len = self.max_seq_len - len(depth_list)
            depth_list.extend([self.stop_token] * pad_len) # fill in the list of predicted depths with stop tokens till max allowed size
            mask.extend([0] * pad_len) # complete the mask list with 0's for the overflow
            outputs.append(torch.tensor(depth_list, device=features.device))
            masks.append(torch.tensor(mask, device=features.device))

        return torch.stack(outputs), torch.stack(masks)"""
        return torch.transpose(x, 0, 1)


class TabularPropertyPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TabularPropertyPredictor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class HorizonEmbedder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HorizonEmbedder, self).__init__()
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.embedder(x)

class HorizonClassifier(nn.Module):
    def __init__(self, geo_temp_input_dim, geo_temp_output_dim=32, transformer_dim=128, num_transformer_heads=4, num_transformer_layers=2,
                 max_seq_len=10, stop_token=100):#embedding_dim=64):
        super(HorizonClassifier, self).__init__()
        self.image_encoder = ImageEncoder()
        self.geo_temp_encoder = GeoTemporalEncoder(geo_temp_input_dim, geo_temp_output_dim)
        self.depth_marker_predictor = DepthMarkerPredictor(self.image_encoder.num_img_features + geo_temp_output_dim,
                                                           transformer_dim, num_transformer_heads, num_transformer_layers,
                                                           max_seq_len, stop_token)
        #self.tabular_property_predictor = TabularPropertyPredictor(image_feature_dim + geo_temp_hidden_dim, tabular_output_dim)

    def forward(self, image, geo_temp):
        image_features = self.image_encoder(image)
        geo_temp_features = self.geo_temp_encoder(geo_temp)
        combined_features = torch.cat([image_features, geo_temp_features], dim=-1)
        depth_markers = self.depth_marker_predictor(combined_features)
        #tabular_properties = self.tabular_property_predictor(combined_features)
        return depth_markers#, tabular_properties


