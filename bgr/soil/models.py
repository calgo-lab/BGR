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
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class TransformerDepthMarkerPredictor(nn.Module):
    def __init__(self, input_dim, transformer_dim=128, num_heads=4, num_layers=2, max_seq_len=10, stop_token=1.0):
        super(TransformerDepthMarkerPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, transformer_dim),
            nn.BatchNorm1d(transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.BatchNorm1d(transformer_dim),
            nn.ReLU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads), num_layers=num_layers
        )
        self.predictor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(transformer_dim, max_seq_len)
        )
        self.max_seq_len = max_seq_len
        self.stop_token = stop_token

    def forward(self, x):
        x = self.fc(x).unsqueeze(0)
        x = self.transformer(x)
        x = self.predictor(x).squeeze(-1)
        x = torch.transpose(x, 0, 1)
        x = torch.sigmoid(x).squeeze(1) # (batch_size, max_seq_len)

        # Round values very near to stop_token and above it to stop_token
        x = torch.where(
            x > self.stop_token - 0.01,
            torch.full_like(x, self.stop_token),
            x)

        return x


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


class LSTMDepthMarkerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, max_seq_len=10, stop_token=1.0):
        super(LSTMDepthMarkerPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.stop_token = stop_token
        self.num_lstm_layers = 2

        # First fully connected layer (projecting the concatenated vector of image and geotemp features)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Decoder - will store the previous predictions in the hidden state
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True,
                           num_layers=self.num_lstm_layers, dropout=0.2)

        # Output Layer - predicts one depth at a time
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        hidden_state = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_dim).to(x.device)
        cell_state   = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_dim).to(x.device)
        depth_markers = []

        x = self.fc(x).unsqueeze(1) # (batch_size, 1, hidden_dim)
        for _ in range(self.max_seq_len):
            output, (hidden_state, cell_state) = self.rnn(x, (hidden_state, cell_state))
            depth_marker = self.predictor(output).squeeze()
            depth_markers.append(depth_marker)

        depth_markers = torch.stack(depth_markers, dim=1) # (batch_size, max_seq_len)

        # Round values very near to stop_token and above it to stop_token
        depth_markers = torch.where(
            depth_markers > self.stop_token - 0.01,
            torch.full_like(depth_markers, self.stop_token),
            depth_markers)

        return depth_markers


class HorizonClassifier(nn.Module):
    def __init__(self,
                 geo_temp_input_dim, geo_temp_output_dim=32, # params for geotemp encoder
                 #transformer_dim=128, num_transformer_heads=4, num_transformer_layers=2, # params for Transformer- or CrossAttentionDepthPredictor
                 rnn_hidden_dim=256, # params for LSTMDepthPredictor
                 #embedding_dim=64,
                 max_seq_len=10, stop_token=1.0):
        super(HorizonClassifier, self).__init__()
        self.image_encoder = ImageEncoder()
        self.geo_temp_encoder = GeoTemporalEncoder(geo_temp_input_dim, geo_temp_output_dim)

        # Choose from different depth predictors
        #self.depth_marker_predictor = TransformerDepthMarkerPredictor(self.image_encoder.num_img_features + geo_temp_output_dim,
        #                                                              transformer_dim, num_transformer_heads, num_transformer_layers,
        #                                                              max_seq_len, stop_token)
        self.depth_marker_predictor = LSTMDepthMarkerPredictor(self.image_encoder.num_img_features + geo_temp_output_dim,
                                                               rnn_hidden_dim, max_seq_len, stop_token)

        #self.tabular_property_predictor = TabularPropertyPredictor(image_feature_dim + geo_temp_hidden_dim, tabular_output_dim)

    def forward(self, image, geo_temp, targets=None):
        image_features = self.image_encoder(image)
        geo_temp_features = self.geo_temp_encoder(geo_temp)
        combined_features = torch.cat([image_features, geo_temp_features], dim=-1)
        depth_markers = self.depth_marker_predictor(combined_features)#, targets=targets) # use targets for teacher forcing, only if DepthPredictor accepts it in forward()
        #tabular_properties = self.tabular_property_predictor(combined_features)
        #horizon_embedding = ...

        return depth_markers#, tabular_properties, horizon_embedding


### DEPRECATED ###
# Problem: this one only predicts the same number throughout the depths list.
# Aggressive regularization and adjusted loss only help insignificantly in varying the output.
# By removing the .repeat() in forward and making self.fc_out predict max_seq_len depths, there is more variance in the predicted depths list,
# but they still stay identical for all the samples. I did the same with the TransformerDepthPredictor above.
class CrossAttentionDepthMarkerPredictor(nn.Module):
    def __init__(self, input_dim, transformer_dim=128, num_heads=4, num_layers=2,
                 max_seq_len=10, stop_token=1.0):
        super(CrossAttentionDepthMarkerPredictor, self).__init__()

        self.stop_token = stop_token
        self.max_seq_len = max_seq_len

        # Encoder: Processes input features (image + geotemporal features)
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, transformer_dim),
            nn.BatchNorm1d(transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.BatchNorm1d(transformer_dim),
            nn.ReLU()
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads), num_layers=num_layers
        )

        # Decoder: Predicts depth markers step-by-step using cross-attention
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=transformer_dim, nhead=num_heads), num_layers=num_layers
        )

        # Output Layers
        self.depth_embed = nn.Embedding(max_seq_len + 1, transformer_dim)  # Embedding for positional depth tokens
        self.fc_out = nn.Linear(transformer_dim, 1)  # Predict a single depth value at each step

        self.sigmoid = nn.Sigmoid()  # Normalize output depths to [0, 1]

    def forward(self, x, targets=None):
        """
        Args:
            x: Input features (batch_size, input_dim)
            targets: Optional ground truth depth markers for teacher forcing (batch_size, seq_len)

        Returns:
            Predicted depth markers (batch_size, max_seq_len)
        """
        batch_size = x.size(0)

        # Encode input features
        x = self.encoder_fc(x)  # (batch_size, transformer_dim)
        x = x.unsqueeze(1).repeat(1, self.max_seq_len, 1).permute(1, 0, 2)  # (seq_len, batch_size, transformer_dim)
        memory = self.encoder(x)  # (seq_len, batch_size, transformer_dim)

        # Initialize decoder inputs
        sos_token = torch.zeros((batch_size, 1, 1), device=x.device)  # Start-of-sequence token
        decoder_input = self.depth_embed(sos_token.long()).squeeze(1).permute(1, 0, 2)  # (1, batch_size, transformer_dim)

        outputs = []
        for step in range(self.max_seq_len):
            # Cross-attention decoding
            decoder_output = self.decoder(decoder_input, memory)  # (step + 1, batch_size, transformer_dim)
            depth_pred = self.fc_out(decoder_output[-1])  # (batch_size, 1)
            depth_pred = self.sigmoid(depth_pred)  # Normalize to [0, 1]
            outputs.append(depth_pred)

            # Teacher forcing: use ground truth depth markers as the next input during training
            if targets is not None and step < targets.size(1):
                next_input = self.depth_embed(targets[:, step].unsqueeze(-1).long())  # (batch_size, transformer_dim)
            else:
                next_input = self.depth_embed(depth_pred.long())  # Use predicted depth marker
            decoder_input = torch.cat([decoder_input, next_input.permute(1, 0, 2)], dim=0)

        outputs = torch.cat(outputs, dim=-1)  # (batch_size, max_seq_len)
        return outputs