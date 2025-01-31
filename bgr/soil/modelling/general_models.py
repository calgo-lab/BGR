import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor
from bgr.soil.utils import concat_img_geotemp_depth
from bgr.soil.modelling.image_encoders import ImageEncoder
from bgr.soil.modelling.geotemp_encoders import GeoTemporalEncoder
from bgr.soil.modelling.depth_markers import LSTMDepthMarkerPredictor
from bgr.soil.modelling.tabular_predictors import MLPTabularPredictor, LSTMTabularPredictor
from bgr.soil.modelling.horizon_embedders import HorizonEmbedder


class HorizonClassifier(nn.Module):
    def __init__(self,
                 geo_temp_input_dim, geo_temp_output_dim=32, # params for geotemp encoder
                 max_seq_len=10, stop_token=1.0, # params for depth predictor (any class)
                 #transformer_dim=128, num_transformer_heads=4, num_transformer_layers=2, # params for Transformer- or CrossAttentionDepthPredictor
                 rnn_hidden_dim=256, # params for LSTMDepthPredictor
                 tabular_predictors_dict={}, # params for the tabular predictors
                 tab_pred_device='cpu'
                 #embedding_dim=64,
                 ):
        super(HorizonClassifier, self).__init__()
        self.stop_token = stop_token
        self.image_encoder = ImageEncoder()
        self.geo_temp_encoder = GeoTemporalEncoder(geo_temp_input_dim, geo_temp_output_dim)

        # Choose from different depth predictors
        #self.depth_marker_predictor = TransformerDepthMarkerPredictor(self.image_encoder.num_img_features + geo_temp_output_dim,
        #                                                              transformer_dim, num_transformer_heads, num_transformer_layers,
        #                                                              max_seq_len, stop_token)
        self.depth_marker_predictor = LSTMDepthMarkerPredictor(self.image_encoder.num_img_features + geo_temp_output_dim,
                                                               rnn_hidden_dim, max_seq_len, stop_token)

        # Define list of tabular predictors
        # Each takes as input the image_geotemp_vector extended with upper and lower bound for each horizon (MLPTabularPredictor)
        # or extended with the full padded depth marker list (LSTMTabularPredictor)
        self.tabular_predictors = []
        for tab_pred_name in tabular_predictors_dict:
            tab_output_dim = tabular_predictors_dict[tab_pred_name]['output_dim']
            tab_classif = tabular_predictors_dict[tab_pred_name]['classification']
            self.tabular_predictors.append(MLPTabularPredictor(input_dim=self.image_encoder.num_img_features + geo_temp_output_dim + 2,
                                                               output_dim=tab_output_dim,
                                                               classification=tab_classif,
                                                               name=tab_pred_name).to(tab_pred_device))
            #self.tabular_predictors.append(LSTMTabularPredictor(input_dim=self.image_encoder.num_img_features + geo_temp_output_dim + max_seq_len,
            #                                                    output_dim=tab_output_dim,
            #                                                    max_seq_len=max_seq_len,
            #                                                    stop_token=stop_token,
            #                                                    classification=tab_classif,
            #                                                    name=tab_pred_name).to(tab_pred_device))

    def forward(self, image, geo_temp, true_depths=None):
        # Extract image + geotemp features, then concatenate them
        image_features = self.image_encoder(image)
        geo_temp_features = self.geo_temp_encoder(geo_temp)
        img_geotemp_vector = torch.cat([image_features, geo_temp_features], dim=-1)

        # Predict depth markers based on concatenated vector
        depth_markers = self.depth_marker_predictor(img_geotemp_vector)

        # Concatenate segmentation info. with image_geotemp_vector
        # MLPTabular: for every horizon, only the corresponding upper and lower bound are added to the image_geotemp_vector
        # During training, these boundaries are taken from true_depths, during inference from the predicted depth_markers
        if true_depths:
            tab_inputs = concat_img_geotemp_depth(img_geotemp_vector, true_depths, self.stop_token) # at training time
        else:
            tab_inputs = concat_img_geotemp_depth(img_geotemp_vector, depth_markers, self.stop_token) # at evaluation time
        # LSTMTabular: for every horizon, the full padded depth marker list is added to the concatenated img_geotemp vector
        # During training, these boundaries are taken from padded true_depths, during inference from the predicted depth_markers
        #if true_depths:
        #    tab_inputs = torch.concat([img_geotemp_vector, true_depths], dim=1) # at training time
        #else:
        #    tab_inputs = torch.concat([img_geotemp_vector, depth_markers], dim=1) # at evaluation time

        # Note: for every feature the output of the tab_predictor has a different dimension
        tabular_predictions = {}
        for tab_predictor in self.tabular_predictors:
            tabular_predictions[tab_predictor.name] = tab_predictor(tab_inputs).squeeze()

        #horizon_embedding = ...

        # Shapes:
        # -depth_markers: (batch_size, max_seq_len)
        # -every pred_[tabular], for MLPTabular:  (total_horizons_in_batch, output dim. for that feature)
        # -every pred_[tabular], for LSTMTabular: (batch_size, max_seq_len, output dim. for that feature)
        # -horizon_embedding:
        return depth_markers, tabular_predictions #, horizon_embedding


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
