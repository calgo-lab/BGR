import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor
from bgr.soil.utils import concat_img_geotemp_depth
from bgr.soil.modelling.image_encoders import ImageEncoder
from bgr.soil.modelling.geotemp_encoders import GeoTemporalEncoder
from bgr.soil.modelling.depth_markers import LSTMDepthMarkerPredictor
from bgr.soil.modelling.tabular_predictors import MLPTabularPredictor, LSTMTabularPredictor
from bgr.soil.modelling.horizon_embedders import HorizonEmbedder


class SegmentToTabular(nn.Module):
    def __init__(self, tab_output_dim, classification=True, stop_token=1.0):
        super(SegmentToTabular, self).__init__()
        self.segment_encoder = ImageEncoder(resnet_version='18')

        self.tabular_predictor = MLPTabularPredictor(input_dim=self.segment_encoder.num_img_features,
                                                     output_dim=tab_output_dim, classification=classification)
        self.stop_token = stop_token

    def forward(self, cropped_images):

        seg_features = self.segment_encoder(cropped_images)
        tab_predictions = self.tabular_predictor(seg_features)

        return tab_predictions

class HorizonSegmenter(nn.Module):
    def __init__(self,
                 geo_temp_input_dim, geo_temp_output_dim=32, # params for geotemp encoder
                 max_seq_len=10, stop_token=1.0,  # params for depth predictor (any class)
                 rnn_hidden_dim=256  # params for LSTMDepthPredictor
                 ):
        super(HorizonSegmenter, self).__init__()
        self.stop_token = stop_token
        self.image_encoder = ImageEncoder(resnet_version='18')
        self.geo_temp_encoder = GeoTemporalEncoder(geo_temp_input_dim, geo_temp_output_dim)

        # Choose from different depth predictors
        #self.depth_marker_predictor = TransformerDepthMarkerPredictor(self.image_encoder.num_img_features + geo_temp_output_dim,
        #                                                              transformer_dim, num_transformer_heads, num_transformer_layers,
        #                                                              max_seq_len, stop_token)
        self.depth_marker_predictor = LSTMDepthMarkerPredictor(self.image_encoder.num_img_features + geo_temp_output_dim,
                                                               rnn_hidden_dim, max_seq_len, stop_token)


    def forward(self, images, geo_temp):
        # Extract image + geotemp features, then concatenate them
        image_features = self.image_encoder(images)
        geo_temp_features = self.geo_temp_encoder(geo_temp)
        img_geotemp_vector = torch.cat([image_features, geo_temp_features], dim=-1)

        # Predict depth markers based on concatenated vector
        depth_markers = self.depth_marker_predictor(img_geotemp_vector)

        return depth_markers


class HorizonClassifier(nn.Module):
    def __init__(self,
                 geo_temp_input_dim, geo_temp_output_dim=32, # params for geotemp encoder
                 max_seq_len=10, stop_token=1.0, # params for depth predictor (any class)
                 #transformer_dim=128, num_transformer_heads=4, num_transformer_layers=2, # params for Transformer- or CrossAttentionDepthPredictor
                 rnn_hidden_dim=256, # params for LSTMDepthPredictor
                 tabular_predictors_dict={}, # params for the tabular predictors
                 embedding_dim=64,
                 device='cpu'
                 ):
        super(HorizonClassifier, self).__init__()
        self.stop_token = stop_token
        self.device = device
        self.image_encoder = ImageEncoder(resnet_version='18')
        self.geo_temp_encoder = GeoTemporalEncoder(geo_temp_input_dim, geo_temp_output_dim)

        # Choose from different depth predictors
        #self.depth_marker_predictor = TransformerDepthMarkerPredictor(self.image_encoder.num_img_features + geo_temp_output_dim,
        #                                                              transformer_dim, num_transformer_heads, num_transformer_layers,
        #                                                              max_seq_len, stop_token)
        self.depth_marker_predictor = LSTMDepthMarkerPredictor(self.image_encoder.num_img_features + geo_temp_output_dim,
                                                               rnn_hidden_dim, max_seq_len, stop_token)

        self.segment_encoder = ImageEncoder(resnet_version='18') # after predicting the depths, the original image is cropped and fed into another vision model

        # Define list of tabular predictors
        # Each takes as input the image_geotemp_vector extended the feature vector from the horizon segments
        dim_input_tab = self.image_encoder.num_img_features + geo_temp_output_dim + self.segment_encoder.num_img_features
        dim_output_all_tabs = 0
        self.tabular_predictors = nn.ModuleList()
        for tab_pred_name in tabular_predictors_dict:
            dim_output_tab = tabular_predictors_dict[tab_pred_name]['output_dim']
            dim_output_all_tabs += dim_output_tab # needed for constructing the horizon embedder below
            tab_classif = tabular_predictors_dict[tab_pred_name]['classification']
            self.tabular_predictors.append(MLPTabularPredictor(input_dim=dim_input_tab,
                                                               output_dim=dim_output_tab,
                                                               classification=tab_classif,
                                                               name=tab_pred_name).to(self.device))

        self.horizon_embedder = HorizonEmbedder(input_dim=dim_input_tab + dim_output_all_tabs,
                                                output_dim=embedding_dim)

    def forward(self, images, geo_temp, true_depths=None):
        # Extract image + geotemp features, then concatenate them
        image_features = self.image_encoder(images)
        geo_temp_features = self.geo_temp_encoder(geo_temp)
        img_geotemp_vector = torch.cat([image_features, geo_temp_features], dim=-1)

        # Predict depth markers based on concatenated vector
        depth_markers = self.depth_marker_predictor(img_geotemp_vector)

        # Crop images, extract visual features, then concatenate with img_geotemp_vec for every horizon
        batch_size, C, H, W = images.shape
        img_geotemp_seg_vector = []
        for i in range(batch_size):
            image = images[i]  # (C, H, W)
            depths = true_depths[i] if true_depths else depth_markers[i].tolist() # true depths for training, pred. depths for inference
            # Stop at the first occurrence of stop_token (inclusive)
            if self.stop_token in depths:
                depths = depths[:depths.index(self.stop_token) + 1]

            # Convert normalized depth markers to pixel indices
            pixel_depths = [int(d * H) for d in [0.0] + depths]  # Add 0.0 for upmost bound

            for j in range(len(pixel_depths) - 1):
                upper, lower = pixel_depths[j], pixel_depths[j + 1]
                cropped = image[:, upper:lower, :]  # Crop along the height axis
                cropped_resized = F.interpolate(cropped.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
                seg_features = self.segment_encoder(cropped_resized) # apply second image encoder
                img_geotemp_seg_vector.append( torch.cat([img_geotemp_vector[i], seg_features.squeeze(0)]) )
        img_geotemp_seg_vector = torch.stack(img_geotemp_seg_vector)

        # Note: for every feature the output of the tab_predictor has a different dimension
        tabular_predictions = {}
        img_geotemp_seg_tab_vector = img_geotemp_seg_vector
        for tab_predictor in self.tabular_predictors:
            tab_pred = tab_predictor(img_geotemp_seg_vector)
            tabular_predictions[tab_predictor.name] = tab_pred.squeeze()

            # Concatenate img_geotemp_seg_vector with predictions for current tab. feature
            img_geotemp_seg_tab_vector = torch.cat([img_geotemp_seg_tab_vector, tab_pred], dim=-1)

        # Compute horizon embedding from the final concatenated vector
        horizon_embedding = self.horizon_embedder(img_geotemp_seg_tab_vector)

        # Shapes:
        # -depth_markers: (batch_size, max_seq_len)
        # -every pred_[tabular]: (total_horizons_in_batch, output dim. for that feature)
        # -every horizon_embedding: (total_horizons_in_batch, embedding_dim)
        return depth_markers, tabular_predictions, horizon_embedding


class ImageTabularModel(nn.Module):
    """
    Simple baseline model that combines ViT image features with tabular features
    """
    def __init__(self, vision_backbone, num_tabular_features, num_classes):
        super(ImageTabularModel, self).__init__()

        # Load pretrained DINOv2-Model from Hugging Face or ResNet
        if vision_backbone:
            self.image_encoder = AutoModel.from_pretrained(vision_backbone)
            #self.feature_extractor = AutoFeatureExtractor.from_pretrained(vision_backbone)
            num_img_features = self.image_encoder.config.hidden_size
        else:
            self.image_encoder = ImageEncoder(resnet_version='18')
            num_img_features = self.image_encoder.num_img_features

        # MLP for the tabular data
        dim_geotemp_output = 256
        self.fc_tabular = GeoTemporalEncoder(num_tabular_features, dim_geotemp_output)

        # Combined Fully Connected Layers
        self.fc_combined = HorizonEmbedder(input_dim=num_img_features + dim_geotemp_output, output_dim=num_classes)

    def forward(self, image, tabular_features):
        # Extract image features
        if self.vision_backbone:
            image_features = self.image_encoder(pixel_values=image).last_hidden_state[:, 0, :] # Use [CLS] token representation
        else:
            image_features = self.image_encoder(image)

        # Process tabular features with the MLP
        tabular_features_processed = self.fc_tabular(tabular_features)

        # Combine image and tabular features
        combined_features = torch.cat((image_features, tabular_features_processed), dim=1)

        # Final prediction
        output = self.fc_combined(combined_features)
        return output
