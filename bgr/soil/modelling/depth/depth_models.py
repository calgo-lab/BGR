import torch
import torch.nn as nn
from bgr.soil.modelling.depth.depth_modules import LSTMDepthMarkerPredictor, CrossAttentionTransformerDepthMarkerPredictor
from bgr.soil.modelling.geotemp_modules import GeoTemporalEncoder
from bgr.soil.modelling.image_modules import PatchCNNEncoder, ResNetPatchEncoder

class SimpleDepthModel(nn.Module):
    def __init__(self,
                geo_temp_input_dim : int, 
                geo_temp_output_dim : int = 256, # params for geotemp encoder
                image_encoder_output_dim : int = 512,
                max_seq_len : int = 10, 
                stop_token : float = 1.0,
                rnn_hidden_dim : int = 256,  # params for LSTMDepthPredictor
                patch_size : int = 512,
                predefined_random_patches : bool = False
                ):
        super(SimpleDepthModel, self).__init__()
        
        self.stop_token = stop_token
        self.predefined_random_patches = predefined_random_patches
        
        if self.predefined_random_patches:
            self.image_encoder = ResNetPatchEncoder(output_dim=image_encoder_output_dim, resnet_version='18')
        else:
            self.image_encoder = PatchCNNEncoder(output_dim=image_encoder_output_dim, patch_size=patch_size, patch_stride=patch_size)
        self.geo_temp_encoder = GeoTemporalEncoder(geo_temp_input_dim, geo_temp_output_dim)

        self.depth_marker_predictor = LSTMDepthMarkerPredictor(image_encoder_output_dim + geo_temp_output_dim, rnn_hidden_dim, max_seq_len, stop_token)

    def forward(self, images, geo_temp):
        # Extract image + geotemp features, then concatenate them
        image_features = self.image_encoder(images)
        geo_temp_features = self.geo_temp_encoder(geo_temp)
        img_geotemp_vector = torch.cat([image_features, geo_temp_features], dim=-1)

        # Predict depth markers based on concatenated vector
        depth_markers = self.depth_marker_predictor(img_geotemp_vector)

        return depth_markers
    
    
class SimpleDepthModelCrossAttention(nn.Module):
    def __init__(self,
                geo_temp_input_dim : int, 
                geo_temp_output_dim : int = 32, # params for geotemp encoder
                image_encoder_output_dim : int = 512,
                max_seq_len : int = 10, 
                stop_token : float = 1.0,  # params for depth predictor (any class)
                decoder_hidden_dim : int = 256, 
                decoder_num_heads : int = 8,
                decoder_num_layers : int = 2,
                patch_size : int = 512,
                predefined_random_patches : bool = False
                ):
        super(SimpleDepthModelCrossAttention, self).__init__()
        
        self.stop_token = stop_token
        self.predefined_random_patches = predefined_random_patches
        
        if self.predefined_random_patches:
            self.image_encoder = ResNetPatchEncoder(output_dim=image_encoder_output_dim, resnet_version='18')
        else:
            self.image_encoder = PatchCNNEncoder(output_dim=image_encoder_output_dim, patch_size=patch_size, patch_stride=patch_size)
        self.geo_temp_encoder = GeoTemporalEncoder(geo_temp_input_dim, geo_temp_output_dim)

        self.depth_marker_predictor = CrossAttentionTransformerDepthMarkerPredictor(image_encoder_output_dim + geo_temp_output_dim,
                                                                                    decoder_hidden_dim, max_seq_len, stop_token,
                                                                                    decoder_num_heads, decoder_num_layers)

    def forward(self, images, geo_temp):
        # Extract image + geotemp features, then concatenate them
        image_features = self.image_encoder(images)
        geo_temp_features = self.geo_temp_encoder(geo_temp)
        img_geotemp_vector = torch.cat([image_features, geo_temp_features], dim=-1)

        # Predict depth markers based on concatenated vector
        depth_markers = self.depth_marker_predictor(img_geotemp_vector)

        return depth_markers