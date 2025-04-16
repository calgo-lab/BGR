import torch
import torch.nn as nn
import torch.nn.functional as F
from bgr.soil.modelling.image_modules import HDCNNEncoder, ResNetPatchEncoder, PatchCNNEncoder
from bgr.soil.modelling.geotemp_modules import GeoTemporalEncoder
from bgr.soil.modelling.tabulars.tabular_modules import LSTMTabularPredictor, MLPTabularPredictor

class SimpleTabularModel(nn.Module):
    def __init__(self,
        tabular_output_dim_dict : dict[str, int],
        geotemp_input_dim : int,
        segment_encoder_output_dim : int = 512,
        geotemp_output_dim : int = 256,
        patch_size : int = 512,
        rnn_hidden_dim : int = 1024,
        num_lstm_layers : int = 2,
        predefined_random_patches : bool = False
        ):
        super(SimpleTabularModel, self).__init__()
        
        self.tabular_output_dim_dict = tabular_output_dim_dict
        self.predefined_random_patches = predefined_random_patches
        
        if self.predefined_random_patches:
            self.segment_encoder = ResNetPatchEncoder(output_dim=segment_encoder_output_dim, resnet_version='18')
        else:
            self.segment_encoder = PatchCNNEncoder(output_dim=segment_encoder_output_dim, patch_size=patch_size)
        self.geotemp_encoder = GeoTemporalEncoder(geotemp_input_dim, geotemp_output_dim)
        
        self.tabular_predictors = nn.ModuleDict()
        input_dim = segment_encoder_output_dim + geotemp_output_dim
        
        # Create LSTM predictors for each tabular output
        for key, output_dim in tabular_output_dim_dict.items():                        
            self.tabular_predictors[key] = LSTMTabularPredictor(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=rnn_hidden_dim,
                num_lstm_layers=num_lstm_layers
            )
        
    def forward(self, segments: torch.Tensor, geo_temp_features: torch.Tensor):
        if self.predefined_random_patches:
            batch_size, num_segments, num_patches, C, H, W = segments.shape
        else:
            batch_size, num_segments, C, H, W = segments.shape
        
        # Encode each segment individually
        segment_features_list = []
        for i in range(num_segments):
            if self.predefined_random_patches:
                segment_patches = segments[:, i, :, :, :, :] # One additional dimension for the random patches
                segment_features = self.segment_encoder(segment_patches)
            else:
                segment = segments[:, i, :, :, :]
                segment_features = self.segment_encoder(segment)
            segment_features_list.append(segment_features)
        segment_features = torch.stack(segment_features_list, dim=1)
        
        geo_temp_features = self.geotemp_encoder(geo_temp_features)
        
        # Replicate geo_temp_features for each segment
        geo_temp_features = geo_temp_features.unsqueeze(1).repeat(1, num_segments, 1)
        
        # Concatenate segment features with geotemporal features
        combined_features = torch.cat([segment_features, geo_temp_features], dim=-1)
        
        # Pass through LSTM predictors
        tabular_predictions = {}
        for key, predictor in self.tabular_predictors.items():
            tabular_predictions[key] = predictor(combined_features)
        
        return tabular_predictions
    
# DEPRECATED. used for training one tabular at a time
class SegmentToTabular(nn.Module):
    def __init__(self, tab_output_dim, classification=True, stop_token=1.0):
        super(SegmentToTabular, self).__init__()
        #self.segment_encoder = ResNetEncoder(resnet_version='18')
        self.segment_encoder = HDCNNEncoder()

        self.tabular_predictor = MLPTabularPredictor(input_dim=self.segment_encoder.num_img_features,
                                                     output_dim=tab_output_dim, classification=classification)
        self.stop_token = stop_token

    def forward(self, cropped_images):

        seg_features = self.segment_encoder(cropped_images)
        tab_predictions = self.tabular_predictor(seg_features).squeeze()

        return tab_predictions