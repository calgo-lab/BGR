import torch
import torch.nn as nn
import torch.nn.functional as F
from bgr.soil.modelling.depth.depth_modules import LSTMDepthMarkerPredictor
from bgr.soil.modelling.geotemp_modules import GeoTemporalEncoder
from bgr.soil.modelling.horizon.horizon_modules import HorizonEmbedder, HorizonLSTMEmbedder
from bgr.soil.modelling.image_modules import PatchCNNEncoder, ResNetEncoder, ResNetPatchEncoder
from bgr.soil.modelling.tabulars.tabular_modules import LSTMTabularPredictor, MLPTabularPredictor


class SoilNet_LSTM(nn.Module):
    """End-to-end model predicting depth markers, tabular features and horizon labels.
    All task models are LSTM-based.
    """
    def __init__(
        self,
        # Parameters for image encoder, geotemp encoder and depth predictor:
        geo_temp_input_dim : int, 
        geo_temp_output_dim : int = 256, # params for geotemp encoder
        image_encoder_output_dim : int = 512,  # may be different from the segment encoder output dim below
        max_seq_len : int = 10, 
        stop_token : float = 1.0,
        depth_rnn_hidden_dim : int = 256, # params for LSTMDepthPredictor
        #depth_num_lstm_layers : int = 2, # should we allow the user to select this?
        img_patch_size : int = 512,  # may be different from segment patch sizes below
        predefined_random_patches : bool = False, # currently used for both img and seg encoders

        # Parameters for tabular predictors:
        tabular_output_dim_dict : dict[str, int] = {}, # name_tabular: output_dim
        segment_encoder_output_dim : int = 512,
        seg_patch_size : int = 512,
        tab_rnn_hidden_dim : int = 1024,
        tab_num_lstm_layers : int = 2,
        
        # Parameters for horizon predictor:
        #segments_tabular_input_dim, # not necessary, derived from tabular_output_dim_dict
        segments_tabular_output_dim : int = 64, # for the final MLP before entering the horizon predictor
        #hor_rnn_hidden_dim : int = 256, # should we allow the user to select this?
        #hor_num_lstm_layers : int = 2, # should we allow the user to select this?
        embedding_dim : int = 61,
        #embed_horizons_linearly : bool = True # will always be true, since only LSTMs are used here
    ):
        super(SoilNet_LSTM, self).__init__()
        
        ### Set attributes ###
        self.geo_temp_input_dim = geo_temp_input_dim
        self.geo_temp_output_dim = geo_temp_output_dim
        self.image_encoder_output_dim = image_encoder_output_dim
        self.max_seq_len = max_seq_len
        self.stop_token = stop_token
        self.depth_rnn_hidden_dim = depth_rnn_hidden_dim
        self.img_patch_size = img_patch_size
        self.predefined_random_patches = predefined_random_patches
        self.tabular_output_dim_dict = tabular_output_dim_dict
        self.segment_encoder_output_dim = segment_encoder_output_dim
        self.seg_patch_size = seg_patch_size
        self.tab_rnn_hidden_dim = tab_rnn_hidden_dim
        self.tab_num_lstm_layers = tab_num_lstm_layers
        self.segments_tabular_output_dim = segments_tabular_output_dim
        self.embedding_dim = embedding_dim
        
        ### Define modules and task models ###
        # Image and segment encoders
        if self.predefined_random_patches:
            self.image_encoder   = ResNetPatchEncoder(output_dim=self.image_encoder_output_dim,   resnet_version='18')
            self.segment_encoder = ResNetPatchEncoder(output_dim=self.segment_encoder_output_dim, resnet_version='18')
        else:
            self.image_encoder   = PatchCNNEncoder(patch_size=img_patch_size, patch_stride=img_patch_size, output_dim=image_encoder_output_dim)
            self.segment_encoder = PatchCNNEncoder(patch_size=seg_patch_size, patch_stride=seg_patch_size, output_dim=segment_encoder_output_dim)
            
        # Geotemp encoder
        self.geo_temp_encoder = GeoTemporalEncoder(self.geo_temp_input_dim, self.geo_temp_output_dim)
        
        # Depth marker predictor
        self.depth_marker_predictor = LSTMDepthMarkerPredictor(self.image_encoder_output_dim + self.geo_temp_output_dim, self.depth_rnn_hidden_dim, self.max_seq_len, self.stop_token)
        
        # Tabular predictors (one for each tabular)
        self.tabular_predictors = nn.ModuleDict()
        segments_tabular_input_dim = 0 # sum of all tabular output dims (needed for the extra MLP layer below)
        for key, output_dim in self.tabular_output_dim_dict.items():                        
            self.tabular_predictors[key] = LSTMTabularPredictor(
                input_dim = self.segment_encoder_output_dim + self.geotemp_output_dim, # all get the same concatenated input segment_geotemp vector
                output_dim = output_dim, # each has a different output dim, depending on how many classes it predicts
                hidden_dim = self.tab_rnn_hidden_dim,
                num_lstm_layers = self.tab_num_lstm_layers
            )
            segments_tabular_input_dim += output_dim
            
        # Extra linear layer for the tabular predictions
        self.segments_tabular_encoder = nn.Sequential(
            nn.Linear(segments_tabular_input_dim, self.segments_tabular_output_dim),
            nn.ReLU()
        )    
        
        # Horizon predictor
        # Choose between the MLP (later with cross entropy loss) and LSTM horizon embedder
        # Takes the fully concatenated segment_geotemp_tabular vector as input
        self.horizon_embedder = HorizonLSTMEmbedder(input_dim = self.segment_encoder_output_dim + self.geo_temp_output_dim + self.segments_tabular_output_dim, 
                                                    output_dim = self.embedding_dim, hidden_dim = 256)
        
    def forward(self, image, segments, geo_temp_features, tabular_features, true_depths=True, true_tabs=True):
        # TODO: Can we rewrite this method in the form:
        # depth_markers = SimpleDepthModel(...).forward(image, geo_temp_features)
        # tabular_predictions = SimpleTabularModel(...).forward(segments, geo_temp_features)
        # horizon_embedding = SimpleHorizonClassifierWithEmbeddingsGeotempsMLPTabMLP(...).forward(segments, segments_tabular_features, geo_temp_features)
        # Right now, every task model extracts again image/segment features and geotemp features. 
        # Here we only need to extract them once and reuse them in subsequent tasks.
        # Maybe modify the task models to accept pretrained image/segment/geotemp features as input?
        
        # Extract image + geotemp features, then concatenate them
        image_features = self.image_encoder(image)
        geo_temp_features = self.geo_temp_encoder(geo_temp_features)
        img_geotemp_vector = torch.cat([image_features, geo_temp_features], dim=-1)

        ### TASK 1: Predict depth markers based on concatenated vector
        depth_markers = self.depth_marker_predictor(img_geotemp_vector)
        
        ### TASK 2: Predict tabular features for each segment
        # TODO: Currently, we are always using the true segments for the subsequent tasks
        #       Include true_depths=False to use the predicted depths to segment the image at inference time
        
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
        
        # Replicate geo_temp_features for each segment
        geo_temp_features = geo_temp_features.unsqueeze(1).repeat(1, num_segments, 1)
        
        # Concatenate segment features with geotemporal features
        segment_geotemp_features = torch.cat([segment_features, geo_temp_features], dim=-1)
        
        # Pass through LSTM predictors
        tabular_predictions = {}
        for key, predictor in self.tabular_predictors.items():
            tabular_predictions[key] = predictor(segment_geotemp_features)
        
        ### TASK 3: Compute horizon embedding from the final concatenated vector
        # TODO: Currently, we are always using the true tabulars for the final task
        #       Include true_tabs=False to use the predicted tabulars to compute the horizon embedding at inference time
        
        # Extra MLP for the tabular predictions before entering the horizon predictor
        tabular_features = self.segments_tabular_encoder(tabular_features)
        
        # Concatenate segment features with tabular features
        segment_tabular_features = torch.cat([segment_features, tabular_features], dim=-1)
        
        # Flatten to match the expected input dimensions
        segment_tabular_features = segment_tabular_features.view(batch_size * num_segments, -1)
        geo_temp_features = geo_temp_features.view(batch_size * num_segments, -1)
        
        # Concatenate segment features with geotemporal features
        segment_tabular_geotemp_features = torch.cat([segment_tabular_features, geo_temp_features], dim=-1)
        
        # Compute the horizon embeddings
        # Embeddings are returned all at once (for each sample)   
        horizon_embeddings = self.horizon_embedder(segment_tabular_geotemp_features, num_segments)
        
        return depth_markers, tabular_predictions, horizon_embeddings


# DEPRECATED:
class HorizonClassifier(nn.Module):
    def __init__(
        self,
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
        self.image_encoder = ResNetEncoder(resnet_version='18')
        self.geo_temp_encoder = GeoTemporalEncoder(geo_temp_input_dim, geo_temp_output_dim)

        # Choose from different depth predictors
        #self.depth_marker_predictor = TransformerDepthMarkerPredictor(self.image_encoder.num_img_features + geo_temp_output_dim,
        #                                                              transformer_dim, num_transformer_heads, num_transformer_layers,
        #                                                              max_seq_len, stop_token)
        self.depth_marker_predictor = LSTMDepthMarkerPredictor(self.image_encoder.num_img_features + geo_temp_output_dim, rnn_hidden_dim, max_seq_len, stop_token)

        self.segment_encoder = ResNetEncoder(resnet_version='18') # after predicting the depths, the original image is cropped and fed into another vision model

        # Define list of tabular predictors
        # Each takes as input the image_geotemp_vector extended the feature vector from the horizon segments
        dim_input_tab = self.image_encoder.num_img_features + geo_temp_output_dim + self.segment_encoder.num_img_features
        dim_output_all_tabs = 0
        self.tabular_predictors = nn.ModuleList()
        for tab_pred_name in tabular_predictors_dict:
            dim_output_tab = tabular_predictors_dict[tab_pred_name]['output_dim']
            dim_output_all_tabs += dim_output_tab # needed for constructing the horizon embedder below
            tab_classif = tabular_predictors_dict[tab_pred_name]['classification']
            self.tabular_predictors.append(MLPTabularPredictor(
                input_dim=dim_input_tab,
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