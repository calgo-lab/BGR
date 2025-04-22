import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor
from bgr.soil.utils import concat_img_geotemp_depth
from bgr.soil.modelling.image_modules import ResNetEncoder, HDCNNEncoder, PatchCNNEncoder, ResNetPatchEncoder
from bgr.soil.modelling.geotemp_modules import GeoTemporalEncoder
from bgr.soil.modelling.depth.depth_modules import LSTMDepthMarkerPredictor
from bgr.soil.modelling.tabulars.tabular_modules import MLPTabularPredictor
from bgr.soil.modelling.horizon.horizon_modules import HorizonEmbedder, HorizonLSTMEmbedder


class SimpleHorizonClassifier(nn.Module):
    def __init__(
        self,
        segment_encoder_output_dim=512,
        patch_size=512,
        num_classes=87
    ):
        super(SimpleHorizonClassifier, self).__init__()
        
        self.segment_encoder = PatchCNNEncoder(patch_size=patch_size, patch_stride=patch_size, output_dim=segment_encoder_output_dim)
        self.classifier = nn.Linear(segment_encoder_output_dim, num_classes)
        
    def forward(self, segments):
        batch_size, num_segments, C, H, W = segments.shape
        
        # Encode each segment individually
        segment_features_list = []
        for i in range(num_segments):
            segment = segments[:, i, :, :, :]
            segment_features = self.segment_encoder(segment)
            segment_features_list.append(segment_features)
        segment_features = torch.stack(segment_features_list, dim=1)
        
        # Classify each segment
        segment_logits = self.classifier(segment_features)
        
        return segment_logits
    
class SimpleHorizonClassifierWithEmbeddings(nn.Module):
    def __init__(
        self,
        patch_size=512,
        segment_encoder_output_dim=512,
        embedding_dim=61,
    ):
        super(SimpleHorizonClassifierWithEmbeddings, self).__init__()
        
        self.segment_encoder = PatchCNNEncoder(patch_size=patch_size, patch_stride=patch_size, output_dim=segment_encoder_output_dim)
        self.horizon_embedder = HorizonEmbedder(input_dim=segment_encoder_output_dim, output_dim=embedding_dim)
    
    def forward(self, segments):
        batch_size, num_segments, C, H, W = segments.shape
        
        # Encode each segment individually
        segment_features_list = []
        for i in range(num_segments):
            segment = segments[:, i, :, :, :]
            segment_features = self.segment_encoder(segment)
            segment_features_list.append(segment_features)
        segment_features = torch.stack(segment_features_list, dim=1)
        
        # Flatten the segment features and geo_temp_features to match the expected input dimensions
        segment_features = segment_features.view(batch_size * num_segments, -1)
        
        # Compute the horizon embeddings
        horizon_embeddings = self.horizon_embedder(segment_features)
        
        return horizon_embeddings

class SimpleHorizonClassifierWithEmbeddingsGeotemps(nn.Module):
    def __init__(
        self,
        geo_temp_input_dim,
        patch_size=512,
        segment_encoder_output_dim=512,
        embedding_dim=61
    ):
        super(SimpleHorizonClassifierWithEmbeddingsGeotemps, self).__init__()
        
        self.segment_encoder = PatchCNNEncoder(patch_size=patch_size, patch_stride=patch_size, output_dim=segment_encoder_output_dim)
        
        self.horizon_embedder = HorizonEmbedder(input_dim=self.segment_encoder.num_img_features + geo_temp_input_dim, output_dim=embedding_dim)
        
    def forward(self, segments, geo_temp_features):
        batch_size, num_segments, C, H, W = segments.shape
        
        # Encode each segment individually
        segment_features_list = []
        for i in range(num_segments):
            segment = segments[:, i, :, :, :]
            segment_features = self.segment_encoder(segment)
            segment_features_list.append(segment_features)
        segment_features = torch.stack(segment_features_list, dim=1)
        
        # Replicate geo_temp_features for each segment
        geo_temp_features = geo_temp_features.unsqueeze(1).repeat(1, num_segments, 1)
        
        # Flatten the segment features and geo_temp_features to match the expected input dimensions
        segment_features = segment_features.view(batch_size * num_segments, -1)
        geo_temp_features = geo_temp_features.view(batch_size * num_segments, -1)
        
        # Concatenate segment features with geotemporal features
        combined_features = torch.cat([segment_features, geo_temp_features], dim=-1)
        
        # Compute the horizon embeddings
        horizon_embeddings = self.horizon_embedder(combined_features)
        
        # Reshape the horizon embeddings back to the original batch and segment dimensions
        horizon_embeddings = horizon_embeddings.view(batch_size, num_segments, -1)
        
        return horizon_embeddings

class SimpleHorizonClassifierWithEmbeddingsGeotempsMLP(nn.Module):
    def __init__(
        self,
        geo_temp_input_dim,
        geo_temp_output_dim=32,
        patch_size=512,
        segment_encoder_output_dim=512,
        embedding_dim=61
    ):
        super(SimpleHorizonClassifierWithEmbeddingsGeotempsMLP, self).__init__()
        
        self.segment_encoder = PatchCNNEncoder(patch_size=patch_size, patch_stride=patch_size, output_dim=segment_encoder_output_dim)
        self.geo_temp_encoder = GeoTemporalEncoder(geo_temp_input_dim, geo_temp_output_dim)
        
        self.horizon_embedder = HorizonEmbedder(input_dim=self.segment_encoder.num_img_features + geo_temp_output_dim, output_dim=embedding_dim)
        
    def forward(self, segments, geo_temp_features):
        batch_size, num_segments, C, H, W = segments.shape
        
        # Encode each segment individually
        segment_features_list = []
        for i in range(num_segments):
            segment = segments[:, i, :, :, :]
            segment_features = self.segment_encoder(segment)
            segment_features_list.append(segment_features)
        segment_features = torch.stack(segment_features_list, dim=1)
        
        geo_temp_features = self.geo_temp_encoder(geo_temp_features)
        
        # Replicate geo_temp_features for each segment
        geo_temp_features = geo_temp_features.unsqueeze(1).repeat(1, num_segments, 1)
        
        # Flatten the segment features and geo_temp_features to match the expected input dimensions
        segment_features = segment_features.view(batch_size * num_segments, -1)
        geo_temp_features = geo_temp_features.view(batch_size * num_segments, -1)
        
        # Concatenate segment features with geotemporal features
        combined_features = torch.cat([segment_features, geo_temp_features], dim=-1)
        
        # Compute the horizon embeddings
        horizon_embeddings = self.horizon_embedder(combined_features)
        
        # Reshape the horizon embeddings back to the original batch and segment dimensions
        horizon_embeddings = horizon_embeddings.view(batch_size, num_segments, -1)
        
        return horizon_embeddings

class SimpleHorizonClassifierWithEmbeddingsGeotempsMLPTabMLP(nn.Module):
    def __init__(
        self,
        geo_temp_input_dim,
        segments_tabular_input_dim,
        segment_encoder_output_dim=512,
        segments_tabular_output_dim=64,
        geo_temp_output_dim=64,
        patch_size=512,
        embedding_dim=61,
        embed_horizons_linearly=True,
        predefined_random_patches=False
    ):
        super(SimpleHorizonClassifierWithEmbeddingsGeotempsMLPTabMLP, self).__init__()
        
        self.embed_horizons_linearly = embed_horizons_linearly
        self.predefined_random_patches = predefined_random_patches
        
        if self.predefined_random_patches:
            self.segment_encoder = ResNetPatchEncoder(output_dim=segment_encoder_output_dim,resnet_version='18')
        else:
            self.segment_encoder = PatchCNNEncoder(patch_size=patch_size, patch_stride=patch_size, output_dim=segment_encoder_output_dim)
        self.geo_temp_encoder = GeoTemporalEncoder(geo_temp_input_dim, geo_temp_output_dim)
        
        # Simple tabular encoder for the segment-specific tabular features
        self.segments_tabular_encoder = nn.Sequential(
            nn.Linear(segments_tabular_input_dim, segments_tabular_output_dim),
            nn.ReLU()
        )
        
        # Choose between the MLP and LSTM horizon embedder
        if embed_horizons_linearly:
            self.horizon_embedder = HorizonEmbedder(input_dim=self.segment_encoder.num_img_features + geo_temp_output_dim + segments_tabular_output_dim, output_dim=embedding_dim)
        else:
            self.horizon_embedder = HorizonLSTMEmbedder(input_dim=self.segment_encoder.num_img_features + geo_temp_output_dim + segments_tabular_output_dim, output_dim=embedding_dim, hidden_dim=256)
        
    def forward(self, segments, segments_tabular_features, geo_temp_features):
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
        
        segments_tabular_features = self.segments_tabular_encoder(segments_tabular_features)
        
        # Concatenate segment features with segment-specific tabular features
        segment_features = torch.cat([segment_features, segments_tabular_features], dim=-1)
        
        geo_temp_features = self.geo_temp_encoder(geo_temp_features)
        
        # Replicate geo_temp_features for each segment
        geo_temp_features = geo_temp_features.unsqueeze(1).repeat(1, num_segments, 1)
        
        # Flatten the segment features and geo_temp_features to match the expected input dimensions
        segment_features = segment_features.view(batch_size * num_segments, -1)
        geo_temp_features = geo_temp_features.view(batch_size * num_segments, -1)
        
        # Concatenate segment features with geotemporal features
        combined_features = torch.cat([segment_features, geo_temp_features], dim=-1)
        
        # Compute the horizon embeddings
        if self.embed_horizons_linearly:
            # Embeddings are returned one by one
            horizon_embeddings = self.horizon_embedder(combined_features)
            # Reshape the horizon embeddings back to the original batch and segment dimensions
            horizon_embeddings = horizon_embeddings.view(batch_size, num_segments, -1)
        else:
            # Embeddings are returned all at once (for each sample)   
            horizon_embeddings = self.horizon_embedder(combined_features, num_segments)
        
        return horizon_embeddings
    
    
class SimpleHorizonClassifierWithEmbeddingsGeotempsMLPTabMLPHybrid(nn.Module):
    """Predicts horizon embeddings, then projects them again linearly. The embeddings are fed into the cosine loss,
    the final output in cross entropy.
    """
    def __init__(
        self,
        geo_temp_input_dim,
        segments_tabular_input_dim,
        segment_encoder_output_dim=512,
        segments_tabular_output_dim=64,
        geo_temp_output_dim=64,
        patch_size=512,
        embedding_dim=61,
        num_classes=87,
        embed_horizons_linearly=True,
        predefined_random_patches=False
    ):
        super(SimpleHorizonClassifierWithEmbeddingsGeotempsMLPTabMLPHybrid, self).__init__()
        
        self.embed_horizons_linearly = embed_horizons_linearly
        self.predefined_random_patches = predefined_random_patches
        
        if self.predefined_random_patches:
            self.segment_encoder = ResNetPatchEncoder(output_dim=segment_encoder_output_dim,resnet_version='18')
        else:
            self.segment_encoder = PatchCNNEncoder(patch_size=patch_size, patch_stride=patch_size, output_dim=segment_encoder_output_dim)
        self.geo_temp_encoder = GeoTemporalEncoder(geo_temp_input_dim, geo_temp_output_dim)
        
        # Simple tabular encoder for the segment-specific tabular features
        self.segments_tabular_encoder = nn.Sequential(
            nn.Linear(segments_tabular_input_dim, segments_tabular_output_dim),
            nn.ReLU()
        )
        
        # Choose between the MLP and LSTM horizon embedder
        if embed_horizons_linearly:
            self.horizon_embedder = HorizonEmbedder(input_dim=self.segment_encoder.num_img_features + geo_temp_output_dim + segments_tabular_output_dim, output_dim=embedding_dim)
        else:
            self.horizon_embedder = HorizonLSTMEmbedder(input_dim=self.segment_encoder.num_img_features + geo_temp_output_dim + segments_tabular_output_dim, output_dim=embedding_dim, hidden_dim=256)
        
        # Final Dense layer for class logits
        self.final_fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, segments, segments_tabular_features, geo_temp_features):
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
        
        segments_tabular_features = self.segments_tabular_encoder(segments_tabular_features)
        
        # Concatenate segment features with segment-specific tabular features
        segment_features = torch.cat([segment_features, segments_tabular_features], dim=-1)
        
        geo_temp_features = self.geo_temp_encoder(geo_temp_features)
        
        # Replicate geo_temp_features for each segment
        geo_temp_features = geo_temp_features.unsqueeze(1).repeat(1, num_segments, 1)
        
        # Flatten the segment features and geo_temp_features to match the expected input dimensions
        segment_features = segment_features.view(batch_size * num_segments, -1)
        geo_temp_features = geo_temp_features.view(batch_size * num_segments, -1)
        
        # Concatenate segment features with geotemporal features
        combined_features = torch.cat([segment_features, geo_temp_features], dim=-1)
        
        # Compute the horizon embeddings
        if self.embed_horizons_linearly:
            # Embeddings are returned one by one
            horizon_embeddings = self.horizon_embedder(combined_features)
            # Reshape the horizon embeddings back to the original batch and segment dimensions
            horizon_embeddings = horizon_embeddings.view(batch_size, num_segments, -1)
        else:
            # Embeddings are returned all at once (for each sample)   
            horizon_embeddings = self.horizon_embedder(combined_features, num_segments)
            
        # Project emebddings onto logits
        class_logits = self.final_fc(horizon_embeddings)
        
        return horizon_embeddings, class_logits
    
# DEPRECATED: used for initial testing
class ImageTabularModel(nn.Module):
    """
    Simple baseline model that combines ViT image features with tabular features
    """
    def __init__(self, vision_backbone, num_tabular_features, num_classes):
        super(ImageTabularModel, self).__init__()

        self.vision_backbone = vision_backbone
        # Load pretrained DINOv2-Model from Hugging Face or ResNet
        if vision_backbone:
            self.image_encoder = AutoModel.from_pretrained(vision_backbone)
            #self.feature_extractor = AutoFeatureExtractor.from_pretrained(vision_backbone)
            num_img_features = self.image_encoder.config.hidden_size
        else:
            self.image_encoder = ResNetEncoder(resnet_version='18')
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
